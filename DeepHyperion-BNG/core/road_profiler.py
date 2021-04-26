#
# Compute a path given the shape of the road and uses beamNGpy set Script
#
from shapely.geometry import Point, LineString, Polygon
import numpy as np
import sys
import logging as l
import math


def log_exception(extype, value, trace):
    l.exception('Uncaught exception:', exc_info=(extype, value, trace))


def setup_logging(level=l.INFO):
    term_handler = l.StreamHandler()
    l.basicConfig(format='Driver AI: %(asctime)s %(levelname)-8s %(message)s',
                  level=level, handlers=[term_handler])
    sys.excepthook = log_exception


def pairs(lst):
    for i in range(1, len(lst)):
        yield lst[i - 1], lst[i]


def dot(a, b):
    return np.sum(a * b, axis=-1)


def get_poligon_from_geometry(geometry):
    left_edge_x = np.array([e['left'][0] for e in geometry])
    left_edge_y = np.array([e['left'][1] for e in geometry])
    right_edge_x = np.array([e['right'][0] for e in geometry])
    right_edge_y = np.array([e['right'][1] for e in geometry])

    # Note that one must be in reverse order for the polygon to close
    right_edge = LineString(zip(right_edge_x[::-1], right_edge_y[::-1]))
    left_edge = LineString(zip(left_edge_x, left_edge_y))

    l_edge = left_edge.coords
    r_edge = right_edge.coords

    return Polygon(list(l_edge) + list(r_edge))

class RoadSegment:
    start = None
    end = None

    def __init__(self, length, max_speed, curvature, geometry, direction):
        self.length = length
        self.max_speed = max_speed
        self.curvature = curvature
        # TODO Straight have no direction
        self.direction = direction
        # Store the geometry of this segment
        self.geometry = geometry
        # Create a Polygon from the geometry
        self.polygon = get_poligon_from_geometry(geometry)


class Road:
    _LIMIT = 100

    def __init__(self, road_segments):
        self.road_segments = []

        self.total_length = sum(int(rs.length) for rs in road_segments)

        # Define road segments start and end
        # This is over x-axis only
        initial_position = 0

        for rs in road_segments:

            # l.debug("Consider: %s, %s", rs.length, rs.max_speed)

            # # TODO: FIXME: THIS IS NOT OK. Same curvature but in different direction results in the same max speed !
            # if len(self.road_segments)> 0 and self.road_segments[-1].max_speed == rs.max_speed:
            #     # We need to merge the two consecutive segments to form one
            #
            #     # Compute start position is the initial position of the previous [-1]
            #     start = initial_position - self.road_segments[-1].length
            #     end = start + self.road_segments[-1].length + rs.length
            #     # NOTE: to have exactly the same speed the segments have the same curvature !
            #     curvature = self.road_segments[-1].curvature
            #     merged_length = end - start
            #
            #     # Create the merged segment. We need to
            #     the_road_segment = RoadSegment(merged_length, rs.max_speed, curvature, self.road_segments[-1].geometry + rs.geometry)
            #
            #     # Replace the road segment in the road
            #     self.road_segments[-1] = the_road_segment
            #
            #     l.debug("Merged Segments %s and %s ", self.road_segments[-1], the_road_segment)
            # else:
            #     # Compute start and end
            #     start = initial_position
            #     end = start + rs.length
            #     curvature = rs.curvature
            #
            #     the_road_segment = RoadSegment(rs.length, rs.max_speed, curvature, rs.geometry)
            #
            #     # Store the road segment in the road
            #     self.road_segments.append(the_road_segment)

            start = initial_position
            end = start + rs.length

            # TODO Maybe clone would have worked as well?
            the_road_segment = RoadSegment(rs.length, rs.max_speed, rs.curvature, rs.geometry, rs.direction)

            # Store the road segment in the road
            self.road_segments.append(the_road_segment)

            # No matter what we update the last element in the list
            self.road_segments[-1].start = start
            self.road_segments[-1].end = end

            # Update loop variable
            initial_position = end


    def get_segment_at_position(self, point):
        for segment in self.road_segments:
            if segment.polygon.contains(point):
                return segment

        return None

    # Utility methods
    @staticmethod
    def _compute_acceleration_line(starting_point: Point, delta_v):
        # y1 = y0 + A * (x1 - x0)
        x0 = starting_point.x
        y0 = starting_point.y
        # Not sure why here is 50 ?
        x1 = x0 + 200
        y1 = y0 + delta_v * (x1 - x0)
        ending_point = Point(x1, y1)
        # Add a little epsilon here...
        x2 = x0 - 0.01
        y2 = y0 + delta_v * (x2 - x0)
        # Replace the original starting_point with the new one
        starting_point = Point(x2, y2)
        return LineString([starting_point, ending_point])

    @staticmethod
    def _compute_deceleration_line(starting_point: Point, delta_v):
        # y1 = y0 + A * (x1 - x0)
        x0 = starting_point.x
        y0 = starting_point.y
        x1 = x0 - 200
        y1 = y0 + delta_v * (x1 - x0)
        ending_point = Point(x1, y1)
        # Add a little epsilon here...
        x2 = x0 + 0.01
        y2 = y0 + delta_v * (x2 - x0)
        # Replace the original starting_point with the new one
        starting_point = Point(x2, y2)
        return LineString([starting_point, ending_point])

    # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    @staticmethod
    def _line(the_line_segment):
        """ A * x + B * y = C """

        assert len(list(zip(*the_line_segment.coords.xy))) == 2, "The line segment" + str(the_line_segment) + "is NOT defined by 2 points!"

        p1 = list(zip(*the_line_segment.coords.xy))[0]
        p2 = list(zip(*the_line_segment.coords.xy))[1]
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])

        return A, B, -C

    @staticmethod
    def _intersection(L1, L2):
        " Given two lines returns whether the lines intersect (x,y) or not "
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return Point(x, y)
        else:
            return False

    @staticmethod
    # Return the line, the distance, the new intersection point
    # Valid intersection points must be "after" (on the right than the input the_intersection_point
    def _find_closest_intersecting_line(the_line, the_intersection_point, the_other_lines):

        # TODO It might happen that in the same point more than two lines pass?
        # intersecting_lines = []
        closest_intersection_point_and_line = (Point(math.inf,0),  None)

        # This should work in the assumption the_line is defined only by two points, which should be the case for us !
        L1 = Road._line(the_line)

        for the_other_line in the_other_lines:
            L2 = Road._line(the_other_line)
            # Can be improved. Those are the first and last point that define that line
            p1 = Point(list(zip(*the_other_line.coords.xy))[0])
            p2 = Point(list(zip(*the_other_line.coords.xy))[1])

            R = Road._intersection(L1, L2)

            if R:
                # Since we are considering the lines that extend the segments for checking the intersection
                #   we must ensure that we are not intersecting OUTSIDE the segments

                if the_intersection_point.x < R.x <= closest_intersection_point_and_line[0].x and \
                        (p1.x <= R.x <= p2.x or p2.x <= R.x <= p1.x):
                    # l.debug("Intersection detected at", R)
                    # The only point in which speed can be 0 is the initial point
                    assert R.y > 0.0, "Speed is negative or zero"
                    # l.debug("It's a closest intersection point to", the_intersection_point.x)
                    closest_intersection_point_and_line = (R, the_other_line)
            else:
                # Overlapping or parallel
                l.debug("WARN, No single intersection point detected between", the_line, "=", L1, "and", the_other_line, "=", L2)

        # Either there's a line or None is returned
        if closest_intersection_point_and_line[1] is not None:
            return closest_intersection_point_and_line
        else:
            return None

    def _compute_acceleration_lines(self, max_acc):
        # Because of floating point computation, points might not be exactly on the line, which means that they will
        #   not interpreted as intersections, so we add a small epsilon to them
        right_points = [Point(0, 0)] + [Point(rs.end, rs.max_speed) for rs in self.road_segments]
        return [Road._compute_acceleration_line(rp, max_acc) for rp in right_points]

    def _compute_deceleration_lines(self, max_dec):
        left_points = [Point(rs.start, rs.max_speed) for rs in self.road_segments]
        return [Road._compute_deceleration_line(rp, max_dec) for rp in left_points]

    def _compute_max_achievable_speed(self, max_acc, max_dec, speed_limit):
        # Iteratively identify the intersection points among acceleration, deceleration and max_speed lines
        acceleration_lines = self._compute_acceleration_lines(max_acc)
        deceleration_lines = self._compute_deceleration_lines(max_dec)
        #
        max_speed_lines = [LineString([(rs.start-0.01, rs.max_speed), (rs.end+0.01, rs.max_speed)]) for rs in self.road_segments]

        # Get the x value of the furthest possible point
        last_point = max_speed_lines[-1].coords[-1][0]

        # Start from position 0 and speed 0:
        intersection_points = [Point(0, 0)]
        last_intersection_point = intersection_points[0]
        # Initially we have to accelerate
        current_line = acceleration_lines[0]
        # Keep track of the last interescting line to include the last point defining the LineString
        last_intersecting_line = current_line

        while True:
            # We look for intersections with segments of different type than current_line, because according to
            # the model we cannot accelerate more than what we are currently doing (or dec)

            lines_to_check = []
            if current_line not in acceleration_lines:
                # l.debug("Consider ACC")
                lines_to_check.extend(acceleration_lines)
            if current_line not in deceleration_lines:
                # l.debug("Consider DEC")
                lines_to_check.extend(deceleration_lines)
            if current_line not in max_speed_lines:
                # l.debug("Consider MAX SPEED")
                lines_to_check.extend(max_speed_lines)

            # Find the closest line that intersects the current_line (after the last intersection point).
            # Note that the last intersection points always belong to the current_line
            intersection = Road._find_closest_intersecting_line(current_line, last_intersection_point, lines_to_check)

            # If we did not find any new intersection point we stop
            # TODO Check that we actually reached the end of the road !
            if intersection is None:
                # TODO Replace this with: until there's lines to check, check them otherwise exit the loop...
                # l.debug("Cannot find additional intersections") #. Intersection Points", intersection_points)
                break

            # Remove current_line from the search set, reduce the search space
            if current_line in acceleration_lines:
                acceleration_lines.remove(current_line)
            if current_line in deceleration_lines:
                deceleration_lines.remove(current_line)
            if current_line in max_speed_lines:
                max_speed_lines.remove(current_line)

            # Store the new intersection point and update the loop variables
            intersection_points.append(intersection[0])
            last_intersection_point = intersection[0]

            # TODO: Remove all the acceleration/deceleration lines that are before the last intersection point
            current_line = intersection[1]
            last_intersecting_line = current_line

        # Add the very last intersection point corresponding to last_point on current_line
        # if last_intersection_point.x < last_point:
        # l.debug("Debug", "Add last point to conclude the LineString using the last intersecting line", last_intersecting_line )
        # Get the coefficients corresponding to this line segment

        L = Road._line(last_intersecting_line)
        vertical_line = LineString([(last_point, 0), (last_point, speed_limit + 10)])
        V = Road._line(vertical_line)

        R = Road._intersection(L, V)

        assert R, "No single intersection to identify final point"

        # l.debug("Final point is ", R)
        intersection_points.append(R)

        # last_intersection_point = current_line.intersection( vertical_line )
        # if last_intersection_point.type == 'Point':
        #     intersection_points.append( last_intersection_point )

        # The "max_achievable_speed" of the road is defined by the list of the
        # intersection points. This information can be used to compute the speed profile of the road given
        # a "discretization function"
        return LineString([p for p in intersection_points])

    def _get_curvature_between(self, begin, end):
        # Get the segments between begin and end
        segments = [ s for s in self.road_segments if s.end >= begin and s.start <= end ]
        # Order segments from longest to smallest
        # TODO Check this !!!
        segments.sort(key=lambda rs: rs.length, reverse=False)

        # Take the first. ASSUMPTION: Always one?
        return segments[0].curvature

    def compute_curvature_profile(self, curvature_bins, distance_step):
        # Compute length of segment
        # Compute curvature of segment as 1/radius:
        # - Use angles to decide wheter this is a left/right (positive/negative)
        # - curvature == 0 for straigth

        # Compute the curvatyre in each segment of the road, defined by distance_step
        # For pieces that belong to multiple segments use the majority/bigger one to decide, and then
        # lexycografic order
        start = 0
        stop = self.total_length
        number_of_segments = math.ceil(stop / distance_step)
        discretized_road = np.linspace(start, stop, num=number_of_segments, endpoint=True)

        observations = []
        for begin, end in zip(discretized_road[0:], discretized_road[1:]):
            observations.append(self._get_curvature_between(begin, end))

        # Hist contains always one element less than bins because you need to consecutive points to define a bin...
        # [a, b, c] -> bins(a-b, b-c)
        hist, bins = np.histogram(observations, bins=curvature_bins)

        total = sum(hist)

        # Normalize the profile to get the percentage
        hist = [bin / total for bin in hist]

        return hist

    def _get_speed_at_position(self, max_achievable_speed, x):
        # Find the segment in which x belongs

        # last_point_before_x
        p1 = [coord for coord in list(zip(*max_achievable_speed.coords.xy)) if coord[0] <= x][-1]

        # first_point_after_x. This might be empty/missing due to discretizations, so we return None
        after_x = [coord for coord in list(zip(*max_achievable_speed.coords.xy)) if coord[0] >= x]

        if len(after_x) == 0:
            l.debug("DEBUG, Cannot find any point after", x, "Return None")
            return None

        p2 = after_x[0]

        # Can there be a case in which only last_point_before_x is found ? What do we do then?
        if p1 == p2:
            return p1[1]
        else:
            # Interpolate since those are piecewise linear
            # Slope (y2-y1)/(x2-x1)
            slope = (p2[1]-p1[1])/(p2[0]-p1[0])
            # Y-intercept: y1 = slope * x1 + b -> b = y1 - slope*x1
            y_intercept = p1[1] - slope * p1[0]
            # Finally compute the value which correponds to x by pluggin it inside the formula:
            y = x * slope + y_intercept
            return y

    def compute_speed_profile(self, max_acc, max_dec, speed_limit, speed_bins, distance_step):
        # Return the "actual" speed that the car can reach given its acc/dec
        max_achievable_speed = self._compute_max_achievable_speed(max_acc, max_dec, speed_limit)

        # self._plot_debug(max_acc, max_dec, max_achievable_speed)

        # Compute the avg speed in each segment of the road, defined by distance_step
        start = 0
        stop = self.total_length
        number_of_segments = math.ceil(stop / distance_step)
        discretized_road = np.linspace(start, stop, num=number_of_segments, endpoint=True)

        # Compute the "avg" speed at each segment. The avg is computed by computing the mean of:
        # entry point, exit point, and "intersection" points inside max_achievable_speed
        average_speed = []

        for begin, end in zip(discretized_road[0:], discretized_road[1:]):
            speed_at_begin = self._get_speed_at_position(max_achievable_speed, begin)
            assert speed_at_begin is not None, "Speed at begin of segment cannot be None"
            assert speed_at_begin <= speed_limit + 0.01, "Speed at begin of segment is over speed limit"
            assert speed_at_begin >= 0.0, "Speed at begin of segment is negative " + str(speed_at_begin)

            speed_at_end = self._get_speed_at_position(max_achievable_speed, end)

            if speed_at_end <= 0 or speed_at_end > speed_limit + 0.01:
                self._plot_debug(max_acc, max_dec, max_achievable_speed)

            assert speed_at_end >= 0.0, "Speed at end of segment is negative " + str(speed_at_end)
            assert speed_at_end <= speed_limit + 0.01, "Speed at end of segment " + str(speed_at_end) + " is over speed limit" + str(speed_limit + 0.01)
            assert speed_at_end is not None, "Speed at end of segment cannot be None"

            # Get all the points at which the speed changes between "begin" and "end" on the max_achievable_speed line
            speeds = [s[1] for s in list(zip(*max_achievable_speed.coords.xy)) if begin < s[0] < end]

            observations = []
            observations.append(speed_at_begin)
            observations = observations + speeds
            observations.append(speed_at_end)

            # Compute the average speed in the segment
            average_speed.append(np.mean(observations))

        # Compute the histogram of the average speeds across all the pieces of road
        hist, bins = np.histogram(average_speed, bins=speed_bins)

        total = sum(hist)

        # Normalize the profile to get the percentage
        hist = [bin / total for bin in hist]

        return hist

    def get_position_at_distance(self, distance):
        # Find the segment that can be reach at given distance from start
        cumulative_length = 0
        target_segment = None
        for segment in self.road_segments:
            cumulative_length += segment.length
            if cumulative_length >= distance:
                target_segment = segment
                break

        # Find the position inside the target_segment on the middle line
        distance_to_end = cumulative_length - distance
        # This is the target distance from the starting of the middle line of this segment
        distance_from_start = target_segment.length - distance_to_end

        cumulative_length_in_segment = 0
        for pair in pairs(list(target_segment.geometry)):
            # Each pair is a couple of dictionaries, each describing a point in the road using left, middle, and right point
            # We use the middle on
            A = Point(pair[0]['middle'][0], pair[0]['middle'][1])
            B = Point(pair[1]['middle'][0], pair[1]['middle'][1])

            cumulative_length_in_segment += A.distance(B)

            if cumulative_length_in_segment >= distance_from_start:
                # The point we look for is between the points that define this pair at distance D from the second point
                D = cumulative_length_in_segment - distance_from_start
                # Note that the point MUST be on the middle-line !
                the_point = LineString([B, A]).interpolate(D)
                return the_point

        return None

    def discretize(self, meters):
        """ Return the coordinates of the road at various positions"""
        distance = 0
        locations = list()
        # TODO probably we can add speed as well?
        while distance < self.total_length:
            locations.append((distance, self.get_position_at_distance(distance)))
            distance += meters

        return locations


class RoadPlotter:

    def __init__(self, ax, distance_step):
        self.ax = ax
        self.distance_step = distance_step

    def _plot_lineStrings(self, lineStrings, color):
        for lineString in lineStrings:
            x,y = lineString.xy
            self.ax.plot(x, y, color = color)

    def _draw_grid(self, road: Road):
        # Draw grid again
        # Show the grid that corresponds to the discretization
        self.ax.xaxis.grid(True)
        # Compute the avg speed in each segment of the road, defined by distance_step
        start = 0
        stop = road.total_length
        number_of_segments = math.ceil(stop / self.distance_step)
        discretized_road = np.linspace(start, stop, num=number_of_segments, endpoint=True)
        # Compute the avg speed in each segment of the road, defined by distance_step
        ticks = discretized_road
        self.ax.set_xticks(ticks)

    def plot_acceleration_lines(self, road: Road, max_acc:float):
        self._plot_lineStrings(road._compute_acceleration_lines(max_acc), 'red')

    def plot_deceleration_lines(self, road: Road, max_dec:float):
        self._plot_lineStrings(road._compute_deceleration_lines(max_dec), 'blue')

    def plot_max_speed(self, road: Road):
        for road_segment in road.road_segments:
            # Plot max speed
            x = [road_segment.start, road_segment.end]
            y = [road_segment.max_speed, road_segment.max_speed]
            self.ax.plot(x, y, color='#999999')
        # self._draw_grid(road)


class RoadProfiler:

    G = 9.81
    _DISTANCE_STEP = 5  # Meters

    def __init__(self, mu, speed_limit_meter_per_second, road_geometry, discretization_factor=10,
                    max_acc = 3.5, max_dec = -0.5):

        self.FRICTION_COEFFICIENT = mu
        self.speed_limit_meter_per_second = speed_limit_meter_per_second

        # The "model of the car used to estimate speed"
        self.max_car_acc = max_acc
        self.max_car_dec = max_dec

        # TODO This is buggy. We cannot distinguish left from right turns
        self.road = self._build_road_object(road_geometry)

        # Decide how fine grained the control his. Smaller values results in better controls but higher computational costs
        self.discretization_factor = discretization_factor
        #


    def _compute_max_speed(self, radius: float):
        max_speed = self.speed_limit_meter_per_second

        if radius == math.inf:
            return max_speed

        # Assumption: All turns are unbanked turns. Pivot_off should be the radius of the curve
        max_speed = math.sqrt(self.FRICTION_COEFFICIENT * self.G * radius)
        if (max_speed >= self.speed_limit_meter_per_second):
            max_speed = self.speed_limit_meter_per_second

        return max_speed

    def _compute_curvature(self, radius: float):
        """ NOTE: We do not distinguish left and right turns here..."""
        if radius == math.inf:
            return 0
        else:
            return 1.0 / radius

    # TODO Tell whether this is a RIGH or LEFT turn
    def _compute_radius_turn(self, road_segment: list): # List(line_segment)
        """ Returns the position of the center and radius of the circle defining the arc used in this curve/turn"""

        if len(road_segment) <= 2:
            # Straight segment - direction is 0, curvature and radius not defined
            return Point(math.inf, math.inf), math.inf, 0.0

        # Use triangulation.
        # First point forming the first line_segment
        p1 = Point(road_segment[0][0]['middle'][0], road_segment[0][0]['middle'][1])
        x1 = p1.x
        y1 = p1.y
        # Last point forming the last line_segment
        p2 = Point(road_segment[-1][1]['middle'][0], road_segment[-1][1]['middle'][1])
        x2 = p2.x
        y2 = p2.y
        # This is the second point forming the first line_segment
        p3 = Point(road_segment[0][1]['middle'][0], road_segment[0][1]['middle'][1])
        x3 = p3.x
        y3 = p3.y

        center_x = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (
                    y1 - y2)) / (2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2))
        center_y = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (
                    x2 - x1)) / (2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2))

        # https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
        # The line is defined by p1=A(x1, y1) and middle_point=B(x3, y3) under the assumption the angle is less than 180 ?
        # P(center_x, center_y)
        d = (center_x - x1) * (y3 - y1) - (center_y - y1) * (x3 - x1)

        # If 𝑑<0 then the point lies on one side of the line, and if 𝑑>0 then it lies on the other side. If 𝑑=0 then the point lies exactly line.
        assert d != 0

        radius = math.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)

        return Point(center_x, center_y), radius, d

    def _build_road_object(self, road_geometry):
        # To identify segments we use the following heuristic:
        # - form a line_segment for each pair of points
        # - compute the angle between consecutive line_segments
        # - assuming that the geometry does not contains clothoids we group line_segments by angle
        # Remark: BeamNG interpolates roads using splines, so even if as input we provide pieces that are exactly arcs
        #       and straigths, as output we might obtain sligtly different results, especially between segments
        # This method is not necessary accurate but might be enough for us.
        line_segments = []
        for pair in pairs(list(road_geometry)):
            line_segments.append(pair)

        angles = []
        for pair in pairs(list(line_segments)):
            # Derive vectors from the line segments. Difference last point - first point
            first_segment = [Point(pair[0][0]['middle'][0], pair[0][0]['middle'][1]), Point(pair[0][1]['middle'][0], pair[0][1]['middle'][1])]
            second_segment = [Point(pair[1][0]['middle'][0], pair[1][0]['middle'][1]), Point(pair[1][1]['middle'][0], pair[1][1]['middle'][1])]

            # initialize arrays
            A = np.array([first_segment[1].x - first_segment[0].x, first_segment[1].y - first_segment[0].y])
            B = np.array([second_segment[1].x - second_segment[0].x, second_segment[1].y - second_segment[0].y])
            #
            direction = math.copysign(1.0, np.cross(A, B))
            angle_between_segments = math.atan2(abs(np.cross(A, B)), np.dot(A,B))
            # So we have neg for left and positive for right... I hope :D
            angles.append(( direction * angle_between_segments, pair))

        last_angle = 0
        road_segments = list()
        current_road_segment = list()
        road_segments.append(current_road_segment)
        # For each pair establish if they belong together
        for angle, segments_pair in angles:
            if math.isclose(angle, last_angle, rel_tol=10e-4):
                current_road_segment.append(segments_pair[0])
            else:
                # Reset and restore in road_segment
                current_road_segment = [segments_pair[0]]
                road_segments.append(current_road_segment)

            last_angle = angle

        # We assume the last segmen belongs to the last road_segment
        current_road_segment.append( angles[-1][1][1])

        input_road_segments = list()
        # Segments now contains a partition of the points that form the road (left, middle, and right)
        for road_segment in road_segments:
            # Skip empty Might happen at the beginning
            if len(road_segment) == 0:
                continue

            center_point, radius, direction = self._compute_radius_turn(road_segment)

            curvature = self._compute_curvature(radius)
            max_speed = self._compute_max_speed(radius)
            length = self._compute_length(road_segment)
            geometry = self._compute_geometry(road_segment)

            # Really the above computation could be moved inside the constructor... since they are computed from road_segment
            the_road_segment = RoadSegment(length, max_speed, curvature, geometry, direction)

            input_road_segments.append(the_road_segment)

        return Road(input_road_segments)

    def _compute_geometry(self, road_segment):
        # Approximate length as cumulative distance between the points that define the segment
        road_geometry = []
        for line_segment in road_segment:
            first_geometry = line_segment[0]
            second_geometry = line_segment[1]
            if first_geometry not in road_geometry:
                road_geometry.append( first_geometry)
            if second_geometry not in road_geometry:
                road_geometry.append( second_geometry)

        return road_geometry


    def _compute_length(self, road_segment):
        # Approximate length as cumulative distance between the points that define the segment
        length = 0
        for line_segment in road_segment:
            first_point = Point(line_segment[0]['middle'][0], line_segment[0]['middle'][1])
            second_point = Point(line_segment[1]['middle'][0], line_segment[1]['middle'][1])
            length += first_point.distance(second_point)

        return length


    def compute_segments(self):
        return self.road.road_segments

    def compute_sectors_by_driving_distance(self, number_of_sectors):
        # Compute the overall distance/lenght of the road and divide by the number of sectors
        # TODO Adjust by segments. In some cases is better to make a segment more but smaller.
        #   For example, if the last segment is longer than the target distance.
        #   Smells like an optimization problem based on a cost function....
        #
        # From the driving path, a geometry, generate the data structure to compute the speed profile
        # This compute the metadata about the road and the segments
        sector_target_travel_distance = float(self.road.total_length / number_of_sectors)
        # Define the sectors as the smallest subset of road_segments whose length sums to sector_target_travel_distance
        sectors = []
        cumulative_travel_distance = 0
        current_sector = []
        for segment in self.road.road_segments:
            current_sector.append(segment)

            cumulative_travel_distance = cumulative_travel_distance + segment.length

            if cumulative_travel_distance > sector_target_travel_distance:
                # make a copy and then reset.
                sectors.append( [ s for s in current_sector] )
                # Reset loop variables
                cumulative_travel_distance = 0
                current_sector = []

        # Add the last segment if we have left it out
        if len(current_sector) > 0:
            sectors.append([s for s in current_sector])

        return sectors

    def compute_sectors_by_travel_time(self, number_of_sectors):
        """
        Compute the "min" travel time using the estimated max speed per segment. This is quite approximate but being
        a simpler algorithm it should be more robust

        :param number_of_sectors:
        :return:
        """
        total_travel_distance = self.road.total_length
        max_travel_speed_list = [rs.max_speed for rs in self.road.road_segments]
        mode_of_max_travel_speed = max(set(max_travel_speed_list), key=max_travel_speed_list.count)
        estimated_travel_time = total_travel_distance / mode_of_max_travel_speed

        sector_target_travel_time = estimated_travel_time / float(number_of_sectors)

        # for each segment we estimate its travel_time at the max speed it declars

        sectors = []

        current_sector = []
        sectors.append(current_sector)

        cumulative_estimated_travel_time = 0.0
        for road_segment in self.road.road_segments:
            segment_estimated_travel_time = road_segment.length / road_segment.max_speed
            cumulative_estimated_travel_time += segment_estimated_travel_time
            current_sector.append(road_segment)

            if cumulative_estimated_travel_time >= sector_target_travel_time:
                current_sector = []
                sectors.append(current_sector)
                cumulative_estimated_travel_time = 0.0

        # Last sector might be empty
        if len(sectors[-1]) == 0:
            sectors = sectors[:-1]

        # If there are more sectors than asked for merge the last two
        if( len(sectors) > number_of_sectors ):
            sectors[-2] = sectors[-2] + sectors[-1]
            del sectors[-1]
            pass

        return sectors

    # Deprecated. Does not provide the same results if we rotate the road...
    def __compute_sectors_by_travel_time(self, number_of_sectors):

        # Locations is a list of tuples (distance, Point) computed from driving path every 10 meters
        # with 10 meter for discretization the driver cuts sharp turns
        locations = self.road.discretize(self.discretization_factor)

        # Return the "actual" speed that the car can reach given its acc/dec
        max_achievable_speed = self.road._compute_max_achievable_speed(self.max_car_acc, self.max_car_dec,
                                                                        self.speed_limit_meter_per_second)
        # Speeds is similar to Locations but contains distance and speed data
        speeds = list()
        for distance, _ in locations:
            speeds.append((distance, self.road._get_speed_at_position(max_achievable_speed, distance)))

        location_and_speed = list()
        for location, speed in zip(locations, speeds):
            location_and_speed.append((location[0], location[1], speed[1]))

        # Now we compute the time using the constant speed law
        timing = list()
        total_travel_time = 0
        for pair in pairs(location_and_speed):
            avg_speed = 0.5 * (pair[0][2] + pair[1][2])
            distance = pair[1][0] - pair[0][0]
            travel_time = distance / avg_speed
            total_travel_time += travel_time

            timing.append((pair[1][0], pair[1][1], travel_time))

        sector_target_travel_time = float(total_travel_time/number_of_sectors)

        # Simple heuristic include the entire segment if the location belongs to it. And start computing from next segment
        # Define the sectors as the smallest subset of road_segments whose travel time sums to sector_target_travel_time
        sectors = []
        current_sector = []
        cumulative_travel_time = 0
        skip_to_next_segment = False
        last_segment = None
        for location_time in timing:
            # Get segment containing this location
            current_segment = self.road.get_segment_at_position(location_time[1])

            assert current_segment is not None

            # Refresh Skip. If current segment is different than the previous one, skip myst be false
            if current_segment != last_segment and skip_to_next_segment:
                skip_to_next_segment = False

            if skip_to_next_segment:
                continue


            if current_segment not in current_sector:
                current_sector.append(current_segment)

            travel_time = location_time[2]

            cumulative_travel_time = cumulative_travel_time + travel_time

            if cumulative_travel_time > sector_target_travel_time:
                # make a copy and segments
                sectors.append([s for s in current_sector])

                # Fast forward to the first location in the next segment and reset the variables
                last_segment = current_segment
                cumulative_travel_time = 0
                current_sector = []
                skip_to_next_segment = True

        # Add the last segment if we have left it out
        if len(current_sector) > 0:
            sectors.append([s for s in current_sector])

        return sectors

    def compute_ai_script(self, driving_path: LineString, car_model: dict):
        # From the driving path, a geometry, generate the data structure to compute the speed profile
        input_road = self._build_road_object(driving_path)

        # Locations is a list of tuples (distance, Point) computed from driving path every 10 meters
        # with 10 meter for discretization the driver cuts sharp turns
        locations = input_road.discretize(self.discretization_factor)

        # Return the "actual" speed that the car can reach given its acc/dec
        max_achievable_speed = input_road._compute_max_achievable_speed(car_model['max_acc'], car_model['max_dec'],
                                                                        self.speed_limit_meter_per_second)

        # input_road._plot_debug(car_model['max_acc'], car_model['max_dec'], max_achievable_speed)

        # Speeds is similar to Locations but contains distance and speed data
        speeds = list()
        for distance, _ in locations:
            speeds.append((distance, input_road._get_speed_at_position(max_achievable_speed, distance)))


        location_and_speed = list()
        for location, speed in zip(locations, speeds):
            location_and_speed.append((location[0], location[1], speed[1]))


        # Now we compute the time using constant speed law
        timing = list()
        cumulative_time = 0
        for pair in pairs(location_and_speed):
            avg_speed = 0.5 * (pair[0][2] + pair[1][2])
            distance = pair[1][0] - pair[0][0]
            travel_time = distance / avg_speed
            cumulative_time += travel_time
            timing.append(cumulative_time)

        assert len(timing) == len(location_and_speed) -1

        script = list()
        # We do  not report the initial position t=0, d=0 and v=0
        for location_and_speed, target_time in zip(location_and_speed[1:], timing[:]):
            travel_distance = location_and_speed[0]
            target_position = location_and_speed[1]
            avg_speed = location_and_speed[2]
            node = {
                'travel_distance': travel_distance,
                'avg_speed': avg_speed,
                'x': target_position.x,
                'y': target_position.y,
                'z': 0.3,
                't': target_time,
            }
            script.append(node)

        return script

