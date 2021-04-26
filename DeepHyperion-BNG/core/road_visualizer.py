
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
import json
import numpy as np
from shapely.geometry import Point, Polygon, LineString, box as Box
from shapely import affinity
from descartes import PolygonPatch
from core.road_profiler import RoadProfiler
from math import pi, atan2
import os


def pairs(lst):
    for i in range(1, len(lst)):
        yield lst[i - 1], lst[i]


class ExportRoadMetadataPlugin(plugins.PluginBase):
    """This plugin defines a global variable roadMetadata filled with data about the road and its sectors"""
    # https://mpld3.github.io/_downloads/custom_plugins.html
    # This is quite brutal, when the plugin is rendered it will define a global variable called theSectors with some
    #   data in it. A better alternative would have been to export a function to access that object, but I cannot
    #   find a way to do that.

    JAVASCRIPT = """
    mpld3.register_plugin("exportRoadMetadata", ExportRoadMetadataPlugin);
    ExportRoadMetadataPlugin.prototype = Object.create(mpld3.Plugin.prototype);
    ExportRoadMetadataPlugin.prototype.constructor = ExportRoadMetadataPlugin;
    ExportRoadMetadataPlugin.prototype.requiredProps = ["road_metadata"];
    ExportRoadMetadataPlugin.prototype.defaultProps = {}

    function ExportRoadMetadataPlugin(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };
    
    ExportRoadMetadataPlugin.prototype.draw = function(){ 
        road_metadata = this.props.road_metadata
    }                                             
    
    """

    def __init__(self, road_metadata: dict):
        # This must match with the register plugin call
        self.dict_ = {"type": "exportRoadMetadata"}
        # Extract the info about sectors that we want to share
        self.dict_['road_metadata'] = road_metadata


class CarVisualizer:
    """ Compact cars are usually 4.1-4.3 meters long, 1.4-1.5 meters high, and 1.7-1.8 meters wide."""
    LENGHT=4.2 # meters
    WIDHT=1.8 # meters

    FOV_DISTANCE=30 # meters
    FOV_WIDE=60 #degrees

    def __init__(self, fig, ax, labels_and_handlers):
        self.fig = fig
        self.ax = ax
        self.labels_and_handlers = labels_and_handlers


        self.car_box = Box(-self.LENGHT/2.0, -self.WIDHT/2.0, self.LENGHT/2.0, self.WIDHT/2.0)

        centerx, centery = 0, 0
        radius = self.FOV_DISTANCE
        start_angle, end_angle = -self.FOV_WIDE/2.0, +self.FOV_WIDE/2.0  # In degrees
        # We do not really need that many, do we?
        numsegments = 10

        # The coordinates of the arc
        theta = np.radians(np.linspace(start_angle, end_angle, numsegments))
        x = centerx + radius * np.cos(theta)
        x = np.insert(x, 0, 0.0)

        y = centery + radius * np.sin(theta)
        y = np.insert(y, 0, 0.0)

        arc = LineString(np.column_stack([x, y]))

        self.fov = Polygon(list(arc.coords))

    def plot_car(self, center_position, rotation_angle):
        # Since this might be invoked many times we need to append each patch
        if 'Car' not in self.labels_and_handlers:
            self.labels_and_handlers['Car'] = []

        transformed_fov = affinity.rotate(self.fov, origin=(0,0), angle=rotation_angle)
        transformed_fov = affinity.translate(transformed_fov, xoff=center_position[0], yoff=center_position[1])

        transformed_car_box = affinity.rotate(self.car_box, origin=(0,0), angle=rotation_angle)
        transformed_car_box= affinity.translate(transformed_car_box, xoff=center_position[0], yoff=center_position[1])

        # TODO Translate and rotate, then plot

        # poly_fov = self.fov
        # self.ax.plot(poly_fov.xy)
        #

        # self.ax.plot()

        poly, = self.ax.plot(*transformed_fov.exterior.xy, color='black')
        # patch = PolygonPatch(transformed_fov, fc='#ffffff30', linewidth=0)
        # plot = self.ax.add_patch(patch)
        self.labels_and_handlers['Car'].extend([poly])

        # poly, = self.ax.plot(*transformed_car_box.exterior.xy, color='red')
        patch = PolygonPatch(transformed_car_box, fc='red', linewidth=0)
        plot = self.ax.add_patch(patch)
        self.labels_and_handlers['Car'].extend([patch, plot])



        # self.labels_and_handlers['Car'].append(f)


class RoadVisualizer:

    # Split the road in that much sectors based on travel time or travel distance
    CELL_SIZE = 20           # meters

    def __init__(self, road_geometry, road_lanes=(1, 1), number_of_sectors=4):
        self.NUMBER_OF_SECTORS = number_of_sectors
        # The layers of the figure.
        self.layers = dict()
        # The reference to the current figure and axes
        # TODO This might be changed later as all the other settings
        # TODO Put somewhere the legend/scale this should match the level of the zoom or we need to write it somewhere
        #   that a square is 20m x 20m
        # self.fig, self.ax = plt.subplots(figsize=(6, 4))

        self.road_geometry = self._standardize(road_geometry)

        self.road_lanes= road_lanes

        # Road Definition
        friction_coefficient = 0.8  # Friction Coefficient
        speed_limit_meter_per_sec = 90.0 / 3.6  # Speed limit m/s
        self.road_profiler = RoadProfiler(friction_coefficient, speed_limit_meter_per_sec, self.road_geometry)

        self._identify_segments()

        self._identify_sectors()
        #
        # Since we need to customize legends we store explictly handlers and labels
        self.labels_and_handlers = dict()

        #self.cv = CarVisualizer(self.fig, self.ax, self.labels_and_handlers)

    def _standardize(self, road_geometry):
        """ Translate the geometry to have the starting position at (0,0) and the first segment pointing North"""
        left_edge_x = np.array([e['left'][0] for e in road_geometry])
        left_edge_y = np.array([e['left'][1] for e in road_geometry])
        left_edge_z = np.array([e['left'][2] for e in road_geometry])

        right_edge_x = np.array([e['right'][0] for e in road_geometry])
        right_edge_y = np.array([e['right'][1] for e in road_geometry])
        right_edge_z = np.array([e['right'][2] for e in road_geometry])

        middle_line_x= np.array([e['middle'][0] for e in road_geometry])
        middle_line_y= np.array([e['middle'][1] for e in road_geometry])
        middle_line_z = np.array([e['middle'][2] for e in road_geometry])

        # Note that one must be in reverse order for the polygon to close
        right_edge = LineString(zip(right_edge_x, right_edge_y))
        left_edge = LineString(zip(left_edge_x, left_edge_y))
        middle_line = LineString(zip(middle_line_x, middle_line_y))


        translate_to_origin_x = - road_geometry[0]['middle'][0]
        translate_to_origin_y = - road_geometry[0]['middle'][1]

        right_edge = affinity.translate(right_edge, xoff=translate_to_origin_x, yoff=translate_to_origin_y, zoff=0.0)
        left_edge = affinity.translate(left_edge, xoff=translate_to_origin_x, yoff=translate_to_origin_y, zoff=0.0)
        middle_line = affinity.translate(middle_line, xoff=translate_to_origin_x, yoff=translate_to_origin_y, zoff=0.0)

        # Rotate
        # https://www.quora.com/What-is-the-angle-between-the-vector-A-2i+3j-and-y-axis#:~:text=If%20we%20wish%20to%20find,manipulate%20the%20dot%20product%20equation.

        delta_y = middle_line_y[1] - middle_line_y[0]
        delta_x = middle_line_x[1] - middle_line_x[0]

        current_angle = atan2(delta_y, delta_x)

        right_edge = affinity.rotate(right_edge , (pi / 2) - current_angle, origin=(0,0), use_radians=True)
        left_edge = affinity.rotate(left_edge, (pi / 2) - current_angle,  origin=(0,0),use_radians=True)
        middle_line = affinity.rotate(middle_line, (pi / 2) - current_angle,  origin=(0,0), use_radians=True)

        # Rebuild the road_geometry objects

        left_edge_x, left_edge_y = left_edge.xy
        right_edge_x, right_edge_y = right_edge.xy
        middle_line_x, middle_line_y = middle_line.xy

        # [{"right": [522.611083984375, 394.12884521484375, -9.765848517417908e-06], "left": [527.388916015625, 387.7122802734375, -9.765848517417908e-06], "middle": [525, 390.9205627441406, -9.765848517417908e-06]},
        # TODO Better to use a comprehension but this is out of my league
        # ALL THE LISTS HAVE THE SAME LENGHT!
        standardized_road_geometry = []
        for idx in range(0, len(left_edge_x)):
            geometry = dict()
            geometry['right']=[right_edge_x[idx], right_edge_y[idx], right_edge_z[idx]]
            geometry['left']=[left_edge_x[idx], left_edge_y[idx], left_edge_z[idx]]
            geometry['middle'] =[middle_line_x[idx], middle_line_y[idx], middle_line_z[idx]]
            standardized_road_geometry.append( geometry )

        return standardized_road_geometry

        # line = LineString([(1, 3), (1, 1), (4, 1)])
        # rotated_a = affinity.rotate(line, 90)
        # rotated_b = affinity.rotate(line, 90, origin='origin')

        pass

    # Compute the distance or travel time
    def _identify_sectors(self):
        # self.sectors  = self.road_profiler.compute_sectors_by_driving_distance(self.NUMBER_OF_SECTORS)
        self.sectors = self.road_profiler.compute_sectors_by_travel_time(self.NUMBER_OF_SECTORS)

    def _identify_segments(self):
        self.segments = self.road_profiler.compute_segments()

    def _plot_grid(self):
        """ We explicitly draw the grid instead of using ax.grid to avoid the zoom plugin to mess it up. We make it
        square so when we zoom in it does not look that bad"""
        # TODO To define a background colot
        # self.ax.set_facecolor(color="#00CC6630")



        # TODO Make it square and centered

        x_ticks = list(range(self.ax.get_xlim()[0].astype(np.int64),self.ax.get_xlim()[1].astype(np.int64)+self.CELL_SIZE,self.CELL_SIZE))
        y_ticks = list(range(self.ax.get_ylim()[0].astype(np.int64),self.ax.get_ylim()[1].astype(np.int64)+self.CELL_SIZE,self.CELL_SIZE))

        l1, = self.ax.plot((0,0), alpha=0.3)
        l2 = self.ax.vlines(x_ticks, y_ticks[0], y_ticks[-1], alpha=0.3)
        l3 = self.ax.hlines(y_ticks, x_ticks[0], x_ticks[-1], alpha=0.3)
        self.labels_and_handlers['Grid'] = (l1, l2, l3)

    def _build_polygon_from_geometry(self, geometry):
        left_edge_x = np.array([e['left'][0] for e in geometry])
        left_edge_y = np.array([e['left'][1] for e in geometry])
        right_edge_x = np.array([e['right'][0] for e in geometry])
        right_edge_y = np.array([e['right'][1] for e in geometry])

        road_edges = dict()
        road_edges['left_edge_x'] = left_edge_x
        road_edges['left_edge_y'] = left_edge_y
        road_edges['right_edge_x'] = right_edge_x
        road_edges['right_edge_y'] = right_edge_y

        # Note that one must be in reverse order for the polygon to close
        right_edge = LineString(zip(road_edges['right_edge_x'][::-1], road_edges['right_edge_y'][::-1]))
        left_edge = LineString(zip(road_edges['left_edge_x'], road_edges['left_edge_y']))

        l_edge = left_edge.coords
        r_edge = right_edge.coords

        return Polygon(list(l_edge) + list(r_edge))

    def _plot_road(self):
        """ Tranform the road geometry coordinates into a Polygon Patch and draw it"""
        poly = self._build_polygon_from_geometry(self.road_geometry)

        # To make lane marking a bit more visible let's make the road a bit larger
        poly = poly.buffer(0.3) # 30 cm ?

        # Draw the road asphalt
        # TODO This does not disappear when clicking on the legend
        patch = PolygonPatch(poly, fc='#d3d3d3', linewidth=0)
        p = self.ax.add_patch(patch)
        # Draw the boundaries on top of the geometry. Not sure if this is really necessary

        # Lanes
        left_edge_x = np.array([e['left'][0] for e in self.road_geometry])
        left_edge_y = np.array([e['left'][1] for e in self.road_geometry])

        right_edge_x = np.array([e['right'][0] for e in self.road_geometry])
        right_edge_y = np.array([e['right'][1] for e in self.road_geometry])

        middle_line_x = np.array([e['middle'][0] for e in self.road_geometry])
        middle_line_y = np.array([e['middle'][1] for e in self.road_geometry])

        # Note that one must be in reverse order for the polygon to close
        l, = self.ax.plot(left_edge_x, left_edge_y, color="white")
        r, = self.ax.plot(right_edge_x, right_edge_y, color="white")
        # TODO Not yet a double line. Probably use some parallel_offset or forget about it
        m, = self.ax.plot(middle_line_x, middle_line_y, color="yellow")

        self.labels_and_handlers['Road Markings'] = (p, l, r, m)

    def _plot_sectors(self, road_id):
        # TODO
        """ Since the number of sectors is limited we plot all of them and use their ID to identify them"""
        colors_map = ['green', 'blue', 'cyan', 'red']
        for idx, sector in enumerate(self.sectors):
            # Sector is a list of segments we need to merge their geometries
            sector_geometry = [element for rs in sector for element in rs.geometry]
            # Define the polygon to draw.
            # TODO We can make this a patch and use alpha=0.3 and the color but I cannot make it disappear for the moment
            sector_poly = self._build_polygon_from_geometry(sector_geometry)
            # https://stackoverflow.com/questions/55522395/how-do-i-plot-shapely-polygons-and-objects-using-matplotlib
            l, = self.ax.plot(*sector_poly.exterior.xy, color=colors_map[idx % len(colors_map)])
            # Make sure that each sector appears as single element in the figure
            self.labels_and_handlers["".join(['Sector', str(idx)])] = (l)
            # ht_figure = self.ax.get_figure()
            # fig_name = road_id.replace('/', '_')
            # ht_figure.savefig("correlation/"+fig_name+".png", dpi=400)
        

    def _plot_road_segments(self):
        # TODO COLORS CAN BE DEFINED ON PROPERTIES OF THE SEGMENTS. like lenght and curvature

        color_map = ['cyan', 'blue', 'green', 'yellow']

        # Segments are visualized as borders of the polygon over the road geometry
        self.labels_and_handlers['road segments'] = []

        # TODO Not sure which one we want to plot here...
        for int, segment in enumerate(self.segments):
            # Sector is a list of segments we need to merge their geometry
            sector_poly = self._build_polygon_from_geometry(segment.geometry)
            # https://stackoverflow.com/questions/55522395/how-do-i-plot-shapely-polygons-and-objects-using-matplotlib
            l, = self.ax.plot(*sector_poly.exterior.xy, marker='.', color=color_map[int % len(color_map)])
            self.labels_and_handlers['road segments'].append(l)

    def _plot_car_between_geometries(self, geometry, another_geometry):
        # Get the angle from the direction of the road. Assume there's at least two points in a segment
        A = Point(geometry['middle'][0], geometry['middle'][1])
        B = Point(another_geometry['middle'][0], another_geometry['middle'][1])
        #
        delta_y = B.y - A.y
        delta_x = B.x - A.x

        # Move the car in the middle of the lane assuming only one lane is there
        car_position_x = 0.5 * (geometry['middle'][0] + geometry['right'][0])
        car_position_y = 0.5 * (geometry['middle'][1] + geometry['right'][1])

        angle = atan2(delta_y, delta_x)
        self.cv.plot_car((car_position_x, car_position_y), np.degrees(angle))

    def _plot_car(self):
        # PLOT THE CAR EVERY X meters
        # I tried to plot for sector and segment but becomes clumsly and unpredicatble
        if self.road_profiler.road.total_length / 50.0 < 3:
            min_distance = 30 # meters
        else:
            min_distance = 50 # meters
        last_distance = 0
        for pair in pairs(self.road_geometry):
            if last_distance <= 0:
                self._plot_car_between_geometries(pair[0], pair[1])
                last_distance = min_distance

            last_distance -= Point(pair[0]['middle'][0], pair[0]['middle'][1]) \
                .distance(Point(pair[1]['middle'][0], pair[1]['middle'][1]))


    # This should be done AFTER rotating and translating the roads to ensure it always fit.
    def _setup_axes(self):

        min_x = min([e['left'][0] for e in self.road_geometry] + [e['right'][0] for e in self.road_geometry])
        min_y = min([e['left'][1] for e in self.road_geometry] + [e['right'][1] for e in self.road_geometry])
        max_x = max([e['left'][0] for e in self.road_geometry] + [e['right'][0] for e in self.road_geometry])
        max_y = max([e['left'][1] for e in self.road_geometry] + [e['right'][1] for e in self.road_geometry])

        h_size = abs(min_x) + abs(max_x)
        v_size = abs(min_y) + abs(max_y)

        if h_size > v_size:
            # Take h_size and the size of the square
            center_y = min_y + v_size / 2
            min_y = center_y - h_size / 2
            max_y = center_y + h_size / 2
            pass
        else:
            # Take v_size and the size of the square
            center_x = min_x + h_size/2
            min_x = center_x - v_size/2
            max_x = center_x + v_size/2
            pass

        # Make sure the graph is a square
        self.ax.set_xlim(min_x-50, max_x+50)
        self.ax.set_ylim(min_y-50, max_y+50)
        self.ax.set_aspect('equal')
        # Hide ticks: https://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplotlib-plot
        self.ax.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off',
            right='off',
            left='off',
            labelleft='off')
        # self.ax.axis('off')
        # self.ax.set_axis_off()

        self.fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        # or fig ?
        self.ax.margins(0, 0)
        self.ax.xaxis.set_major_locator(plt.NullLocator())
        self.ax.yaxis.set_major_locator(plt.NullLocator())

        # TODO Setting the legend will not work, since we are using the Interactive Legend Plugin

        # TODO We hide those so we can show this in
        # self.ax.set_xlabel('meters')
        # self.ax.set_ylabel('meters')
        # TODO We replaced the title with metadata so we can visualize them as we like in the HTML page
        # Visualize some metadata about the road as a text box in the upper left in axes coords
        # textstr = " ".join(["Total road length = ", str(self.road_profiler.road.total_length), " m.",
        #                     "Cell size 20m x 20m"])
        # self.ax.set_title(textstr, size=20)

    def _collect_road_metadata(self):
        """ Use this function to collect all the metadata about the roads"""
        road_metadata = dict()

        road_metadata['plot'] = dict()
        road_metadata['plot']['cell_size'] = self.CELL_SIZE

        the_road = self.road_profiler.road;
        road_metadata['road'] = dict()
        road_metadata['road']['lenght'] = the_road.total_length

        road_metadata['road']['sectors'] = dict()
        for idx, sector in enumerate(self.sectors):
            # Add here sector metadata if necessary
            road_metadata['road']['sectors'][idx] = dict()
            road_metadata['road']['sectors'][idx]['label'] = "".join(["Sector", str(idx)])

        return road_metadata

    def plot(self):
        """Generate the plot and connects the plugins for interaction but do not show it"""

        self._setup_axes()

        self._plot_grid()

        self._plot_road()

        self._plot_sectors()

        # We do not care about indivual segments. But plotting them might help during debugging
        # self._plot_road_segments()

        self._plot_car()

        # Initialize the plot by visualizing nothign more than the grid. The road geometry will be there, as I cannot
        #   hide it for the moment....
        visible_elements = [ False ] * len(list(self.labels_and_handlers.keys()))
        visible_elements[0] = True # Visualize the grid only by default

        # Configure the plugin that shows the interactive legend
        interactive_legend = plugins.InteractiveLegendPlugin(self.labels_and_handlers.values(),
                                                            list(self.labels_and_handlers.keys()),
                                                             # Setting alpha to 0.0 makes lines disappear but not patches
                                                             alpha_unsel=0.0,
                                                             alpha_over=1.5,
                                                             start_visible=visible_elements,
                                                             font_size=14,
                                                             legend_offset=(-10, 0)
                                                             )

        # Configure the plugin that export road metadata as global variable
        export_road_metadata = ExportRoadMetadataPlugin(self._collect_road_metadata())

        plugins.connect(self.fig, interactive_legend, export_road_metadata)

    def store_html_to(self, html_file):
        """ Store a string HTML to the given file"""
        print("Storing:", html_file)
        with open(html_file, 'wt') as outfile:
            outfile.write(mpld3.fig_to_html(self.fig))

    def store_json_to(self, json_file):
        """ Store a dictionary about the plot into a JSON object """
        print("Storing:", json_file)
        with open(json_file, 'wt') as outfile:
            json.dump(mpld3.fig_to_dict(self.fig), outfile)


if __name__ == '__main__':
    # This is only to illustrate how the class can be used...
    from road_profiler import setup_logging
    from logging import INFO, DEBUG

    setup_logging(INFO)
    import os
    road_geometry = []
    with open('test-data/road_01.json') as json_file:
        road_geometry = json.load(json_file)

    rv = RoadVisualizer(road_geometry)
    rv.plot()

    html_file=os.path.join('.', 'testplot.html')

    rv.store_html_to(html_file)
    ## Interactive visualization
    # mpld3.show()
