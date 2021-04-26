import sqlite3

class DBUtils():

    CREATE_DISTANCE_TABLE_SQL = """
            CREATE TABLE IF NOT EXISTS `Distance` ( 
                'road_1_id' TEXT,
                'road_2_id' TEXT,
                `name` TEXT,
                `value` DOUBLE,
                PRIMARY KEY (road_1_id, road_2_id, name)
        );
        """

    CREATE_ROAD_TABLE_SQL = """
                CREATE TABLE IF NOT EXISTS `Road` ( 
                    'road_id' TEXT,
                    'road_geometry' TEXT,
                    PRIMARY KEY (road_id)
            );
            """

    INSERT_DISTANCE_SQL = """
            INSERT INTO Distance('road_1_id', 'road_2_id', 'name', 'value')
            VALUES (?, ?, ?, ?);
    """

    INSERT_ROAD_SQL = """
                INSERT INTO Road('road_id', 'road_geometry')
                VALUES (?, ?);
        """
    
    def __init__(self, db_file):
        self._create_connection(db_file)

    def close_and_del(self):
        self.connection.close()
        # Ugly hack to ensure no one can reuse the same object ?
        del self


    def get_roads_count(self):
        COUNT_ROADS_SQL = """
            SELECT COUNT(*)
            FROM Road
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(COUNT_ROADS_SQL)
            # Return only the actual value and not the result tuple
            return cursor.fetchone()[0]

        except Exception as e:
            print("Cannot create Distance table", e)



    def _create_connection(self, db_file):
        """ create a database connection to the SQLite database
            specified by db_file
        :param db_file: database file
        :return: Connection object or None
        """
        self.connection = None
        try:
            self.connection = sqlite3.connect(db_file)
        except Exception as e:
            print("Cannot crete the connection to the database", e)

    def create_distances_table(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute(self.CREATE_DISTANCE_TABLE_SQL)
            self.connection.commit()

        except Exception as e:
            print("Cannot create Distance table", e)

    def create_roads_table(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute(self.CREATE_ROAD_TABLE_SQL)
            self.connection.commit()

        except Exception as e:
            print("Cannot create Road table", e)

    def randomly_sample_roads(self, limit, generator):
        RANDOM_SELECTION_DEEPJANUS_SQL="""
            SELECT * FROM Road 
            WHERE road_id IN (
                SELECT road_id FROM Road 
                WHERE road_id LIKE '%_deepjanus_%'
                ORDER BY RANDOM() 
                LIMIT ?
            );
        """

        RANDOM_SELECTION_ASFAULT_SQL = """
                    SELECT * FROM Road 
                    WHERE road_id IN (
                        SELECT road_id FROM Road
                        WHERE road_id LIKE '%.geometry.json' 
                        ORDER BY RANDOM() 
                        LIMIT ?
                    );
                """

        RANDOM_SELECTION_ANY_SQL = """
                           SELECT * FROM Road 
                           WHERE road_id IN (
                               SELECT road_id FROM Road 
                               ORDER BY RANDOM() 
                               LIMIT ?
                           );
                       """

        try:
            cursor = self.connection.cursor()
            if generator == "asfault":
                cursor.execute(RANDOM_SELECTION_ASFAULT_SQL, [limit])
            elif generator == "deepjanus":
                cursor.execute(RANDOM_SELECTION_DEEPJANUS_SQL, [limit])
            elif generator == "any":
                cursor.execute(RANDOM_SELECTION_ANY_SQL, [limit])
            else:
                raise Exception("Generator " + str(generator) + " is unknown")
            return cursor.fetchall()

        except Exception as error:
            print("parameterized query failed {}".format(error))

    def insert_roads_in_db(self, road_tuples):
        try:
            cursor = self.connection.cursor()
            cursor.executemany(self.INSERT_ROAD_SQL, road_tuples)
            self.connection.commit()
        except Exception as error:
            print("parameterized query failed {}".format(error))

    def insert_distances_in_db(self, distance_tuples):
        try:
            cursor = self.connection.cursor()
            cursor.executemany(self.INSERT_DISTANCE_SQL, distance_tuples)
            self.connection.commit()
            print("Storing", len(distance_tuples), "distance measurements into DB")
        except Exception as error:
            print("parameterized query failed {}".format(error))

    # Mostly DEBUG
    def count_all_samples(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM Distance")
            rows = cursor.fetchall()

            for row in rows:
                print(row)

        except Exception as error:
            print("parameterized query failed {}".format(error))

    def get_id_of_all_individuals(self, distance_name, generator):
        # Road evaluation should be simmetric but we keep everything nevertheless
        # DISTINCT should be unecessary as UNION removes duplicates?

        SELECT_ALL_ROAD_ANY_IDS = """
            SELECT road_1_id
                FROM Distance
                WHERE road_1_id IS NOT NULL AND name=?
            UNION
                SELECT road_2_id
                FROM Distance
                WHERE road_2_id IS NOT NULL AND name=?;
        """

        SELECT_ALL_ROAD_ASFAULT_IDS = """
                    SELECT road_1_id
                        FROM Distance
                        WHERE road_1_id IS NOT NULL AND name=? AND road_1_id LIKE '%.geometry.json'
                    UNION
                        SELECT road_2_id
                        FROM Distance
                        WHERE road_2_id IS NOT NULL AND name=? AND road_2_id LIKE '%.geometry.json';
                """

        SELECT_ALL_ROAD_DEEPJANUS_IDS = """
                    SELECT road_1_id
                        FROM Distance
                        WHERE road_1_id IS NOT NULL AND name=? AND road_1_id LIKE '%_deepjanus_%'
                    UNION
                        SELECT road_2_id
                        FROM Distance
                        WHERE road_2_id IS NOT NULL AND name=? AND road_2_id LIKE '%_deepjanus_%';
                """

        try:
            old_row_factory = self.connection.row_factory
            self.connection.row_factory = lambda cursor, row: row[0]

            cursor = self.connection.cursor()
            if generator == "asfault":
                cursor.execute(SELECT_ALL_ROAD_ASFAULT_IDS, (distance_name, distance_name))
            elif generator == "deepjanus":
                cursor.execute(SELECT_ALL_ROAD_DEEPJANUS_IDS, (distance_name, distance_name))
            elif generator == "any":
                cursor.execute(SELECT_ALL_ROAD_ANY_IDS, (distance_name, distance_name))
            else:
                raise Exception("Generator " + str(generator) + " is unknown")
            ids = cursor.fetchall()

            # Reset the row_factory
            self.connection.row_factory = old_row_factory

            return ids

        except Exception as error:
            print("parameterized query failed {}".format(error))

    def get_min_distance_from_set(self, distance_name, road_id, set_of_road_ids):
        """ Return the minimum distance. Raises an error if some distance measurement is missing?"""
        distances = list()
        # Individual to evaluate
        # ind_spine = get_spine(ind)
        # Compute the distance between the individual and all the elements in the solution
        #
        # SELECT count(distance) FROM Distance WHERE road_1_id == ind and name='il' in .... solutions
        # https://stackoverflow.com/questions/5766230/select-from-sqlite-table-where-rowid-in-list-using-python-sqlite3-db-api-2-0
        # args=[2,3]
        # sql="select * from sqlitetable where rowid in ({seq})".format(
        #     seq=','.join(['?']*len(args)))
        #
        # cursor.execute(sql, args)
        #
        # TODO Assumes that such distance exists !
        COUNT_DISTANCES_SQL = """
                SELECT COUNT(value)
                FROM Distance
                WHERE name = ? AND 
                    road_1_id = ? AND
                    road_2_id IN ({seq})
            """.format(seq=','.join(['?'] * len(set_of_road_ids)))

        GET_MIN_DISTANCE_SQL = """
               SELECT MIN(value) 
               FROM Distance
               WHERE name = ? AND 
                    road_1_id = ? AND
                    road_2_id IN ({seq})
            """.format(seq=','.join(['?'] * len(set_of_road_ids)))

        params = [distance_name, road_id]
        params.extend(set_of_road_ids)

        try:
            cursor = self.connection.cursor()
            # Flatten the tuple ?
            cursor.execute(COUNT_DISTANCES_SQL, params)
            # Cursor returns a tuple !
            total = cursor.fetchone()[0]

            if total < len(set_of_road_ids):
                print("WARNING: Some", (len(set_of_road_ids) - total), "distance evaluations are missing!")

            cursor = self.connection.cursor()
            cursor.execute(GET_MIN_DISTANCE_SQL, params)
            # cursor fetch returns tuples
            return cursor.fetchone()[0]

        except Exception as error:
            print("parameterized query failed {}".format(error))

    def get_roads(self, list_of_road_ids):
        """ Return the roads corresponding to the given road list """

        GET_ROADS_SQL = """
               SELECT * 
               FROM Road
               WHERE road_id IN ({seq})
            """.format(seq=','.join(['?'] * len(list_of_road_ids)))

        params = list_of_road_ids[:]

        try:
            cursor = self.connection.cursor()
            cursor.execute(GET_ROADS_SQL, params)
            # cursor fetch returns tuples
            return cursor.fetchall()

        except Exception as error:
            print("parameterized query failed {}".format(error))

    def get_distance_matrix_for(self, distance_name, dataset):
        DISTANCE_MATRIX_SQL = """  
            SELECT name, road_1_id, road_2_id, value 
            FROM Distance
            WHERE name = ? AND 
                road_1_id IN ({seq}) AND
                road_2_id IN ({seq})
            """.format(seq=','.join(['?'] * len(dataset)))

        params = [distance_name]
        params.extend(dataset)
        params.extend(dataset)

        try:
            cursor = self.connection.cursor()
            cursor.execute(DISTANCE_MATRIX_SQL, params)
            # List of tuples
            return cursor.fetchall()

        except Exception as error:
            print("parameterized query failed {}".format(error))


