import os
import json
import jaro
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats
from vincenty import vincenty
from .data_column_names import columname
from shapely.geometry import Point

# NLP tokenization:
import string

class utils_general:

    def __init__(self):

        # Global string variables for paths and relative paths:
        self.dataPath =  'J:/My Drive/political-integration/Politician Career Path/data'
        self.raw = "raw"
        self.intermediate = "intermediate"
        self.clean = "clean"

        # Relative path for datasets:
        self.districts_folder = "District Shapefiles"
        self.covid_folder = "Covid"
        self.infogob_folder = "Infogob Datasets"
        self.mef_folder = "Mef Datasets"
        self.onpe_folder = "Onpe"        

        # Access Infogob Datasets:
        self.folder_prefix_constituyente = "CONSTITUYENTE "
        self.folder_prefix_diputados = "DIPUTADOS "
        self.folder_prefix_presidencial = "PRESIDENCIAL "
        self.folder_prefix_congresal = "CONGRESAL "
        self.folder_prefix_distrital = "MUNICIPAL DISTRITAL "
        self.folder_prefix_provincial = "MUNICIPAL PROVINCIAL "
        self.folder_prefix_regional = "REGIONAL "

        self.years_general = [2001, 2006, 2011, 2016, 2016]
        self.years_local = [2002, 2006, 2010, 2014, 2018]

        # File Names: Procesos
        self.municipales = "ERM"
        self.generales   = "EG"

        # File Name: Suffix
        self.autoridades = "Autoridades"
        self.candidatos = "Candidatos"
        self.padron = "Padron"
        self.resultados = "Resultados"

        # File Name: Geo Nivel
        self.distrital = "Distrital"
        self.provincial = "Provincial"
        self.regional = "Regional"
        self.presidencial = "Presidencial"
        self.congresal = "Congresal"

        self.year_default = 2018

        self.districts_ubigeo = "distritos-peru.json"


        # Sub Classes:

        self.columname = columname()


    # Functions to read dataframes



    def removeTildesFromVocals(self, nn):

        nn = (nn.replace('á','a')
                .replace('é','e')
                .replace('í','i')
                .replace('ó','o')
                .replace('ú','u')
                .replace('Á','A')
                .replace('É','E')
                .replace('Í','I')
                .replace('Ó','O')
                .replace('Ú','U')
                .replace('Ñ','N')
                .replace('Ń','N')
                )

        return nn


    def load_district_info(self):
        
        ubigeo_info = json.load(open(os.path.join(self.dataPath, 
                                                    self.raw, 
                                                    self.districts_folder, 
                                                    self.districts_ubigeo)
                                                    ))

        print("Number of districts:", len(ubigeo_info))

        district_info = []
        for ii in range(0,len(ubigeo_info)):
            district_ii = [ubigeo_info[ii]['fields'][self.columname.District.region].lower(),
                                    ubigeo_info[ii]['fields'][self.columname.District.provincia].lower(),
                                        ubigeo_info[ii]['fields'][self.columname.District.distrito].lower(),
                                            ubigeo_info[ii]['fields'][self.columname.District.ubigeo].lower()]
            district_info.append(district_ii)

        district_data_frame = pd.DataFrame(district_info).rename(columns={0:self.columname.region,  
                                                                            1:self.columname.provincia,  
                                                                            2:self.columname.distrito,  
                                                                            3:self.columname.ubigeo}
                                                                            )


        return district_data_frame

    
    def obtain_ubigeo(self, Region, Provincia, Distrito, district_data_frame):

        # Recode stuff, very trivial, if nan, replace to empty string:
        if type(Region) == float: Region = ""
        if type(Provincia) == float: Provincia = ""
        if type(Distrito) == float: Distrito = ""

        Region = Region.lower()
        Provincia = Provincia.lower()
        Distrito = Distrito.lower()


        # First pass ... get collection of possible regions:
        # Search Region and keep obs from that region: 
        region_name_collection = list(set(district_data_frame["Region"]))

        if Region in region_name_collection:
            region_dd = district_data_frame.query(self.columname.region + "== @Region")
        else:
            idx = np.argmax([jaro.jaro_winkler_metric(Region, x) for x in region_name_collection])
            region_best_match = region_name_collection[idx]
            region_dd = district_data_frame.query(self.columname.region + "== @region_best_match")


        # Search Provincia and keep obs from that provincia: 
        province_name_collection = list(set(region_dd[self.columname.provincia]))

        if Provincia in province_name_collection:
            provincia_dd = region_dd.query(self.columname.provincia + "== @Provincia")
        else:
            idx = np.argmax([jaro.jaro_winkler_metric(Provincia, x) for x in province_name_collection])
            provincia_best_match = province_name_collection[idx]
            provincia_dd = region_dd.query(self.columname.provincia + "== @provincia_best_match")


        # Search Distrito within provincia:
        distrito_name_collection = list(set(provincia_dd[self.columname.distrito]))

        if Distrito in distrito_name_collection:
            distrito_dd = provincia_dd.query(self.columname.distrito + "== @Distrito")
        else:
            idx = np.argmax([jaro.jaro_winkler_metric(Distrito, x) for x in distrito_name_collection])
            distrito_best_match = distrito_name_collection[idx]
            distrito_dd = provincia_dd.query(self.columname.distrito + "== @distrito_best_match")


        # Return region, provincia, distrito and ubigeo

        location_code = []

        ubigeo = distrito_dd.reset_index(drop=True).loc[0,self.columname.ubigeo]

        # Region is always there ...
        Region = distrito_dd.reset_index(drop=True).loc[0,self.columname.region]
        
        location_code.append(ubigeo[0:2])


        if Provincia != "":
            Provincia = distrito_dd.reset_index(drop=True).loc[0,self.columname.provincia]
            location_code.append(ubigeo[2:4])
        else:
            location_code.append("")


        if Distrito != "":
            Distrito = distrito_dd.reset_index(drop=True).loc[0,self.columname.distrito]
            location_code.append(ubigeo[4:6])
        else:
            location_code.append("")

        ubigeo = "".join(location_code)

        return ubigeo, location_code, Region, Provincia, Distrito


    def obtain_ubigeo_info(self, ubig):

        district_data_frame = self.load_district_info()

        district_query = (district_data_frame
                            .loc[district_data_frame[self.columname.ubigeo] == ubig, :]
                            .reset_index(drop=True)
                            )

        region = district_query[self.columname.region].loc[0]

        provincia = district_query[self.columname.provincia].loc[0]

        distrito = district_query[self.columname.distrito].loc[0]

        return region, provincia, distrito


    def input_ubigeo_to_dataframe(self, dataset, district_variables):

        """
        This function inputs ubigeo to a dataset using district variables.

        Inputs:
            dataset: pandas dataframe
            district_variables: list with district variables (region, provincia, distrito)

        Output:
            dataset: pandas dataframe with ubigeo variable
        """

        district_data_frame = self.load_district_info()# district_variables = [self.columname.region,self.columname.provincia,self.columname.distrito]

        # Transform dataset to district level (I do this to input just 1.8K obs into the scanner):
        dataset_district_level = (dataset.groupby(district_variables)
                                    .first()
                                    .reset_index()
                                    )

        # Scan district to recover ubigeo:
        dataset_district_level[self.columname.ubigeo] = list(map(
                            lambda x, y, z: self.obtain_ubigeo(x , y , z, district_data_frame)[0], 
                                    dataset_district_level[district_variables[0]],
                                    dataset_district_level[district_variables[1]],
                                    dataset_district_level[district_variables[2 ]]
                                    )
                                    )

        # Build crosswalk using district_data_frame:
        crosswalk = dataset_district_level[[district_variables[0],
                                            district_variables[1], 
                                            district_variables[2], 
                                            self.columname.ubigeo]]

        dataset = dataset.merge(crosswalk, on = district_variables, how='left')

        return dataset

    def get_political_party_code(self):

        results_distrital = pd.read_csv(os.path.join(self.dataPath,
                                                        self.intermediate,
                                                        "2014_voting_results_with_location.csv"))

        # Obtain dictionary with just name and code of the party
        political_group_code = (results_distrital.loc[:,[self.columname.Onpe.Results.political_group,
                                    self.columname.Onpe.Results.political_organization_code]]
                                    .sort_values(self.columname.Onpe.Results.political_organization_code)
                                    .groupby(self.columname.Onpe.Results.political_group)
                                    .first()
                                    .to_dict()
                                    )
        
        political_group_code = political_group_code[self.columname.Onpe.Results.political_organization_code]

        return political_group_code

    def input_political_party_code(self, df):

        political_group_code = self.get_political_party_code()

        political_party_name_master = np.array(list(political_group_code.keys()))

        political_party_code_master = np.array(list(political_group_code.values()))

        party_name_x = df[self.columname.Onpe.Results.political_group].unique()    

        idx_list = []

        jaro_dist_list = []

        # Run across all party names of the target databases:
        
        for x in party_name_x:

            jaro_metrics = [jaro.jaro_winkler_metric(x, y) for y in political_party_name_master]
            
            jaro_max_dist = np.max(jaro_metrics)

            jaro_dist_list.append(jaro_max_dist)

            idx = np.argmax(jaro_metrics)

            idx_list.append(idx)


        jaro_dist_list = np.array(jaro_dist_list)

        party_name_match = (pd.DataFrame({'jaro_dist': jaro_dist_list, 
                                            self.columname.Onpe.Results.political_group:party_name_x, 
                                            'name_y':political_party_name_master[idx_list], 
                                            self.columname.Onpe.Results.political_organization_code:political_party_code_master[idx_list]})
                                            .sort_values('jaro_dist'))
        # These guys here are wrong ... 

        party_name_match = (party_name_match.sort_values('jaro_dist')
                                            .groupby(self.columname.Onpe.Results.political_group)
                                            .first()
                                            .reset_index(False)
                                            )

        # TODO: Fix names for the ones below .93 jaro.
        party_name_match = party_name_match.loc[party_name_match['jaro_dist'] > .93]

        df = df.merge(party_name_match[[self.columname.Onpe.Results.political_group,self.columname.Onpe.Results.political_organization_code]], 
                                on = self.columname.Onpe.Results.political_group)
        
        return df



    def get_fixed_effects(self, df, location="Ubigeo"):

        location_FE = pd.get_dummies(df[location].astype("category"))   

        party_FE = pd.get_dummies(df[self.columname.Onpe.Results.political_organization_code])

        ubic_FE = pd.get_dummies(df[self.columname.Onpe.Results.ballot_place])

        # nComp_FE = pd.get_dummies(df['nCompetitors'])

        return location_FE, party_FE, ubic_FE


    def add_stars(self, x, p):

        if p < 0.01:
            x = str(x) + "***"

        elif 0.05 > p >= 0.01:
            x = str(x) + "**"

        elif 0.1 > p >= 0.05:
            x = str(x) + "*"

        else:
            x = str(x) 

        return x



    def get_distance_in_kilometers(self, df, lat_lon_a, lat_lon_b):

        df = df.copy()

        df['km_distance'] = df.apply(
            (lambda row: vincenty(
                (row[lat_lon_a[0]], row[lat_lon_a[1]]),
                (row[lat_lon_b[0]], row[lat_lon_b[1]]), miles = False
            )),
            axis=1
        )
        
        return df['km_distance']
    

    def read_ubigeo_shapefile(self):

        """
        This function reads a shapefile and returns a geopandas dataframe.

        :param path: The path where the shapefile is located
        :param filename: The name of the shapefile
        :return: A geopandas dataframe with the shapefile
        """

        shapefile = gpd.read_file(os.path.join(self.dataPath, self.raw, self.districts_folder, 'LIMITE_DISTRITAL_2020_INEI_geogpsperu_juansuyo_931381206.shp'))

        shapefile  = shapefile.drop(columns=['CONTACTO','WHATSAPP','DESCARGAR','DESCRIPCIO'])

        shapefile['UBIGEO'] = shapefile['UBIGEO'].astype(int)

        return shapefile
    
    def find_ubigeo_using_centroid(self, df_centroids, df_polygons, centroid_variables = ('longitud_ccpp','latitud_ccpp')):

        """
        This function takes two dataframes, one with points and one with polygons, and returns a dataframe with the
        points and the identifier of the polygon they are in.
        
        :param df_centroids: A dataframe with points and their coordinates
        :param df_polygons: A dataframe with polygons and their coordinates
        :param centroid_variables: A tuple with the names of the columns in df_centroids that contain the coordinates
        :return: A dataframe with the points and the identifier of the polygon they are in
        
        """

        longitud, latitud = centroid_variables

        # Convert your df_centroids to a GeoDataFrame
        df_centroids['geometry'] = [Point(xy) for xy in zip(df_centroids[longitud], df_centroids[latitud])]

        # Load the polygons and convert to a GeoDataFrame:
        gdf_centroids = gpd.GeoDataFrame(df_centroids, geometry='geometry')

        # Ensure df_polygons is a GeoDataFrame
        gdf_polygons = gpd.GeoDataFrame(df_polygons, geometry='geometry')

        # Perform a spatial join
        df_joined = gpd.sjoin(gdf_centroids, gdf_polygons, op='within')

        return df_joined  # 'ubigeo' column of df_joined now has the identifiers for the polygon each point is in

