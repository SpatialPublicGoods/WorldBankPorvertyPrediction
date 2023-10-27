import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jaro
from vincenty import vincenty

import shapely.geometry
import geopandas as gpd

# In this script we are going to create the points of reference for the 
# Google API queries

# Open shapefile and xls with segments:

date = '2022-09-09'

data_path = "J:/My Drive/political-integration/Politician Career Path/data"

git_path = "C:/Users/franc/OneDrive/Documents/GitHub/Chicagobooth/spatial_public_goods"

#%% Set functions to build grid: 


def load_and_arrange_shapfiles(data_path, year = 2014):

    '''
    Load files...
    '''

    # Convert to UTM (For Peru is epsg:32718)
    major_home_data = (pd.read_csv(os.path.join(data_path, 'clean', 'spatial_analysis_' + date + '.csv'))
                                .query('Proceso == ' + str(year))
                                .groupby('Ubigeo')
                                .first()
                                .reset_index()
                                .loc[:,['Ubigeo', 'lat_candidate', 'lon_candidate']])

    district_shapefile = gpd.read_file(os.path.join(data_path, 'raw', 'District Shapefiles', "LIMITE_DISTRITAL_2020_INEI_geogpsperu_juansuyo_931381206.shp"))

    district_shapefile['UBIGEO'] = district_shapefile['UBIGEO'].astype(int)

    district_shapefile = district_shapefile.merge(major_home_data, left_on = 'UBIGEO', right_on = 'Ubigeo')

    district_shapefile['centroid_home'] = [shapely.geometry.Point(x,y) for x, y in zip(district_shapefile['lon_candidate'],district_shapefile['lat_candidate'])]

    district_shapefile_polygon = (district_shapefile.loc[:, ['Ubigeo', 'geometry']]
                                                    .set_geometry('geometry')
                                                    .to_crs({'init': 'epsg:32718'})
                                                    .set_index('Ubigeo')
                                                    )

    district_shapefile_home = (district_shapefile.loc[:, ['Ubigeo', 'centroid_home']]
                                                    .set_geometry('centroid_home')
                                                    .to_crs({'init': 'epsg:32718'})
                                                    .set_index('Ubigeo')
                                                    )

    district_shapefile_home = gpd.clip(district_shapefile_home,district_shapefile_polygon) #.to_crs(district_shapefile_polygon.crs)

    
    return district_shapefile_home, district_shapefile_polygon



def draw_hexagon(x, y, side, recenter_x=0, recenter_y=0):

    '''
    Side: This is the size of the hexagon.
    '''

    vertex_1 = x + side * np.sin(np.pi/2) + recenter_x, y + side * np.cos(np.pi/2) + recenter_y
    vertex_2 = x + side * np.sin(np.pi/6) + recenter_x, y + side * np.cos(np.pi/6) + recenter_y
    vertex_3 = x + side * np.sin(11 * np.pi/6) + recenter_x, y + side * np.cos(11 * np.pi/6) + recenter_y
    vertex_4 = x + side * np.sin(3 * np.pi/2) + recenter_x, y + side * np.cos(3 * np.pi/2) + recenter_y
    vertex_5 = x + side * np.sin(7 * np.pi/6) + recenter_x, y + side * np.cos(7 * np.pi/6) + recenter_y
    vertex_6 = x + side * np.sin(5 * np.pi/6) + recenter_x, y + side * np.cos(5 * np.pi/6) + recenter_y
    
    hex_points = [vertex_1, vertex_2, vertex_3, vertex_4, vertex_5, vertex_6]

    hexagon = shapely.geometry.Polygon(hex_points)

    return hexagon


def iterate_over_grid(w, h, cell_size, recenter_x=0, recenter_y=0):

    grid_x_pixels =  w
    grid_y_pixels =  h

    sep_x = 3 * cell_size  # Horizontal Separation 
    sep_y = .86 * cell_size # Vertical Separation

    grid_x = int(grid_x_pixels / sep_x) + 1
    grid_y = int(grid_y_pixels / sep_y) + 1

    # Draw the Grid
    current_x = w/2.0 - grid_x_pixels/2.0
    current_y = h/2.0 - grid_y_pixels/2.0

    hexagon_cells = []

    for i in range(grid_y):
        if (i % 2 == 0):
            current_x += 1.5 * cell_size
        for j in range(grid_x):
            hexagon = draw_hexagon(current_x, current_y, cell_size, recenter_x, recenter_y)
            hexagon_cells.append(hexagon)
            
            current_x += sep_x
        current_x = w/2.0 - grid_x_pixels/2.0
        current_y += sep_y

    return hexagon_cells


def get_point_recentering_coordinates(cells,temp_home):
    
    cells['hexagon_home'] = cells.contains(temp_home.centroid.iloc[0])

    x_hex = cells.query('hexagon_home == True').centroid.x.iloc[0]
    y_hex = cells.query('hexagon_home == True').centroid.y.iloc[0]

    x_home = temp_home.centroid_home.x.iloc[0]
    y_home = temp_home.centroid_home.y.iloc[0]

    x_dif = x_home - x_hex
    y_dif = y_home - y_hex 

    return x_dif, y_dif


def main():

    # Start running code:

    district_shapefile_home, district_shapefile_polygon = load_and_arrange_shapfiles(data_path)

    df = pd.DataFrame()

    # Size of cell
    cell_size = 1000 # map is projected in metres so 1kmx1km

    for uu in district_shapefile_home.index:

        # Here we start loop to create the points for each segment (it is faster
        # to do it by segment and more precise)

        temp_poly = district_shapefile_polygon.loc[[uu]] # segment number i 

        temp_home = district_shapefile_home.loc[[uu]] # segment number i 

        polygon_cointains_home = temp_poly.contains(temp_home).iloc[0]

        if polygon_cointains_home == False:
            continue

        # Now create fishnet (this is going to be useful when 

        xmin, ymin, xmax, ymax = temp_poly.total_bounds

        w = xmax - xmin + cell_size/2
        h = ymax - ymin + cell_size/2

        grid_cells = iterate_over_grid(w, h, cell_size, xmin, ymin)
    
        # Create gridcell polygon

        cells = gpd.GeoDataFrame(grid_cells, columns=['geometry'], 
                                    crs = district_shapefile_polygon.crs)

        x_dif, y_dif = get_point_recentering_coordinates(cells,temp_home)
        
        cells = cells.translate(xoff=x_dif, yoff=y_dif)

        cells = gpd.clip(cells, temp_poly).to_crs(district_shapefile_polygon.crs) # clip to segment
            
        # cells['UBIGEO'] = uu  

        
        temp_cells = pd.DataFrame({'geometry':cells})
        temp_cells['Ubigeo'] = uu
        temp_cells['hexagon_id'] = range(1, temp_cells.shape[0]+1)

        df = df.append(temp_cells, ignore_index = True)
        
        print("Ubigeo number: " + str(uu) + " -- done")
        

    # Convert Dataframe to Geopandas:
    df_gpd = gpd.GeoDataFrame(df, columns=['geometry', 'Ubigeo','hexagon_id'], 
                                    crs = district_shapefile_polygon.crs)

    # Save Geopandas to shp file ...
    # df_gpd.to_file(os.path.join(data_path, 'intermediate', 'fishnet_centered_2014.shp'))

    # Add trailing zeros to ubigeo
    df_gpd['Ubigeo'] = [str(int(x)).rjust(6, '0') for x in df_gpd['Ubigeo']]
    district_shapefile_home.index = [str(int(x)).rjust(6, '0') for x in district_shapefile_home.index]

    # Plot and save region of Arequipa to show how it will look like:
    base = df_gpd.loc[df_gpd['Ubigeo'].str[0:4] == '1201' ].plot(color='white', edgecolor='black', linewidth=.1)
    district_shapefile_home.loc[district_shapefile_home.index.str[0:4] == '1201'].plot(ax= base, color='red', edgecolor='red', markersize=.1)
    plt.savefig(os.path.join('..', 'figures', 'figA1_grid_trujillo.pdf'))

#%% Run code:

main()

