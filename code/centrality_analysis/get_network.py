'''
Gets the network from OSMnx and saves it

Only saves the largest connected component, not the other bits

Save as edge/node csvs and a GML file. GML files is used for analysis and 
'''

import networkx as nx
import osmnx as ox # Open Stree Maps API


def main():
    ## get network
    places =[ # Baltimore Metro Area
        # {'county': 'Anne Arundel County', 'state': 'Maryland', 'country': 'USA'},
        # {'city': 'Baltimore', 'state': 'Maryland', 'country': 'USA'},
        # {'county': 'Baltimore County', 'state': 'Maryland', 'country': 'USA'},
        # {'county': 'Carroll County', 'state': 'Maryland', 'country': 'USA'},
        # {'county': 'Harford County', 'state': 'Maryland', 'country': 'USA'},
        # {'county': 'Howard County', 'state': 'Maryland', 'country': 'USA'},
        {'county': "Queen Anne's County", 'state': 'Maryland', 'country': 'USA'}
    ]
    total_roads_G = ox.graph_from_place( # total raods graph
        places,
        network_type='drive',
        simplify=True
    )

    # filter to LCC
    roads_G = total_roads_G.subgraph( # roads graph
        max(
            nx.strongly_connected_components(total_roads_G),
            key=len
        )
    )
    print(f'Number of Nodes Removed by Using LCC: {len(total_roads_G) - len(roads_G)}')

    # save graphs (as csv)
    gdfs = ox.graph_to_gdfs(roads_G)

    intersections_gdf = gdfs[0].to_crs('EPSG:2893').reset_index() # nodes
    roads_gdf = gdfs[1].to_crs('EPSG:2893').reset_index() # edges
    # crs is a coordinate system measured in feet centered in maryland 
    # so we can use euclidian distance on the points with minimal error
    intersections_gdf['osmid'] = intersections_gdf['osmid'].astype(str)
    roads_gdf['u'], roads_gdf['v'] = roads_gdf['u'].astype(str), roads_gdf['v'].astype(str)

    print(f'Number of Bridge Edges: {(roads_gdf['highway'] == 'destroyed').sum()}')

    intersections_gdf.to_csv('data/network/BMA_intersections.csv', index=False)
    roads_gdf.to_csv('data/network/BMA_roads.csv', index=False)

    # save graphs (as gml)
    nx.set_edge_attributes(roads_G, 0, 'geometry')
    nx.set_node_attributes(roads_G, 0, 'geometry')
    # gml files break if theres a linestring or point object
    # i dont think you can globally remove an attribute, so this is
    # the best option

    nx.write_gml(roads_G, 'data/network/BMA_road_network.gml')


if __name__ == '__main__':
    main()
