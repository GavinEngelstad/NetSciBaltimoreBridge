'''
Get stats about the network

Calculates betweeness centrality, closeness centrality, and all 
shortest paths. Saves the results as their own file in 
`data/network/network_stats` so it can be referenced later

Everything in this file takes many many hours to calculate, so I
suggest just using the values calculated and saved
'''

from multiprocessing import Process # multiprocess things (otherwise its hella slow)
from shapely import distance
from shapely import wkt
import networkx as nx
import datetime as dt # track how long it takes
import pandas as pd

FEET_PER_METER = 3.280839895 # conversion rate


def calc_weight(G: nx.Graph, # graph
                df: pd.DataFrame # dataframe including the road type
                ) -> None:
    from weight import weights # dictionary of road type: weight

    df['weight'] = 1. # make sure there are no nan issues
    for val in weights.keys(): # update weights in gdf
        df.loc[df['highway'] == val, 'weight'] = weights[val]
    
    # set attributes
    attr = df[['u', 'v', 'key', 'weight']].set_index(['u', 'v', 'key']).to_dict('index')

    nx.set_edge_attributes(G, attr)


def eigenvector_centrality(G: nx.Graph, # graph
                           removed: bool # True if bridge roads are removed, False otherwise
                           ) -> None:
    print('Finding Eigenvector Centrality...')

    start = dt.datetime.now() # keep track of how long it takes

    eigcent = nx.eigenvector_centrality_numpy(G, weight='weight') # calc eigenvector centrality

    end = dt.datetime.now()
    print(f'Time to Find Eigenvector Centrality: {end-start}')

    eigcent_df = pd.DataFrame({'node': eigcent.keys(), 'eigenvector': eigcent.values()})

    key = 'wo' if removed else 'w'
    eigcent_df.to_csv(f'data/network_stats/eigenvector_{key}_bridge.csv', index=False)



def betweenness_centrality(G: nx.Graph, # graph
                           removed: bool # True if bridge roads are removed, False otherwise
                           ) -> None:
    print('Finding Betweenness Centrality...')

    start = dt.datetime.now() # keep track of how long it takes

    betcent = nx.betweenness_centrality(G, weight='length') # calc betweenness centrality

    end = dt.datetime.now()
    print(f'Time to Find Betweenness Centrality: {end-start}')

    betcent_df = pd.DataFrame({'node': betcent.keys(), 'betweenness': betcent.values()})

    key = 'wo' if removed else 'w'
    betcent_df.to_csv(f'data/network_stats/betweenness_{key}_bridge.csv', index=False)


def closeness_centrality(G: nx.Graph, # graph
                         removed: bool # True if bridge roads are removed, False otherwise
                         ) -> None:
    print('Finding Closeness Centrality...')

    start = dt.datetime.now() # keep track of how long it takes

    clocent = nx.closeness_centrality(G, distance='length') # calc closeness centrality

    end = dt.datetime.now()
    print(f'Time to Find Closeness Centrality: {end-start}')

    clocent_df = pd.DataFrame({'node': clocent.keys(), 'closeness': clocent.values()})

    key = 'wo' if removed else 'w'
    clocent_df.to_csv(f'data/network_stats/closeness_{key}_bridge.csv', index=False)


def all_shortest_paths(G: nx.Graph, # graph
                       removed: bool # True if bridge roads are removed, False otherwise
                       ) -> None:
    print('Finding Shortest Paths...')

    start = dt.datetime.now() # keep track of how long it takes

    spaths = dict(nx.all_pairs_dijkstra_path_length(G, weight='length')) # calc length of all closest paths

    end = dt.datetime.now()
    print(f'Time to Find All Shortest Paths: {end-start}')

    start = dt.datetime.now() # keep track of how long it takes

    reform = {(outerKey, innerKey): values for outerKey, innerDict in spaths.items() for innerKey, values in innerDict.items()}
    # Dictionary nested with each node indexeing to all other nodes, then those point to distance
    # we reform to get node-node pairs pointing to the distance between them
    i = pd.MultiIndex.from_tuples(reform.keys(), names=['u', 'v']) # index
    spaths_df = pd.DataFrame({'path_len': reform.values()}, index=i)

    end = dt.datetime.now()
    print(f'Time to Setup All Shortest Paths to be Saved: {end-start}')

    key = 'wo' if removed else 'w'
    spaths_df.to_csv(f'data/network_stats/spaths_length_{key}_bridge.csv')


def straightness_file(intersections_df) -> pd.DataFrame:
    # setup straightness calculation
    spaths_df = pd.read_csv('data/network_stats/spaths_length_w_bridge.csv').rename(columns={ # shortest paths w bridge
        'path_len': 'path_len_w_bridge'
    }).merge(
        right=pd.read_csv('data/network_stats/spaths_length_w_bridge.csv').rename(columns={ # shortest paths wo bridge
            'path_len': 'path_len_wo_bridge'
        }),
        on=['u', 'v']
    ).merge(
        right=intersections_df[['osmid', 'geometry']].rename(columns={
            'geometry': 'u_geometry'
        }),
        left_on='u',
        right_on='osmid'
    ).merge(
        right=intersections_df[['osmid', 'geometry']].rename(columns={
            'geometry': 'v_geometry'
        }),
        left_on='v',
        right_on='osmid'
    ).drop(columns=['osmid_x', 'osmid_y'])

    return spaths_df


def straightness_centrality(intersections_df: pd.DataFrame # edge shortest paths/euclidian distances
                            ) -> None:
    print('Making Straightness Centrality DataFrame...')
    df = straightness_file(intersections_df)

    print('Finding Straightness Centrality...')

    start = dt.datetime.now() # keep track of how long it takes

    # get distance 
    df['dist'] = distance(df['u_geometry'], df['v_geometry']) / FEET_PER_METER # convert to meters

    # get ratios
    df['ratio_w_bridge'] = df['dist'] / df['path_len_w_bridge']
    df['ratio_wo_bridge'] = df['dist'] / df['path_len_wo_bridge']

    # sum
    strcent_df = df.set_index(['u', 'v'])[['ratio_w_bridge', 'ratio_wo_bridge']].groupby(
        by='u'
    ).sum().reset_index().rename(columns={ # sum straightness
        'u': 'node',
        'ratio_w_bridge': 'straightness_w_bridge',
        'ratio_wo_bridge': 'straightness_wo_bridge',
    })
    n = len(strcent_df)
    strcent_df[['straightness_w_bridge', 'straightness_wo_bridge']] = 1/(n-1) * strcent_df[['straightness_w_bridge', 'straightness_wo_bridge']] # straightness equation

    end = dt.datetime.now()
    print(f'Time to Find Straightness Centrality: {end-start}')

    strcent_df.to_csv(f'data/network_stats/straightness.csv', index=False)


def merge_centralities():
    print('Merging Centrality Files...')

    # read files
    intersections = pd.read_csv('data/network/BMA_intersections.csv')
    betweenness_w_bridge = pd.read_csv('data/network_stats/betweenness_w_bridge.csv')
    betweenness_wo_bridge = pd.read_csv('data/network_stats/betweenness_wo_bridge.csv')
    closeness_w_bridge = pd.read_csv('data/network_stats/closeness_w_bridge.csv')
    closeness_wo_bridge = pd.read_csv('data/network_stats/closeness_wo_bridge.csv')
    eigenvector_w_bridge = pd.read_csv('data/network_stats/eigenvector_w_bridge.csv')
    eigenvector_wo_bridge = pd.read_csv('data/network_stats/eigenvector_wo_bridge.csv')
    straightness = pd.read_csv('data/network_stats/straightness.csv')

    # merge
    network_stats = intersections.merge(
        right=betweenness_w_bridge,
        left_on='osmid',
        right_on='node'
    ).drop(columns='node').rename(columns={
        'betweenness': 'betweenness_w_bridge'
    }).merge(
        right=betweenness_wo_bridge,
        left_on='osmid',
        right_on='node'
    ).drop(columns='node').rename(columns={
        'betweenness': 'betweenness_wo_bridge'
    }).merge(
        right=closeness_w_bridge,
        left_on='osmid',
        right_on='node'
    ).drop(columns='node').rename(columns={
        'closeness': 'closeness_w_bridge'
    }).merge(
        right=closeness_wo_bridge,
        left_on='osmid',
        right_on='node'
    ).drop(columns='node').rename(columns={
        'closeness': 'closeness_wo_bridge'
    }).merge(
        right=eigenvector_w_bridge,
        left_on='osmid',
        right_on='node'
    ).drop(columns='node').rename(columns={
        'eigenvector': 'eigenvector_w_bridge'
    }).merge(
        right=eigenvector_wo_bridge,
        left_on='osmid',
        right_on='node'
    ).drop(columns='node').rename(columns={
        'eigenvector': 'eigenvector_wo_bridge'
    }).merge(
        right=straightness,
        left_on='osmid',
        right_on='node'
    ).drop(columns='node')

    # save
    network_stats.to_csv('data/network_stats/network_stats.csv', index=False)


def main():
    print(f'Started at {dt.datetime.now()}')

    # read network
    roads_G = nx.read_gml('data/network/BMA_road_network.gml')
    roads_df = pd.read_csv('data/network/BMA_roads.csv')
    roads_df['u'], roads_df['v'] = roads_df['u'].astype(str), roads_df['v'].astype(str)
    
    # read files for straightness
    intersections_df = pd.read_csv('data/network/BMA_intersections.csv')
    intersections_df['geometry'] = intersections_df['geometry'].apply(wkt.loads) # convert to geometry

    # set weights
    calc_weight(roads_G, roads_df)

    # bridge collapse
    roads_wo_bridge_G = roads_G.copy()
    destroyed_edges = roads_df[roads_df['highway'] == 'destroyed']
    destroyed_edges = list(zip(destroyed_edges['u'], destroyed_edges['v'])) # start and end nodes
    roads_wo_bridge_G.remove_edges_from(destroyed_edges) # python multiprocessing doesn't share state

    # network stats (using multiprocessing so its not slow asf)
    eigcent_w_bridge_p = Process(target=eigenvector_centrality, args=(roads_G, False)) #setup processes
    # betcent_w_bridge_p = Process(target=betweenness_centrality, args=(roads_G, False))
    # clocent_w_bridge_p = Process(target=closeness_centrality, args=(roads_G, False))
    # spaths_w_bridge_p = Process(target=all_shortest_paths, args=(roads_G, False))

    # network stats after collapse
    eigcent_wo_bridge_p = Process(target=eigenvector_centrality, args=(roads_wo_bridge_G, True)) #setup processes
    # betcent_wo_bridge_p = Process(target=betweenness_centrality, args=(roads_wo_bridge_G, True))
    # clocent_wo_bridge_p = Process(target=closeness_centrality, args=(roads_wo_bridge_G, True))
    # spaths_wo_bridge_p = Process(target=all_shortest_paths, args=(roads_wo_bridge_G, True))

    # stratcent_p = Process(target=straightness_centrality, args=(intersections_df,))

    eigcent_w_bridge_p.start() # start processes
    # betcent_w_bridge_p.start()
    # clocent_w_bridge_p.start()
    # spaths_w_bridge_p.start()
    eigcent_wo_bridge_p.start()
    # betcent_wo_bridge_p.start()
    # clocent_wo_bridge_p.start()
    # spaths_wo_bridge_p.start()

    # wait for shortest paths to calculate, then find straightness
    # spaths_w_bridge_p.join()
    # spaths_wo_bridge_p.join()

    # straightness centrality
    # stratcent_p.start()

    eigcent_w_bridge_p.join() # wait for it all to be done
    # betcent_w_bridge_p.join()
    # clocent_w_bridge_p.join()
    eigcent_wo_bridge_p.join()
    # betcent_wo_bridge_p.join()
    # clocent_wo_bridge_p.join()
    # stratcent_p.join()

    # merge_centralities()
    print('Done')


if __name__ == '__main__':
    main()
