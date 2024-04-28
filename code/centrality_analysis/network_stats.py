'''
Get stats about the network

Calculates betweeness centrality, closeness centrality, and all 
shortest paths. Saves the results as their own file in 
`data/network/network_stats` so it can be referenced later

Everything in this file takes many many hours to calculate, so I
suggest just using the values calculated and saved
'''

from multiprocessing import Process, Manager # multiprocess things (otherwise its hella slow)
from shapely import wkt, distance
import networkx as nx
import pandas as pd
import numpy as np

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

    start = pd.Timestamp.now() # keep track of how long it takes

    eigcent = nx.eigenvector_centrality_numpy(G, weight='weight') # calc eigenvector centrality

    end = pd.Timestamp.now()
    print(f'Time to Find Eigenvector Centrality: {end-start}')

    eigcent_df = pd.DataFrame({'node': eigcent.keys(), 'eigenvector': eigcent.values()})

    key = 'wo' if removed else 'w'
    eigcent_df.to_csv(f'data/network_stats/eigenvector_{key}_bridge.csv', index=False)



def betweenness_centrality(G: nx.Graph, # graph
                           removed: bool # True if bridge roads are removed, False otherwise
                           ) -> None:
    print('Finding Betweenness Centrality...')

    start = pd.Timestamp.now() # keep track of how long it takes

    betcent = nx.betweenness_centrality(G, weight='length') # calc betweenness centrality

    end = pd.Timestamp.now()
    print(f'Time to Find Betweenness Centrality: {end-start}')

    betcent_df = pd.DataFrame({'node': betcent.keys(), 'betweenness': betcent.values()})

    key = 'wo' if removed else 'w'
    betcent_df.to_csv(f'data/network_stats/betweenness_{key}_bridge.csv', index=False)


def closeness_centrality(G: nx.Graph, # graph
                         removed: bool # True if bridge roads are removed, False otherwise
                         ) -> None:
    print('Finding Closeness Centrality...')

    start = pd.Timestamp.now() # keep track of how long it takes

    clocent = nx.closeness_centrality(G, distance='length') # calc closeness centrality

    end = pd.Timestamp.now()
    print(f'Time to Find Closeness Centrality: {end-start}')

    clocent_df = pd.DataFrame({'node': clocent.keys(), 'closeness': clocent.values()})

    key = 'wo' if removed else 'w'
    clocent_df.to_csv(f'data/network_stats/closeness_{key}_bridge.csv', index=False)


def node_straightness_centrality(node: str, # node to get centrality for
                                 G: nx.Graph, # graphs
                                 nodes: pd.DataFrame # all nodes 
                                 ) -> None:    
    n = len(nodes)-1 # normalization consant
    nodes = nodes.set_index('osmid')

    node_geom = nodes.loc[node, 'geometry']

    network_dists = pd.Series(nx.single_source_dijkstra(G, node, weight='length')[0]) # dists along network
    eucl_dists = distance(node_geom, nodes['geometry']) / FEET_PER_METER # euclidian distance
    dists = pd.DataFrame({'net_dist': network_dists, 'eucl_dist': eucl_dists}) # merge

    str_cent = 1/n * (dists['eucl_dist'] / dists['net_dist']).sum() # calc straightness centrality
    
    return str_cent


def nodes_straightness_centrality(nodes: pd.DataFrame, # nodes to get centrality for)
                                  G: nx.Graph, # graph
                                  all_nodes: pd.DataFrame, # all nodes
                                  i: int, # process number
                                  return_dict: dict # dict to put results in
                                  ) -> None:
    straightness = nodes.apply(node_straightness_centrality, args=(G, all_nodes))
    return_dict[i] = pd.DataFrame({'node': nodes, 'straightness': straightness})


def straightness_centrality(nodes: pd.DataFrame, # nodes to graph
                            G: nx.Graph, # graph
                            n_processes: int, # number of processes to use
                            removed: bool # if bridge has collapsed in network
                            ) -> None:
    print('Finding Straightness Centrality...')

    manager = Manager()
    return_dict = manager.dict() # store results
    jobs = [] # store multithreaded processes
    ns = np.linspace(0, len(nodes), n_processes+1, dtype=int) # ranges to multithreading

    start = pd.Timestamp.now() # keep track of how long it takes

    for i in range(n_processes): # setup and start each process
        jobs.append(Process(target=nodes_straightness_centrality, args=(nodes.loc[ns[i]:ns[i+1]-1, 'osmid'], G, nodes, i, return_dict)))
        jobs[i].start()
    
    for job in jobs: # wait for each process finishes
        job.join()
    
    strcent = pd.concat(return_dict.values())

    end = pd.Timestamp.now()
    print(f'Time to Find Straightness Centrality: {end-start}')

    key = 'wo' if removed else 'w'
    strcent.to_csv(f'data/network_stats/straightness_{key}_bridge.csv', index=False)


def shortest_average_path(G: nx.Graph,
                          removed: bool
                          ) -> None:
    print('Finding Average Shortest Path...')

    start = pd.Timestamp.now() # keep track of how long it takes

    spathlen = nx.average_shortest_path_length(G, weight='length')

    end = pd.Timestamp.now()
    print(f'Time to Find Average Shortest Path: {end-start}')

    key = 'wo' if removed else 'w'
    pd.DataFrame([spathlen], columns=['avg_spath']).to_csv(f'data/network_stats/avg_spath_{key}_bridge.csv', index=False)


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
    straightness_w_bridge = pd.read_csv('data/network_stats/straightness_w_bridge.csv')
    straightness_wo_bridge = pd.read_csv('data/network_stats/straightness_wo_bridge.csv')


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
        right=straightness_w_bridge,
        left_on='osmid',
        right_on='node'
    ).drop(columns='node').rename(columns={
        'straightness': 'straightness_w_bridge'
    }).merge(
        right=straightness_wo_bridge,
        left_on='osmid',
        right_on='node'
    ).drop(columns='node').rename(columns={
        'straightness': 'straightness_wo_bridge'
    })

    # save
    network_stats.to_csv('data/network_stats/network_stats.csv', index=False)


def main():
    print(f'Started at {pd.Timestamp.now()}')

    # read network
    roads_G = nx.read_gml('data/network/BMA_road_network.gml')
    roads_df = pd.read_csv('data/network/BMA_roads.csv')
    roads_df['u'], roads_df['v'] = roads_df['u'].astype(str), roads_df['v'].astype(str)
    
    # read files for straightness
    intersections_df = pd.read_csv('data/network/BMA_intersections.csv')
    intersections_df['geometry'] = intersections_df['geometry'].apply(wkt.loads) # convert to geometry
    intersections_df['osmid'] = intersections_df['osmid'].astype(str)

    # set weights
    calc_weight(roads_G, roads_df)

    # bridge collapse
    roads_wo_bridge_G = roads_G.copy()
    destroyed_edges = roads_df[roads_df['highway'] == 'destroyed']
    destroyed_edges = list(zip(destroyed_edges['u'], destroyed_edges['v'])) # start and end nodes
    roads_wo_bridge_G.remove_edges_from(destroyed_edges) # python multiprocessing doesn't share state

    # network stats (using multiprocessing so its not slow asf)
    eigcent_w_bridge_p = Process(target=eigenvector_centrality, args=(roads_G, False)) #setup processes
    betcent_w_bridge_p = Process(target=betweenness_centrality, args=(roads_G, False))
    clocent_w_bridge_p = Process(target=closeness_centrality, args=(roads_G, False))
    strcent_w_bridge_p = Process(target=straightness_centrality, args=(intersections_df, roads_G, 5, False))
    avg_spath_w_bridge_p = Process(target=shortest_average_path, args=(roads_G, False))

    # network stats after collapse
    eigcent_wo_bridge_p = Process(target=eigenvector_centrality, args=(roads_wo_bridge_G, True)) #setup processes
    betcent_wo_bridge_p = Process(target=betweenness_centrality, args=(roads_wo_bridge_G, True))
    clocent_wo_bridge_p = Process(target=closeness_centrality, args=(roads_wo_bridge_G, True))
    strcent_wo_bridge_p = Process(target=straightness_centrality, args=(intersections_df, roads_wo_bridge_G, 5, True))
    avg_spath_wo_bridge_p = Process(target=shortest_average_path, args=(roads_wo_bridge_G, True))

    eigcent_w_bridge_p.start() # start processes
    betcent_w_bridge_p.start()
    clocent_w_bridge_p.start()
    strcent_w_bridge_p.start()
    avg_spath_w_bridge_p.start()
    eigcent_wo_bridge_p.start()
    betcent_wo_bridge_p.start()
    clocent_wo_bridge_p.start()
    strcent_wo_bridge_p.start()
    avg_spath_wo_bridge_p.start()

    eigcent_w_bridge_p.join() # wait for it all to be done
    betcent_w_bridge_p.join()
    clocent_w_bridge_p.join()
    strcent_w_bridge_p.join()
    avg_spath_w_bridge_p.join()
    eigcent_wo_bridge_p.join()
    betcent_wo_bridge_p.join()
    clocent_wo_bridge_p.join()
    strcent_wo_bridge_p.join()
    avg_spath_wo_bridge_p.join()

    merge_centralities()
    print('Done')


if __name__ == '__main__':
    main()
