'''
Get stats about the network

Calculates betweeness centrality, closeness centrality, and all 
shortest paths. Saves the results as their own file in 
`data/network/network_stats` so it can be referenced later

Everything in this file takes about 4 hours to calculate, so I
suggest just using the values calculated and saved
'''

from multiprocessing import Process, Manager # multiprocess things (otherwise its hella slow)
from shapely import wkt, distance
import igraph as ig # written in c so faster than networkx
import pandas as pd
import numpy as np

FEET_PER_METER = 3.280839895 # conversion rate


def calc_weight(G: ig.Graph # road graph
                ) -> None:
    from weight import weights # dictionary of road type: weight

    edge_weights = pd.Series( # create weights series
        G.es.get_attribute_values('highway')
    ).map(
        weights
    ).replace(np.nan, 1)

    G.es.set_attribute_values('weight', edge_weights) # add weights to graph


def add_geometry(G: ig.Graph,
                 nodes: pd.DataFrame
                 ) -> None:
    node_geoms = pd.Series( # create geom series
        G.vs.get_attribute_values('label')
    ).map(
        nodes.set_index('osmid')['geometry'].to_dict()
    )

    G.vs.set_attribute_values('geometry', node_geoms) # add geoms to graph


def node_straightness(i: int, # node number
                      G: ig.Graph # roads graph
                      ) -> float:
    # straightness cent = 1/(n-1) * eucl_dists / net_dists
    net_dists = pd.Series(G.distances(i, weights='length')[0]) # network distance
    eucl_dists = pd.Series(distance(G.vs[i]['geometry'], G.vs.get_attribute_values('geometry'))) / FEET_PER_METER # euclidian distance
    n = G.vcount() # number of nodes

    return 1/(n-1) * (eucl_dists / net_dists).sum() # straightness centrality def


def straightness(G: ig.Graph # road graph
                 ) -> pd.Series:
    return pd.Series(np.arange(G.vcount())).apply(node_straightness, args=[G]) # calculate centrality for each node and make it into a series


def get_centralities(G: ig.Graph, # raod graph
                     removed: bool, # whether the bridge has been removed
                     return_dict: dict # multithreading return values
                     ) -> pd.DataFrame:
    print(f'Started Getting Centralities at {pd.Timestamp.now()}')
    start_cents = pd.Timestamp.now()

    key = 'wo' if removed else 'w'
    cents = pd.DataFrame({'node': G.vs.get_attribute_values('label')}) # dataframe to keep track of centralities

    # eigenvector centrality
    start = pd.Timestamp.now()
    cents[f'eigenvector_{key}_bridge'] = pd.Series(G.eigenvector_centrality(weights='weight')) 
    print(f'Found Eigenvector Centrality in {pd.Timestamp.now() - start}')

    # closeness centrality
    start = pd.Timestamp.now()
    cents[f'closeness_{key}_bridge'] = G.closeness(mode='out', weights='length')
    print(f'Found Closeness Centrality in {pd.Timestamp.now() - start}')

    # betweeness centrality
    start = pd.Timestamp.now()
    n = G.vcount()
    cents[f'betweenness_{key}_bridge'] = G.betweenness(weights='length')
    cents[f'betweenness_{key}_bridge'] = 1/((n-1)*(n-2)) * cents[f'betweenness_{key}_bridge'] # igraph uses denormalized values
    print(f'Found Betweenness Centrality in {pd.Timestamp.now() - start}')

    # straightness centrality
    start = pd.Timestamp.now()
    cents[f'straightness_{key}_bridge'] = straightness(G)
    print(f'Found Straightness Centrality in {pd.Timestamp.now() - start}')

    print(f'Centralities Found in {pd.Timestamp.now() - start_cents}')

    return_dict[removed] = cents


def get_changed_partners(i: int,
                         G_w_bridge, # roads graph with bridge
                         G_wo_bridge # roads graph without bridge
                         ) -> pd.DataFrame:
    dists_w_bridge = np.array(G_w_bridge.distances(i, weights='length')[0])
    dists_wo_bridge = np.array(G_wo_bridge.distances(i, weights='length')[0])
    filter = np.where(dists_w_bridge != dists_wo_bridge)[0] # edge lengths that changed

    if len(filter) > 0:
        index = pd.MultiIndex.from_tuples(zip([i]*len(filter), filter), names=['u', 'v'])
        changed_partners = pd.DataFrame(data={
            'w_bridge': dists_w_bridge[filter],
            'wo_bridge': dists_wo_bridge[filter],
            },
            index=index
        )
    else:
        changed_partners = pd.DataFrame({
                'w_bridge': [np.nan],
                'wo_bridge': [np.nan],
            },
            index=pd.MultiIndex.from_tuples(tuples=[(None, None)], names=['u', 'v'])
        ).dropna() # empty df

    return changed_partners


def get_changed_pairs(G_w_bridge: ig.Graph, # roads graph with bridge
                      G_wo_bridge: ig.Graph # roads graph without bridge
                      ) -> pd.DataFrame:
    assert G_w_bridge.vcount() == G_wo_bridge.vcount() # make sure they have the same verticies

    print(f'Started Getting Changed Pairs at {pd.Timestamp.now()}')

    start = pd.Timestamp.now()
    changed_pairs = [] # keep track of changed paris
    for i in range(G_w_bridge.vcount()):
        changed_partners = get_changed_partners(i, G_w_bridge, G_wo_bridge) # get partners

        if len(changed_partners > 0): # if any paris exist
            changed_pairs.append(changed_partners)
    changed_pairs = pd.concat( # combine
        changed_pairs
    )
    print(f'Found Changed Pairs in {pd.Timestamp.now() - start}')

    changed_pairs.to_csv('data/network_stats/changed_pairs.csv')


def get_avg_spath_len(G_w_bridge: ig.Graph, # roads graph with bridge
                      G_wo_bridge: ig.Graph # roads graph without bridge
                      ) -> pd.DataFrame:
    print(f'Started Getting Average Shortest Path at {pd.Timestamp.now()}')

    # with bridge
    start = pd.Timestamp.now()
    avg_spath_w_bridge = G_w_bridge.average_path_length(weights='length')
    print(f'Found Average Path Length With Bridge in {pd.Timestamp.now() - start}')

    # without bridge
    start = pd.Timestamp.now()
    avg_spath_wo_bridge = G_wo_bridge.average_path_length(weights='length')
    print(f'Found Average Path Length Without Bridge in {pd.Timestamp.now() - start}')

    pd.DataFrame({
        'w_bridge': [avg_spath_w_bridge],
        'wo_bridge': [avg_spath_wo_bridge]
    }).to_csv('data/network_stats/avg_spath.csv', index=False)



def main():
    print(f'Started at {pd.Timestamp.now()}')

    start = pd.Timestamp.now() # keep track of time to load data

    # nodes
    intersections_df = pd.read_csv('data/network/BMA_intersections.csv')
    intersections_df['geometry'] = intersections_df['geometry'].apply(wkt.loads) # convert to geometry
    intersections_df['osmid'] = intersections_df['osmid'].astype(str)

    # edges
    roads_df = pd.read_csv('data/network/BMA_roads.csv')
    roads_df['u'], roads_df['v'] = roads_df['u'].astype(str), roads_df['v'].astype(str)

    # network
    roads_G = ig.read('data/network/BMA_road_network.gml', format='gml')
    calc_weight(roads_G) # add weights for eigenvector centrality
    add_geometry(roads_G, intersections_df)

    # bridge collapse
    roads_wo_bridge_G = roads_G.copy()
    destroyed_edges = roads_df[roads_df['highway'] == 'destroyed']
    destroyed_edges = list(zip(destroyed_edges['u'].map(roads_G.vs.get_attribute_values('label').index),
                            destroyed_edges['v'].map(roads_G.vs.get_attribute_values('label').index))) # start and end nodes
    print(f'Destroying {len(destroyed_edges)} edges...')
    roads_wo_bridge_G.delete_edges(destroyed_edges) # python multiprocessing doesn't share state

    # check edges were removed
    assert roads_G.ecount() - roads_wo_bridge_G.ecount() == len(destroyed_edges)

    # use multithreading
    manager = Manager()
    cent_dict = manager.dict() # store results

    # get centralities
    dent_w_bridge_p = Process(target=get_centralities, args=(roads_G, False, cent_dict))
    dent_wo_bridge_p = Process(target=get_centralities, args=(roads_wo_bridge_G, True, cent_dict))
    changed_pairs_p = Process(target=get_changed_pairs, args=(roads_G, roads_wo_bridge_G))
    avg_spath_len_p = Process(target=get_avg_spath_len, args=(roads_G, roads_wo_bridge_G))

    dent_w_bridge_p.start() # start processes
    dent_wo_bridge_p.start()
    changed_pairs_p.start()
    avg_spath_len_p.start()

    dent_w_bridge_p.join() # wait for processes to finish
    dent_wo_bridge_p.join()

    # save results
    node_stats = intersections_df.rename(columns={
        'osmid': 'node'
    }).merge(
        right=cent_dict[False],
        on='node',
        how='outer'
    ).merge(
        right=cent_dict[True],
        on='node',
        how='outer'
    )
    node_stats.to_csv('data/network_stats/node_stats.csv', index=False)

    changed_pairs_p.join() # wait for other processes to end
    avg_spath_len_p.join()

    print(f'Time to get Data: {pd.Timestamp.now() - start}')


if __name__ == '__main__':
    main()
