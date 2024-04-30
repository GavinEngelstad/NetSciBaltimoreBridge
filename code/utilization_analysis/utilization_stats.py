'''
Get use-adjusted centrality stats

Network-Wide: Average Shortest Path
Center of Tract Intersections: Closeness, Straightness
All Intersections: Betweenness

Stores everything in csvs for reference later
'''

from multiprocessing import Process
from shapely import wkt, distance
import networkx as nx
import pandas as pd
import numpy as np
import os

FEET_PER_METER = 3.280839895 # conversion rate


def get_all_dists(jtw: pd.DataFrame,
                  tracts: pd.DataFrame,
                  G: nx.Graph
                  ) -> pd.DataFrame:
    jtw = jtw.copy()
    source, target = jtw.index.names
    jtw = jtw.reset_index().merge( # add gerometry
        tracts[['intersection', 'intersection_geom']],
        left_on=source,
        right_on='intersection'
    ).drop(columns='intersection').rename(columns={
        'intersection_geom': source+'_geom'
    }).merge(
        tracts[['intersection', 'intersection_geom']],
        left_on=target,
        right_on='intersection'
    ).drop(columns='intersection').rename(columns={
        'intersection_geom': target+'_geom'
    }).set_index([source, target])

    jtw['net_dist'] = np.nan # network distance 
    jtw['eucl_dist'] = distance(jtw[source+'_geom'], jtw[target+'_geom']) / FEET_PER_METER # euclidian distance, convert to meters
    jtw['spaths'] = None # paths

    for s in jtw.index.levels[0]: # loop over sources
        net_dists, spaths = nx.single_source_dijkstra(G, s, weight='length')
        # this will only find one path for each but is much faster
        # in my testing (>1000 pairs) there is only ever 1 shortest weighted path

        for t in jtw.loc[s].index: # each path that s goes to
            jtw.at[(s, t), 'net_dist'] = net_dists[t]
            jtw.at[(s, t), 'spaths'] = spaths[t]
    
    return jtw


def avg_shortest_path(df: pd.DataFrame
                      ) -> float:
    df = df[df.index.get_level_values(0) != df.index.get_level_values(1)].copy() # exlude self-paths (neq in equation)

    total_commutes = df['flow'].sum()
    total_dist = (df['flow']*df['net_dist']).sum()

    return total_dist / total_commutes


def closeness_centrality(df: pd.DataFrame
                         ) -> pd.DataFrame:
    df = df[df.index.get_level_values(0) != df.index.get_level_values(1)].copy() # exlude self-paths (neq in equation)

    df['dist*n'] = df['flow'] * df['net_dist'] # denominator

    grouped = df[['flow', 'dist*n']].groupby( # add flow and flow*dist (divdide in equation)
        level=0
    ).sum() 
    grouped['closeness'] = grouped['flow'] / grouped['dist*n'] # calc centrality

    return grouped[['closeness']].reset_index().rename(columns={'home_node': 'node'})


def straightness_centrality(df: pd.DataFrame
                            ) -> pd.DataFrame:
    df = df[df.index.get_level_values(0) != df.index.get_level_values(1)].copy() # exlude self-paths (neq in equation)

    df['bilateral_straightness'] = df['flow'] * df['eucl_dist'] / df['net_dist'] # sum this in equation

    grouped = df[['flow', 'bilateral_straightness']].groupby(
        level=0
    ).sum()
    grouped['straightness'] = grouped['bilateral_straightness'] / grouped['flow']

    return grouped[['straightness']].reset_index().rename(columns={'home_node': 'node'})


def betweenness_centrality(df: pd.DataFrame
                           ) -> pd.DataFrame:
    df = df[df.index.get_level_values(0) != df.index.get_level_values(1)].copy() # exlude self-paths (neq in equation)
    source, target = df.index.names

    total_commuters = df['flow'].sum() # get total

    # paths through node
    get_cent = df['spaths'].str[1:-1].repeat( # exlcude start and end and repeat based on flow (multiply by n_ij)
        df['flow']
    ).explode( # split it all up
    ).value_counts( # count occurences
    ).reset_index( # keep i
    ).rename(columns={ # rename to generic label
        'spaths': 'node'
    })

    get_cent = get_cent.merge( # count paths that start at node
        right=df['flow'].groupby(
            level=0
        ).sum().reset_index().rename(columns={ # rename to generic label
            source: 'node',
            'flow': 'source_flow'
        }),
        on='node',
        how='left'
    ).replace(np.nan, 0)
    get_cent = get_cent.merge( # count paths that end at node
        right=df['flow'].groupby(
            level=1
        ).sum().reset_index().rename(columns={ # rename to generic label
            target: 'node',
            'flow': 'target_flow'
        }),
        on='node',
        how='left'
    ).replace(np.nan, 0)
    get_cent['num_paths'] = total_commuters - get_cent['source_flow'] - get_cent['target_flow'] # equation denominator

    # calculate centrality
    get_cent['betweenness'] = get_cent['count'] / get_cent['num_paths']

    return get_cent[['node', 'betweenness']]


def get_stats(jtw: pd.DataFrame,
              tracts: pd.DataFrame,
              roads_G: nx.Graph,
              removed: bool
              ) -> None:
    print('Getting Stats...')

    # Get all paths/distances
    start = pd.Timestamp.now() # keep track of time
    dists = get_all_dists(jtw, tracts, roads_G)
    end = pd.Timestamp.now()
    print(f'Time to Get Distances: f{end-start}')

    # network stats
    start = pd.Timestamp.now() # keep track of time
    avg_spath_len = avg_shortest_path(dists)
    end = pd.Timestamp.now()
    print(f'Time to Get Average Path Length: {end-start}')

    # get tract stats
    start = pd.Timestamp.now() # keep track of time
    tract_cent = closeness_centrality(dists).merge(
        right=straightness_centrality(dists),
        on='node'
    )
    end = pd.Timestamp.now()
    print(f'Time to Get Tract Stats: {end-start}')

    # get intersection stats
    start = pd.Timestamp.now() # keep track of time
    betcent = betweenness_centrality(dists)
    end = pd.Timestamp.now()
    print(f'Time to Get Node Stats: {end-start}')

    key = 'wo' if removed else 'w'
    pd.DataFrame([avg_spath_len], columns=['avg_spath']).to_csv(f'data/utilization_stats/avg_spath_{key}_bridge.csv', index=False) # save average path
    tract_cent.to_csv(f'data/utilization_stats/tract_stats_{key}_bridge.csv', index=False)
    betcent.to_csv(f'data/utilization_stats/node_stats_{key}_bridge.csv', index=False)


def merge_stats(intersections, tracts):
    avg_spath_w_bridge = pd.read_csv('data/utilization_stats/avg_spath_w_bridge.csv')
    avg_spath_wo_bridge = pd.read_csv('data/utilization_stats/avg_spath_wo_bridge.csv')
    tract_stats_w_bridge = pd.read_csv('data/utilization_stats/tract_stats_w_bridge.csv')
    tract_stats_wo_bridge = pd.read_csv('data/utilization_stats/tract_stats_wo_bridge.csv')
    node_stats_w_bridge = pd.read_csv('data/utilization_stats/node_stats_w_bridge.csv')
    node_stats_wo_bridge = pd.read_csv('data/utilization_stats/node_stats_wo_bridge.csv')
    tract_stats_w_bridge['node'], tract_stats_wo_bridge['node'] = tract_stats_w_bridge['node'].astype(str), tract_stats_wo_bridge['node'].astype(str)
    node_stats_w_bridge['node'], node_stats_wo_bridge['node'] = node_stats_w_bridge['node'].astype(str), node_stats_wo_bridge['node'].astype(str)

    # networkwide averages
    avg_spath = avg_spath_w_bridge.rename(columns={
        'avg_spath': 'avg_spath_w_bridge'
    }).merge(
        right=avg_spath_wo_bridge.rename(columns={
            'avg_spath': 'avg_spath_wo_bridge'
        }),
        right_index=True,
        left_index=True
    )

    # tract averages
    tract_stats = tracts.rename(columns={
        'intersection': 'node'
    }).merge(
        right=tract_stats_w_bridge.rename(columns={
            'closeness': 'closeness_w_bridge',
            'straightness': 'straightness_w_bridge'
        }),
        on='node',
        how='outer'
    ).merge(
        right=tract_stats_wo_bridge.rename(columns={
            'closeness': 'closeness_wo_bridge',
            'straightness': 'straightness_wo_bridge'
        }),
        on='node',
        how='outer'
    ).replace(np.nan, 0)

    # node averages
    node_stats = intersections.rename(columns={
        'osmid': 'node'
    }).merge(
        right=node_stats_w_bridge.rename(columns={
            'betweenness': 'betweenness_w_bridge'
        }),
        on='node',
        how='outer'
    ).merge(
        right=node_stats_wo_bridge.rename(columns={
            'betweenness': 'betweenness_wo_bridge'
        }),
        on='node',
        how='outer'
    ).replace(np.nan, 0)

    avg_spath.to_csv(f'data/utilization_stats/avg_spath.csv', index=False) # save average path
    tract_stats.to_csv(f'data/utilization_stats/tract_stats.csv', index=False)
    node_stats.to_csv(f'data/utilization_stats/node_stats.csv', index=False)

    os.remove('data/utilization_stats/avg_spath_w_bridge.csv') # remove unneeded files
    os.remove('data/utilization_stats/avg_spath_wo_bridge.csv')
    os.remove('data/utilization_stats/tract_stats_w_bridge.csv')
    os.remove('data/utilization_stats/tract_stats_wo_bridge.csv')
    os.remove('data/utilization_stats/node_stats_w_bridge.csv')
    os.remove('data/utilization_stats/node_stats_wo_bridge.csv')


def main():
    print(f'Started at {pd.Timestamp.now()}')

    source = 'home_node'
    target = 'work_node'

    # Get Files
    roads_G = nx.read_gml('data/network/BMA_road_network.gml')

    jtw = pd.read_csv('data/utilization_stats/jtw.csv')
    jtw['home'], jtw['work'] = jtw['home'].astype(str), jtw['work'].astype(str) # mergable format
    jtw['home_node'], jtw['work_node'] = jtw['home_node'].astype(str), jtw['work_node'].astype(str)
    jtw = jtw.set_index([source, target])

    intersections = pd.read_csv('data/network/BMA_intersections.csv')
    intersections['geometry'] = intersections['geometry'].apply(wkt.loads) # geometry object
    intersections['osmid'] = intersections['osmid'].astype(str) # mergable format

    tracts = pd.read_csv('data/utilization_stats/tracts.csv')
    tracts['intersection_geom'] = tracts['intersection_geom'].apply(wkt.loads) # geometry object
    tracts['intersection'] = tracts['intersection'].astype(str) # mergable format

    # bridge collapse
    roads_df = pd.read_csv('data/network/BMA_roads.csv')
    roads_df['u'], roads_df['v'] = roads_df['u'].astype(str), roads_df['v'].astype(str)
    roads_wo_bridge_G = roads_G.copy()
    destroyed_edges = roads_df[roads_df['highway'] == 'destroyed']
    destroyed_edges = list(zip(destroyed_edges['u'], destroyed_edges['v'])) # start and end nodes
    print(f'Destroying {len(destroyed_edges)} edges...')
    roads_wo_bridge_G.remove_edges_from(destroyed_edges) # python multiprocessing doesn't share state

    # check edges were removed
    assert len(roads_G.edges) - len(roads_wo_bridge_G.edges) == len(destroyed_edges)

    # Get all paths/distances
    w_bridge_p = Process(target=(get_stats), args=(jtw, tracts, roads_G, False))
    wo_bridge_p = Process(target=(get_stats), args=(jtw, tracts, roads_wo_bridge_G, True))

    w_bridge_p.start()
    wo_bridge_p.start()

    w_bridge_p.join()
    wo_bridge_p.join()

    merge_stats(intersections, tracts)

    print('Done')


if __name__ == '__main__':
    main()
