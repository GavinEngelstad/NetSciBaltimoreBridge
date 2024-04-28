'''
Get utilization stats and convert it to a format thats
compatible with the network

Store everything in dataframes for reference later
'''

from shapely import wkt, distance, centroid
import geopandas as gpd
import pandas as pd
import numpy as np


def get_closest_intersection(tracts: gpd.GeoDataFrame
                             ) -> None:
    # get intersections
    intersections = pd.read_csv('data/network/BMA_intersections.csv')
    intersections['geometry'] = intersections['geometry'].apply(wkt.loads)
    intersections['osmid'] = intersections['osmid'].astype(str)
    intersections = gpd.GeoDataFrame(intersections, geometry='geometry')

    for i in tracts.index:
        dists = distance(centroid(tracts.loc[i, 'geometry']), intersections['geometry'])
        closest_intersection = intersections.index[np.argmin(dists)]
        tracts.loc[i, 'intersection'] = intersections.loc[closest_intersection, 'osmid']
        tracts.loc[i, 'intersection_geom'] = intersections.loc[closest_intersection, 'geometry']


def main():
    ## Journey to work data
    jtw_from_bma = pd.read_csv('data/utilization/2012-2016_CTPP_worker_flows_tract_to_tract_(from_BMC_area).csv')
    jtw_from_bma['RES_FIPS'], jtw_from_bma['WP_FIPS'] = jtw_from_bma['RES_FIPS'].astype(str), jtw_from_bma['WP_FIPS'].astype(str)

    jtw_to_bma = pd.read_csv('data/utilization/2012-2016_CTPP_worker_flows_tract_to_tract_(to_BMC_area).csv')
    jtw_to_bma['RES_FIPS'], jtw_to_bma['WP_FIPS'] = jtw_to_bma['RES_FIPS'].astype(str), jtw_to_bma['WP_FIPS'].astype(str)

    # combine both
    jtw = jtw_from_bma[['RES_FIPS', 'WP_FIPS', 'EST']].rename(columns={
        'EST': 'EST_from'
    }).merge(
        right=jtw_to_bma[['RES_FIPS', 'WP_FIPS', 'EST']].rename(columns={
            'EST': 'EST_to'
        }),
        on=['RES_FIPS', 'WP_FIPS'],
        how='outer'
    )

    ## ratios excluded
    exc_from_bma = jtw.loc[jtw['EST_to'].isna(), 'EST_from'].sum() / jtw['EST_from'].sum()
    print(f'Percent Excluded Who Travel from BMA: {exc_from_bma}')
    exc_to_bma = jtw.loc[jtw['EST_from'].isna(), 'EST_to'].sum() / jtw['EST_to'].sum()
    print(f'Percent Excluded Who Travel to BMA: {exc_to_bma}')

    # drop out of BMA tracts
    jtw = jtw.dropna()

    # check files were the same
    assert (jtw['EST_from'] == jtw['EST_to']).all()

    jtw = jtw.drop(columns='EST_to').rename(columns={
        'RES_FIPS': 'home',
        'WP_FIPS': 'work',
        'EST_from': 'flow'
    })

    ## Census tract locations (Requires JTW Data)
    us_tracts = gpd.read_file('data/utilization/cb_2019_us_tract_500k2019/cb_2019_us_tract_500k.shp')
    us_tracts = us_tracts[us_tracts['GEOID'].isin(np.concatenate((jtw['home'], jtw['work'])))] # only include tracts in BMA
    us_tracts = us_tracts.to_crs('EPSG:2893') # convert coords
    us_tracts = us_tracts[['GEOID', 'geometry']].rename(columns={
        'GEOID': 'tract'
    })

    # merge with utilization
    us_tracts = us_tracts.merge(
        right=jtw[['home', 'flow']].groupby(
            'home'
        ).sum().reset_index().rename(columns={
            'flow': 'residents'
        }),
        left_on='tract',
        right_on='home',
        how='left'
    ).drop(columns='home').merge(
        right=jtw[['work', 'flow']].groupby(
            'work'
        ).sum().reset_index().rename(columns={
            'flow': 'workers'
        }),
        left_on='tract',
        right_on='work',
        how='left'
    ).drop(columns='work')

    # get intersection to find distances from
    get_closest_intersection(us_tracts)

    jtw = jtw.merge(
        right=us_tracts[['tract', 'intersection']],
        left_on='home',
        right_on='tract'
    ).drop(columns='tract').rename(columns={
        'intersection': 'home_node'
    }).merge(
        right=us_tracts[['tract', 'intersection']],
        left_on='work',
        right_on='tract'
    ).drop(columns='tract').rename(columns={
        'intersection': 'work_node'
    })
    
    # save
    jtw.to_csv('data/utilization_stats/jtw.csv', index=False)
    us_tracts.to_csv('data/utilization_stats/tracts.csv', index=False)


if __name__ == '__main__':
    main()