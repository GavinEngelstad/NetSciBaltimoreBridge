### Analysis
The files in this section do the following:
- `get_network.py`: Builds the network and saves it as a CSV (to store geometry) and GML (To do analysis). Requires the OSMnx (Open Street Maps Networkx) package to use an API to get the network
- `network_stats.py`: Calculate centrality measures and shortest paths and saves them in the "network_stats" folder. This file takes a long time hours to run, so I'd recommend against it
- `weight.py`: Dictionary of road type: weight for the eigenvector centrality calculation
