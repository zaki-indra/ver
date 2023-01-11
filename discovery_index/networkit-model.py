import random
import sys
import time
from typing import Union, Optional

import matplotlib.pyplot as plt
import networkit as nk
import numpy as np
import pandas as pd
from tqdm import tqdm
from yaspin import yaspin

graph = nk.Graph()

print(graph.numberOfNodes())


class DiscoveryGraph:

    def __init__(self, debug: bool=False) -> None:
        self.graph = nk.Graph()

        v_id = self.graph.attachNodeAttribute("vertex_id", str)
        minhash = self.graph.attachNodeAttribute("minhash", float)

    def add_vertex(self):
        # TODO: add properties
        return self.graph.addNode()

    def add_edge(self, from_vertex: int, to_vertex: int, add_missing: 
        bool=False, check_multi_edge: bool=False):
        '''
        Add edge from `from_vertex` to `to_vertex`
        '''
        # TODO: add properties
        return self.graph.addEdge(from_vertex, to_vertex, addMissing=add_missing)

    def find_shortest_path(self, source, target, return_path: bool=False, return_finder: bool=True):
        pathfinder = nk.distance.BidirectionalBFS(self.graph, source, target)
        pathfinder.run()
        is_path_exist = not (pathfinder.getDistance() == sys.float_info.max)
        if not return_path:
            temp_path = pathfinder.getPath()
            path = np.array([source] + temp_path + [target])

        if return_finder:
            return is_path_exist, path, pathfinder

        return is_path_exist, path, None

    def find_neighborhood(self, source: int, hop: int, return_finder: bool=True):
        neighbor_finder = nk.distance.BFS(self.graph, source)
        neighbor_finder.run()
        distance_array = np.array(neighbor_finder.getDistances(True)).astype(np.int16)
        if return_finder:
            return distance_array < hop, neighbor_finder
        else:
            return distance_array < hop, None


def test_graph(num_vertices, sparsity):
    dg = DiscoveryGraph(debug=True)

    print(f"Generating {num_vertices}^2 random edges with sparsity of {sparsity}")
    start = time.time()
    for i in tqdm(range(num_vertices)):
        for j in range(i + 1, num_vertices):
            if random.random() < sparsity:
                dg.add_edge(i, j, add_missing=True)
    print(f"Time elapsed {time.time() - start} s")
    print("=============================================")

    print(
        f"Finding shortest between vertex 0 and 2\n"\
        f"Number of vertices: {num_vertices}\n"\
        f"Sparsity: {sparsity}"
    )

    with yaspin(text="Find shortest path...", color="white") as sp:
        path_start = time.time()
        paths = dg.find_shortest_path(0, 2)
        path_end = time.time()
        sp.write("DONE!")

    print(f"Time elapsed {path_end-path_start} s")
    print("=============================================")

    print(
        f"Finding 2-hop neighborhood of vertex 2\n"\
        f"Number of vertices: {num_vertices}\n"\
        f"Sparsity: {sparsity}"
    )

    with yaspin(text="Finding neighborhood...", color="white") as sp:
        nbhd_start = time.time()
        neighbors = dg.find_neighborhood(2, 2)
        nbhd_end = time.time()
        sp.write("DONE!")

    print(f"Time elapsed {nbhd_end-nbhd_start} s")
    print("=============================================")

    return path_end-path_start, nbhd_end-nbhd_start

def test_scalability():
    '''
    Testing the scalability of finding the 2-hop neighborhood of a node as well
    as finding paths between 2-nodes
    '''
    nodes = [100, 200, 400, 800, 1600, 3000, 5500, 8000, 10000, 12000, 14000]
    sparsity = [0.1, 0.2]
    nbhd_result = []
    path_result = []
    titles = ['2-hop Neighborhood Search', 'Path Finding']

    for n in nodes:
        for s in sparsity:
            path_time, nbhd_time = test_graph(n, s)
            nbhd_result.append([n, s, nbhd_time])
            path_result.append([n, s, path_time])

    with yaspin(text="Generating plot...") as sp:
        for result, title in zip([nbhd_result, path_result], titles):
            df = pd.DataFrame(result, columns = ['No. of Nodes', 'Sparsity', 'Time'])
            df = df.pivot(index='No. of Nodes', columns='Sparsity', values='Time')
            plt.figure()
            df.plot(title=f'{title} Scalability')
            plt.xticks(nodes, rotation=90)
            plt.xlabel('No. of Nodes')
            plt.ylabel('Time')
            plt.savefig(f'{title}.png')
        sp.write("DONE!")

if __name__ == "__main__":
    test_scalability()
        