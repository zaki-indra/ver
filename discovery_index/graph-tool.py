import time
import timeit
from typing import Union
import random
from argparse import ArgumentParser

from graph_tool import Graph, topology, search
from graph_tool.all import graph_draw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


class VertexProperty:
    #TODO: not implemented yet
    def __init__(self) -> None:
        pass


class NeighborhoodVisitor(search.BFSVisitor):
    def __init__(self, hops:int, name, pred, hop, param: Union[dict, None]=None, debug=False) -> None:
        self.hops = hops
        self.name = name
        self.pred = pred
        self.hop = hop
        self.param = param
        self.debug = debug

    def discover_vertex(self, u):
        if self.debug:
            print("-->", self.name[int(u)], "has been discovered!")
        return super().discover_vertex(u)

    def examine_vertex(self, u):
        if self.debug:
            print(self.name[int(u)], "has been examined")
        return super().examine_vertex(u)

    def tree_edge(self, e):
        src_index = int(e.source())
        tgt_index = int(e.target())

        # Don't let bfs examine all vertices
        if self.hop[src_index] == self.hops:
            raise search.StopSearch()

        self.pred[tgt_index] = src_index
        self.hop[tgt_index] = self.hop[src_index] + 1
        return super().tree_edge(e)

    
class DiscoveryGraph:
    def __init__(self) -> None:
        self.graph = Graph(directed=False)

    def add_node(self, column: dict):
        '''
        Adds a single node, representing a column, into the nodes graph
        '''

        vertex = self.graph.add_vertex()
        #TODO: Add vertex properties

        return vertex

    def add_node_test(self, num_vertices:int=1) -> None:
        '''
        Only add vertex with no properties attached
        '''

        vertices = self.graph.add_vertex(num_vertices)

    def add_edge(self, from_vertex, to_vertex):
        '''
        Add edge from `from_vertex` to `to_vertex`
        '''
        #TODO: Add edge properties
        graph = self.graph
        edge = graph.add_edge(graph.vertex(from_vertex), graph.vertex(to_vertex))
        return edge

    def find_neighborhood(self, vertex, hops):
        num_vertices = self.graph.num_vertices(True)

        #TODO: add more relevant properties
        name = np.arange(num_vertices)
        hop = np.empty(num_vertices, np.int32)
        pred = np.empty(num_vertices, np.int32)

        # Initialize hop and pred value to -1
        hop.fill(-1); pred.fill(-1)

        # Init 0-hop for first vertex
        hop[int(vertex)] = 0

        # Search neighborhood
        search.bfs_search(self.graph, self.graph.vertex(vertex), NeighborhoodVisitor(hops, name, pred, hop))
        
        return np.where((hop <= hops) & (hop >= 0))

    #TODO: Implement neighbor search recursively using vertex .all_neighbors() method
    def find_neighborhood2(self, vertex, hops):
        vertex = self.graph.vertex(vertex)
        if hops < 0:
            raise Exception()
        if hops == 0:
            return vertex
        
        vertices = [vertex]
        hop = 1
        vertices.append(vertex.all_neighbors())


    def find_all_paths(self, from_vertex, to_vertex):
        return topology.all_paths(self.graph, self.graph.vertex(from_vertex), self.graph.vertex(to_vertex))
        

def test_graph(num_vertex, sparsity):
    '''
    Generate a random graph with given number of nodes and sparsity
    '''
    dg = DiscoveryGraph()
    graph = dg.graph

    dg.add_node_test(num_vertex)

    # Generate a random set of edges
    print(f"Generating {num_vertex}^2 random edges with sparsity of {sparsity}")
    start = time.time()
    for i in tqdm(range(num_vertex)):
        for j in range(i + 1, num_vertex):
            if random.random() < sparsity:
                dg.add_edge(i, j)
    print(f"Time elapsed {time.time() - start} s")
    print("=============================================")

    # Make image of the graph if there are less than 30 vertex
    if num_vertex < 30:
        graph_draw(graph, vertex_text=graph.vertex_index, output="graph.png")

    print(
        f"Finding all paths between vertex 0 and 2...\n"\
        f"Number of vertices: {num_vertex}\n"\
        f"Sparsity: {sparsity}"
    )
    path_start = time.time()
    paths = dg.find_all_paths(0, 2)
    path_end = time.time()

    print("DONE!")
    print(f"Time elapsed {path_end-path_start} s")
    print("=============================================")

    print(
        f"Finding 2-hop neighborhood of vertex 2\n"\
        f"Number of vertices: {num_vertex}\n"\
        f"Sparsity: {sparsity}"
    )
    nbhd_start = time.time()
    neighbors = dg.find_neighborhood(2, 2)
    nbhd_end = time.time()
    
    print("DONE!")
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

    for result, title in zip([nbhd_result, path_result], titles):
        df = pd.DataFrame(result, columns = ['No. of Nodes', 'Sparsity', 'Time'])
        df = df.pivot(index='No. of Nodes', columns='Sparsity', values='Time')
        plt.figure()
        df.plot(title=f'{title} Scalability')
        plt.xticks(nodes, rotation=90)
        plt.xlabel('No. of Nodes')
        plt.ylabel('Time')
        plt.savefig(f'{title}.png')

def main():
    test_scalability()

if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    main()

