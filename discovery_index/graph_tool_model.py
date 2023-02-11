import json
import os
import random
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from exception.exception import DirectoryError, FileError
from graph_tool import Graph, search, topology
from graph_tool.all import graph_draw
from tqdm import tqdm
from yaspin import yaspin


class VertexProperty:
    def __init__(self, profile) -> None:
        self.profile = profile


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

        # Don't let bfs visit all vertices
        if self.hop[src_index] == self.hops:
            raise search.StopSearch()

        self.pred[tgt_index] = src_index
        self.hop[tgt_index] = self.hop[src_index] + 1
        return super().tree_edge(e)


class DiscoveryGraph:
    '''
    Discovery Graph class
    '''
    def __init__(self, data_dir: str=".", debug=False) -> None:
        self.graph = Graph(directed=False)

        self.minhash_perm = None

        self.graph.vertex_properties["profile"] = self.graph.new_vertex_property("python::object")
        self.graph.vertex_properties["minhash"] = self.graph.new_vertex_property("python::object")
        self.graph.edge_properties["profile"] = self.graph.new_edge_property("python::object")

        if debug:
            return None

        self.parse_dir(data_dir)
        self.make_similarity_edges()


    def make_similarity_edges(self, threshold: int=0.5):
        ''''
        Construct the graph (edges) based on minHash signatures of the nodes
        '''
        content_index = MinHashLSH(threshold, num_perm=self.minhash_perm)
        
        df = pd.DataFrame(columns=["id", "minhash"])
        df.astype("object")
        id_to_index = {}
        start_time = time.time()

        for vertex in self.graph.iter_vertices():
            vertex_id = self.graph.vertex_properties.profile[vertex].profile["id"]
            minhash = self.graph.vertex_properties.minhash[vertex]

            if minhash is not None:
                id_to_index[vertex_id] = vertex
            
                temp = pd.Series({"id": vertex_id, "minhash": minhash})
                df = pd.concat([df, temp.to_frame().T], ignore_index=True)

        df.apply(lambda row : content_index.insert(row['id'],
                                                   row['minhash']), axis=1)
        spent_time = time.time() - start_time
        print(f'Indexed all minHash signatures: Took {spent_time}')

        vertex_pair = set()
        for _, row in df.iterrows():
            neighbors = content_index.query(row['minhash'])
            for neighbor in neighbors:
                from_vertex = id_to_index[row["id"]]
                to_vertex = id_to_index[neighbor]

                # No more than 1 edge connecting between two vertices
                pair = self.helper(int(from_vertex), int(to_vertex))
                if pair not in vertex_pair:
                    vertex_pair.add(pair)
                    self.add_edge(from_vertex, to_vertex)

    def helper(self, a: int, b: int):
        return (a, b) if a < b else (b, a)

    def parse_dir(self, data_dir: Union[Path, str]):
        '''
        Parse whole directory of JSON file
        '''
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise DirectoryError("The current path is not a directory")

        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            data = self.parse_json_file(filepath)

            vertex_prop = VertexProperty(data)
            if data["minhash"] is not None:
                self.minhash_perm = len(data["minhash"])
            self.add_vertex(vertex_prop)

    def parse_json_file(self, file_path: Union[Path, str]):
        '''
        Parse JSON file
        '''
        if Path(file_path).suffix == ".json":
            with open(file_path, 'r', encoding="utf-8") as file, \
            yaspin(f"Reading {file_path} ...", color="white") as sp:
                data = json.load(file)
                sp.write(f"Finished reading {file_path}")
            return data
        else:
            raise FileError(f"{file_path} is not a valid JSON file")

    def add_vertex(self, profile: VertexProperty):
        '''
        Adds a single vertex with column properties
        '''

        vertex = self.graph.add_vertex()
        self.graph.vertex_properties.profile[vertex] = profile
        if profile.profile["minhash"] is not None:
            self.graph.vertex_properties.minhash[vertex] = MinHash(self.minhash_perm, hashvalues=profile.profile["minhash"])
        return vertex

    def add_node_test(self, num_vertices:int=1) -> None:
        '''
        Only add vertex with no properties attached
        '''

        vertices = self.graph.add_vertex(num_vertices)
        return vertices

    def add_edge(self, from_vertex, to_vertex):
        '''
        Add edge from `from_vertex` to `to_vertex`
        '''

        graph = self.graph
        edge = graph.add_edge(graph.vertex(from_vertex), graph.vertex(to_vertex))
        return edge

    def find_neighborhood(self, vertex, hops):
        num_vertices = self.graph.num_vertices(True)

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

    def find_all_paths(self, from_vertex, to_vertex):
        return topology.all_paths(self.graph, self.graph.vertex(from_vertex), self.graph.vertex(to_vertex))

    def find_shortest_path(self, from_vertex, to_vertex):
        return topology.shortest_path(self.graph, self.graph.vertex(from_vertex), self.graph.vertex(to_vertex))


def test_graph(num_vertex, sparsity):
    '''
    Generate a random graph with given number of nodes and sparsity
    '''
    dg = DiscoveryGraph(debug=True)
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
        f"Finding all paths between vertex 0 and 2\n"\
        f"Number of vertices: {num_vertex}\n"\
        f"Sparsity: {sparsity}"
    )

    with yaspin(text="Finding all paths...", color="white") as sp:
        path_start = time.time()
        paths = dg.find_all_paths(0, 2)
        path_end = time.time()
        sp.write("DONE!")

    print(f"Time elapsed {path_end-path_start} s")
    print("=============================================")

    print(
        f"Finding 2-hop neighborhood of vertex 2\n"\
        f"Number of vertices: {num_vertex}\n"\
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


def main(args):
    print(args.path)
    discovery_graph = DiscoveryGraph(args.path)
    if args.save is not None:
        discovery_graph.graph.save(f"{args.save}.gt", "gt")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog = 'Network Builder',
        description = 'Builds the Entreprise Knowledge Graph')
    parser.add_argument("-p", "--path", help="Directory of JSON profile")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("-s", "--save", help="Save path")
    args = parser.parse_args()
    if args.benchmark:
        test_scalability()
    main(args)