from argparse import ArgumentParser
from datetime import datetime
from enum import Enum
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
from typing import Union, Optional

from datasketch import MinHash, MinHashLSH
import duckdb
import matplotlib.pyplot as plt
import networkit as nk
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
from yaspin import yaspin


class StoreType(Enum):
    """
    Database store type
    """
    MEMORY="memory"
    LOCAL="local"
    REMOTE="remote"

class DatabaseEngine:
    """
    For storing data in the database
    """
    def __init__(self, database: str="duckdb", store_type: StoreType="local", debug: bool=False) -> None:
        """
        Initiate engine and connection
        """
        self.database = database
        if self.database == "duckdb":
            self.__conn = self._init_duckdb(store_type)

        elif self.database == "sqlite":
            self.__conn = self._init_sqlite(store_type)

    def _init_duckdb(self, store_type: StoreType="local") -> duckdb.DuckDBPyConnection:
        """
        init duckdb connection
        """
        if store_type == StoreType.LOCAL.value:
            return duckdb.connect("duck.db", read_only=False)
        elif store_type == StoreType.MEMORY.value:
            return duckdb.connect(":memory:")

    def _init_sqlite(self, store_type: StoreType="local"):
        """
        init sqlite connection
        """
        if store_type == StoreType.LOCAL.value:
            self.engine = create_engine("sqlite:///sqlite.db")
            return self.engine.connect()

    def execute(self, query: str, *args, **kwargs):
        return self.__conn.execute(query, *args, **kwargs)

    @property
    def connection(self):
        return self.__conn


class DiscoveryGraph:

    def __init__(self, data_dir: str=".", debug: bool=False) -> None:
        self.graph = nk.Graph()
        self.graph.indexEdges()

        self.debug = debug
        self.minhash_perm = None

        self.vertex_id = self.graph.attachNodeAttribute("vertex_id", float)
        self.minhash = self.graph.attachNodeAttribute("minhash", float)

        if debug:
            return

        self.database = DatabaseEngine(store_type="local")
        self.conn = self.database.connection
        self.create_table()

        self.parse_dir(data_dir, file_type="json")

        self.make_similarity_edges()

    def create_table(self):
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE translationtable (
                index   INTEGER NOT NULL UNIQUE,
                id      DECIMAL(18, 0) NOT NULL PRIMARY KEY,
            );
            CREATE OR REPLACE TABLE nodes (
                id             DECIMAL(18, 0) NOT NULL PRIMARY KEY,
                dbname         VARCHAR(255),
                path           VARCHAR(255),
                sourcename     VARCHAR(255),
                columnname     VARCHAR(255),
                datatype       VARCHAR(255),
                totalvalues    INT,
                uniquevalues   INT,
                nonemptyvalues INT,
                entities       VARCHAR(255),
                minhash        INTEGER[],
                minvalue       DECIMAL(18, 4),
                maxvalue       DECIMAL(18, 4),
                avgvalue       DECIMAL(18, 4),
                median         DECIMAL(18, 0),
                iqr            DECIMAL(18, 0)
            );
            CREATE OR REPLACE TABLE edges (
                id        INTEGER,
                from_node DECIMAL(18, 0),
                to_node   DECIMAL(18, 0),
                weight    REAL,
            );
            CREATE OR REPLACE TABLE minhash (
                id      DECIMAL(18, 0) NOT NULL PRIMARY KEY,
                minhash BYTEA,
            );
            """
        )

    def parse_dir(self, data_dir: Optional[Union[Path, str]], file_type="json"):
        '''
        Parse whole directory of JSON file
        '''
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise DirectoryError("The current path is not a directory")

        if file_type == "json":
            self.parse_dir_json(data_dir)

    def parse_dir_json(self, data_dir: Optional[Union[Path, str]]):
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            data = self.parse_json_file(filepath)

            if data["minhash"] is not None and self.minhash_perm is None:
                self.minhash_perm = len(data["minhash"])
            self.add_vertex(data)

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
            raise FileError(f"{file_path} is not a JSON file")

    def add_vertex(self, data: dict):
        vertex = self._add_vertex()
        v_id = data.get("id", None)

        nodes_table = self.conn.table("nodes")
        nodes_table.insert(data.values())

        translation_table = self.conn.table("translationtable")
        translation_table.insert([vertex, v_id])
        
        if data.get("minhash") is not None:            
            minhash = MinHash(num_perm=self.minhash_perm, hashvalues=data["minhash"])
            peckle = pickle.dumps(minhash)
            minhash_table = self.conn.table("minhash")
            minhash_table.insert([v_id, peckle])

        return vertex

    def _add_vertex(self):
        return self.graph.addNode()

    def make_similarity_edges(self, threshold: int=0.5):
        ''''
        Construct the graph (edges) based on minHash signatures of the nodes
        '''
        def helper(a: int, b: int):
            return (a, b) if a < b else (b, a)

        content_index = MinHashLSH(threshold, num_perm=self.minhash_perm)
        start_time = time.time()

        df = self.conn.execute(
        """
        select index, minhash from translationtable as tt, minhash as mh
        where tt.id = mh.id;
        """
        ).df()
        df["minhash"] = df["minhash"].map(lambda x: pickle.loads(x))
        df.apply(lambda row : content_index.insert(row['index'],
                                                   row['minhash']), axis=1)
        spent_time = time.time() - start_time
        print(f'Indexed all minHash signatures: Took {spent_time}')

        vertex_pair = set()
        for _, row in df.iterrows():
            neighbors = content_index.query(row['minhash'])
            for neighbor in neighbors:
                source = row["index"]
                target = neighbor

                pair = helper(int(source), int(target))
                if pair not in vertex_pair:
                    vertex_pair.add(pair)
                    self.add_edge(source, target)

    def add_edge(self, source: int, target: int, **kwargs):
        edge = self._add_edge(source, target)

        edges_table = self.conn.table('edges')
        edges_table.insert([edge, source, target, 1])
        edges_table.insert([edge, target, source, 1])

        return edge

    def _add_edge(self, source: int, target: int, add_missing: 
        bool=False, check_multi_edge: bool=False):
        '''
        Add edge from source to target
        '''
        self.graph.addEdge(source, target, addMissing=add_missing)
        return self.graph.edgeId(source, target)

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
    dg.graph.addNodes(num_vertices)

    print(f"Generating {num_vertices}^2 random edges with sparsity of {sparsity}")
    start = time.time()
    for i in tqdm(range(num_vertices)):
        for j in range(i + 1, num_vertices):
            if random.random() < sparsity:
                dg.graph.addEdge(i, j, addMissing=True)
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


def main(args):
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


class Logger:

    @classmethod
    def WARN(cls, warning: Optional[str]=None):
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[WARNING | {now}] {warning}")

    @classmethod
    def INFO(cls, info: Optional[str]=None):
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[INFO | {now}] {info}")


class BaseException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class DirectoryError(BaseException):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class FileError(BaseException):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
