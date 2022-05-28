from math import floor, inf
from typing import Dict, List
from haversine import haversine
from dataclasses import dataclass
import numpy as np
import json
import time

Point = Dict[str, str]

LAT_PER_KILO = (1/111.19508023353306)
LON_PER_KILO = (1/88.14065301693692)

@dataclass
class GridCell:
    start_x: float
    end_x: float
    row_idx: int
    col_idx: int
    points: List[Point]

    def filter_points(self, pivot_x: float, pivot_y: float, dist: float) -> List[Point]:
        result = []
        for p in self.points:
            if haversine((pivot_x, pivot_y), (p["x"], p["y"])) <= dist:
                result.append(p)
        return result

@dataclass
class GridRow:
    start_y: float
    end_y: float
    row_idx: int
    cells: List[GridCell]

class RangeSelector:
    dist: float
    grid: List[GridRow]
    row_count: int
    col_count: int
    datatypes: List[str]
    points_per_kind: Dict[str, List]

    def __init__(self, dist: float, json_path: str) -> None:
        self.dist = dist
        
        proc_py: List[Point] = None
        with open(json_path, "r") as proc_json:
            proc_py = json.load(proc_json)
        
        min_x = 999; max_x = 0
        min_y = 999; max_y = 0
        type_set = set()
        self.points_per_kind = {}
        for record in proc_py:
            min_x = min(min_x, record["x"])
            min_y = min(min_y, record["y"])
            max_x = max(max_x, record["x"])
            max_y = max(max_y, record["y"])
            type_set.add(record["datatype"])
            if record["datatype"] in self.points_per_kind:
                self.points_per_kind[record["datatype"]].append(record)
            else:
                self.points_per_kind[record["datatype"]] = [record]
        
        self.datatypes = list(type_set)

        grid_width = haversine((min_x, max_y), (max_x, max_y))
        grid_height = haversine((min_x, max_y), (min_x, min_y))
        self.row_count = floor(grid_height / dist)
        self.col_count = floor(grid_width / dist)
        cell_width = grid_width / self.col_count
        cell_height = grid_height / self.row_count

        x_points = np.linspace(min_x, max_x, self.col_count)
        y_points = np.linspace(min_y, max_y, self.row_count)
        self.grid = [GridRow(
            start_y = y,
            end_y = y+cell_height,
            row_idx = row_idx,
            cells = [GridCell(
                start_x = x,
                end_x = x+cell_width,
                row_idx = row_idx,
                col_idx = col_idx,
                points = []
            ) for col_idx, x in enumerate(x_points)]
        ) for row_idx, y in enumerate(y_points)]

        for record in proc_py:
            self.__find_cell(record["x"], record["y"]).points.append(record)

    def __find_cell(self, x: float, y: float) -> GridCell:
        left = 0; right = len(self.grid)-1
        while left < right:
            mid = (left + right) // 2
            if y < self.grid[mid].end_y:
                right = mid
            else:
                left = mid+1
        
        row_idx = left
        left = 0; right = len(self.grid[row_idx].cells)-1
        while left < right:
            mid = (left + right) // 2
            if x < self.grid[row_idx].cells[mid].end_x:
                right = mid
            else:
                left = mid+1
        
        col_idx = left
        return self.grid[row_idx].cells[col_idx]

    def __is_valid(self, r0: int, c0: int) -> bool:
        return 0 <= r0 < self.row_count and 0 <= c0 < self.col_count
    
    def __filter_if_exist(self, r: int, c: int, x: float, y: float) -> List[Point]:
        if self.__is_valid(r, c):
            return self.grid[r].cells[c].filter_points(x, y, self.dist)
        return []

    def __get_empty_counter(self, init: int = 0) -> Dict[str, int]:
        count = {}
        for datatype in self.datatypes:
            count[datatype] = init
        return count

    def count_neibors(self, x: float, y: float) -> Dict[str, int]:
        filtered: List[Point] = []
        center = self.__find_cell(x, y)
        r, c = center.row_idx, center.col_idx
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                filtered += self.__filter_if_exist(r+dr, c+dc, x, y)
        
        count: Dict[str, int] = self.__get_empty_counter()
        for p in filtered:
            count[p["datatype"]] += 1
        
        return count
    
    def count_neibors_naively(self, x: float, y: float) -> Dict[str, int]:
        count: Dict[str, int] = self.__get_empty_counter()
        for row in self.grid:
            for cell in row.cells:
                bucket = cell.filter_points(x, y, self.dist)
                for p in bucket:
                    count[p["datatype"]] += 1
        return count
    
    def find_shortest(self, x: float, y: float) -> Dict[str, float]:
        filtered: List[Point] = []
        center = self.__find_cell(x, y)
        r, c = center.row_idx, center.col_idx
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                filtered += self.__filter_if_exist(r+dr, c+dc, x, y)
        
        dist: Dict[str, int] = self.__get_empty_counter(inf)
        for p in filtered:
            dist[p["datatype"]] = min(dist[p["datatype"]], haversine((x, y), (p["x"], p["y"])))
        
        for datatype in dist:
            if dist[datatype] == inf:
                for p in self.points_per_kind[datatype]:
                    dist[datatype] = min(dist[datatype], haversine((x, y), (p["x"], p["y"])))
        
        return dist

    def get_data_types(self) -> List[str]:
        return self.datatypes

if __name__ == "__main__":
    start = time.time()
    selector = RangeSelector(1)
    end = time.time()
    print(f"Setting up: {end-start} sec\n")

    start = time.time()
    print(selector.count_neibors_naively(37.5642135, 127.0016985))
    end = time.time()
    print(f"Naive: {end-start} sec\n")

    start = time.time()
    print(selector.count_neibors(37.5642135, 127.0016985))
    end = time.time()
    print(f"With grid: {end-start} sec\n")