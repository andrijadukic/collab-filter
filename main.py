from typing import List

import numpy as np

from collab_filtering import CollaborativeFiltering


def main():
    data = load_data_from_stdin()
    collab_filter = CollaborativeFiltering()
    collab_filter.fit(data)

    queries = load_queries_from_stdin()
    for query in queries:
        print(collab_filter.predict(target=(query[0] - 1, query[1] - 1), query_type=query[2], k=query[3]))


def load_data_from_stdin() -> np.ndarray:
    dimensions = tuple(int(dim) for dim in input().split(' '))
    data_raw = [input() for _ in range(dimensions[0])]
    data_processed = np.genfromtxt(data_raw, dtype=float, delimiter=' ', missing_values='X', filling_values=np.nan)
    if dimensions != data_processed.shape:
        raise ValueError(f"Dimension mismatch: ({dimensions} != {data_processed.shape})")
    return data_processed


def load_queries_from_stdin() -> List[List]:
    return [[int(arg) for arg in input().split(' ')] for _ in range(int(input()))]


if __name__ == '__main__':
    main()
