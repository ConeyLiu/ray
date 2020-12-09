from typing import Dict, List, Iterable, Type

import pandas as pd


class DataPiece:
    def __iter__(self) -> Iterable[pd.DataFrame]:
        raise NotImplementedError

    @property
    def num_records(self) -> int:
        """The number of records for this data piece.

        This should be equal with sum of all pandas DataFrame rows.
        """
        raise NotImplementedError


class SourceShard:

    """A interface for source shard data"""
    def prefix(self) -> str:
        raise NotImplementedError

    @property
    def shard_id(self) -> int:
        raise NotImplementedError

    def get_data_pieces(self) -> List[DataPiece]:
        raise NotImplementedError

    @property
    def num_records(self) -> int:
        """The number of records for this source shard.

        This should be equal with sum of all pandas DataFrame rows.
        """
        return sum([p.num_records for p in self.get_data_pieces()])

    def __iter__(self) -> Iterable[pd.DataFrame]:
        for piece in self.get_data_pieces():
            for pdf in iter(piece):
                yield pdf

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{self.prefix()}SourceShard[{self.shard_id}]"


def divide_data_pieces(data_pieces: List[Type[DataPiece]],
                       world_size: int) -> Dict[int, List[int]]:
    """Divide the data pieces into world_size partitions

    Divide the data pieces into world_size partitions and return a word rank
    to a list of data pieces mapping.
    Args:
        data_pieces (List[Type[DataPiece]]): the data pieces to be divided
        world_size (int): the data pieces will be divided into world_size
                          partitions
    Returns:
        a dict, the key is the world rank, and the value the data pieces
    """
    if len(data_pieces) < world_size:
        raise Exception("do not have enough data pieces to divide")
    results = {}
    tmp_queue = {}
    for i in range(world_size):
        results[i] = []
        tmp_queue[i] = 0
    sorted_pieces = sorted(data_pieces,
                           key=lambda item: item.num_records,
                           reverse=True)
    for piece in sorted_pieces:
        rank = sorted(tmp_queue, key=lambda x: tmp_queue[x])[0]
        results[rank].append(piece)
        tmp_queue[rank] = tmp_queue[rank] + piece.num_records

    return results
