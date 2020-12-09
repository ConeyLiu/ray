from typing import List, Iterable

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
        raise NotImplementedError

    def __iter__(self) -> Iterable[pd.DataFrame]:
        raise NotImplementedError

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{self.prefix()}SourceShard[{self.shard_id}]"
