from abc import ABC
from typing import Dict, List


class HackMetaData(ABC):
    def get_dtype(self, feature: str) -> str:
        pass

    def get_dtypes_dict(self) -> Dict[str, str]:
        pass
    
    def get_target_feature(self) -> str:
        pass


class HackatonMetaData(HackMetaData):
    def __init__(
            self,
            metadata: Dict,
    ):
        metadata = metadata["columns"]
        self.metadata = {key: item for key, item in metadata.items()}

    def _get_column_by_type(self, type: str) -> List[str]:
        columns = dict(filter(lambda seq: seq[1].get('type') == type, self.metadata.items()))
        return list(columns.keys())

    def get_dtype(self, feature: str) -> str:
        dtype = self.metadata.get(feature).get('dtype')
        return dtype

    def get_dtypes_dict(self) -> Dict[str, str]:
        dtype_dict = dict()
        for key in self.get_hopy_features():
            dtype_dict[key] = self.metadata.get(key).get('dtype')

        return dtype_dict

    def get_target_feature(self) -> str:
        features = self._get_column_by_type('target')
        return features[0]
