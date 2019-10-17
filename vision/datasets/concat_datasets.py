import bisect

from tensorflow.python.keras.utils.data_utils import Sequence


class ConcatDataset(Sequence):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        self.datasets = datasets
        self.data_length = 0
        for dataset in datasets:
            self.data_length += dataset.num_records
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.batch_size = datasets[0].batch_size

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        return self.datasets[dataset_idx][idx]

    def __len__(self):
        return self.cumulative_sizes[-1]
