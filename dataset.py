from torch.utils.data import Dataset
import csv
import pandas as pd



data_file = "data/youtube_dataset.csv"


class CommentDataset(Dataset):
    def __init__(self, csv_file):
        super(CommentDataset, self).__init__()
        self._data = pd.read_csv(csv_file, error_bad_lines=False)
        self._comments = self._data['Comment']
        self._comments = self._comments.values.tolist()
        self._other_details = self._data[['Video Name', 'Channel Name', 'User Name', 'Date']].values.tolist()
        self._likes = self._data['Likes'].values.tolist()

        self.labels = self.like_range_bins([0, 10, 100, 1000, 10000, 100000, 1000000])

        self.channel_dict = dict()
        self.video_dict = dict()
        video_idx, channel_idx = 0, 0
        for vid in self._other_details:
            if vid[0] not in self.video_dict:
                self.video_dict[vid[0]] = video_idx
                video_idx += 1
            if vid[1] not in self.channel_dict:
                self.channel_dict[vid[1]] = channel_idx
                channel_idx += 1

    def like_range_bins(self, max_values):
        bins = {k: [] for k in max_values}
        for comment, n in enumerate(self._likes):
            key = min(x for x in max_values if x >= n)
            bins[key].append(comment)
        return bins

    def index_to_video_name(self, index):
        return list(self.video_dict)[index]

    def index_to_channel(self, index):
        return list(self.channel_dict)[index]

    def __getitem__(self, index):
        return_value = (self._comments[index], self._other_details[index])
        return return_value

    def __len__(self):
        return len(self._comments)


CommentDataset(data_file)