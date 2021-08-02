from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pandas as pd
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, NestedField

data_file = "data/youtube_dataset.csv"


class CommentDataset(Dataset):
    def __init__(self, csv_file):
        super(CommentDataset, self).__init__()
        self._data = pd.read_csv(csv_file, error_bad_lines=False)
        self._comments = self._data['Comment']
        self._comments = self._comments.values.tolist()

        self.parser = Field(sequential=True, use_vocab=True, lower=True, init_token='<sos>', eos_token='<eos>', dtype=torch.long)

        self._other_details = self._data[['Video Name', 'Channel Name', 'User Name', 'Date']].values.tolist()
        self._likes = self._data['Likes'].values.tolist()

        self.labels, self.comment_labels = self.like_range_bins([0, 10, 100, 1000, 10000, 100000, 1000000])

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
        likes_label = [None for _ in range(len(self._likes))]
        for comment, n in enumerate(self._likes):
            key = min(x for x in max_values if x >= n)
            bins[key].append(comment)
            likes_label[comment] = key
        return bins, likes_label

    def index_to_video_name(self, index):
        return list(self.video_dict)[index]

    def index_to_channel(self, index):
        return list(self.channel_dict)[index]

    def __getitem__(self, index):
        in_value = (self._comments[index], self._other_details[index])
        return in_value, self.comment_labels[index]

    def __len__(self):
        return len(self._comments)


dataset = CommentDataset(data_file)
train_size = round(0.7*len(dataset))
test_size = round(0.2*len(dataset))
valid_size = len(dataset) - test_size - train_size
train, validation, test = random_split(dataset, [train_size, valid_size, test_size])
train_loader = DataLoader(dataset=train, batch_size=4, shuffle=True)
valid_loader = DataLoader(dataset=validation, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset=test, batch_size=4, shuffle=False)
x = next(iter(train_loader))
parser = Field(sequential=True, use_vocab=True, lower=True, init_token='<sos>', eos_token='<eos>', dtype=torch.long)
sentences = [sen.split() for sen in list(x[0][0])]
parser.build_vocab(train)
padded = parser.pad(sentences)
print(list(x[0][0]))
print(padded)

# x[0][0] - tuple of comments in batch
# x[0][1][0] - tuple of video names in batch
# x[0][1][1] - tuple of chanel names
# x[0][1][2] - tuple of names of comment writers
# x[0][1][3] - time of comment

