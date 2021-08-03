from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, NestedField
import spacy
import re




class CommentDataset(Dataset):
    def __init__(self, csv_file):
        super(CommentDataset, self).__init__()

        # spacy english tokenizer
        self.spacy_en = spacy.load('en_core_web_sm')

        # parsing fields
        self.comment_parser = Field(sequential=True, use_vocab=True, lower=True, tokenize=self.tokenize, init_token="<sos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
        self.video_name_parser = Field(sequential=False, use_vocab=True, tokenize=self.tokenize)
        self.channel_parser = Field(sequential=False, use_vocab=True)
        #self.user_parser = Field(sequential=True, use_vocab=True, tokenize=lambda x: [char for char in x], init_token="<sos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
        self.other_parser = Field(sequential=False, use_vocab=False)
        self.label_parser = Field(sequential=False, use_vocab=False)

        field = {'Comment': ('c', self.comment_parser),
                 'Video Name': ('vn', self.video_name_parser),
                 'Channel Name': ('cn', self.channel_parser),
                 #'User Name': ('un', self.user_parser),
                 'Date': ('d', self.other_parser),
                 'Likes': ('y', self.label_parser)}

        # dataset
        self.sets = TabularDataset(path='data/youtube_dataset.csv', format='csv', fields=field)

        train, test = self.sets.split(split_ratio=0.7)
        test, validation = test.split(split_ratio=0.66)
        self.comment_parser.build_vocab(self.sets, min_freq=2)
        self.video_name_parser.build_vocab(self.sets)
        self.channel_parser.build_vocab(self.sets)
        #self.user_parser.build_vocab(self.sets)
        self.comment_dict = self.comment_parser.vocab.stoi
        self.channel_dict = self.channel_parser.vocab.stoi
        self.video_dict = self.video_name_parser.vocab.stoi
        #self.user_dict = self.user_parser.vocab.stoi

        self.update_to_tensors()

        self.train_iter, self.valid_iter, self.test_iter = BucketIterator.splits((train, validation, test), batch_size=4, sort_key=lambda x: len(x.c), sort=False, sort_within_batch=True)
        self.train_iter.create_batches()
        self.valid_iter.create_batches()
        self.test_iter.create_batches()

    def tokenize(self, text):
        return [token.text for token in self.spacy_en.tokenizer(text)]

    def update_to_tensors(self):
        for exp in self.sets.examples:
            # change exact like number to bins (output)
            for b in range(6):
                if int(exp.y) // 10**b == 0:
                    exp.y = torch.tensor(b)
                    break
            if type(exp.y) == str:
                exp.y = torch.tensor(6)

            # change string of time to date tensor
            arr = re.split(r"[-T:Z]+", exp.d)[:-1]
            arr = [int(x) for x in arr]
            exp.d = torch.tensor(arr)

            # string comment to vocabulary indexes
            exp.c = torch.tensor([self.comment_dict[x] for x in exp.c])

            # channel+video name to index
            exp.cn = torch.tensor(self.channel_dict[exp.cn])
            exp.vn = torch.tensor(self.video_dict[exp.vn])

            # user name to tensor
            #exp.un = torch.tensor([self.user_dict[x] for x in exp.un])

    def index_to_video_name(self, index):
        return list(self.video_dict)[index]

    def index_to_channel(self, index):
        return list(self.channel_dict)[index]

    def __getitem__(self, index):
        in_value = (self._comments[index], self._other_details[index])
        return in_value, self.comment_labels[index]

    def __len__(self):
        return len(self._comments)


data_file = "data/youtube_dataset.csv"
dataset = CommentDataset(data_file)
for b in dataset.test_iter:
    x = b.c
