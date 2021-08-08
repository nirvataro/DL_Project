from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, NestedField
import spacy
import re


class CommentDataset:
    def __init__(self, csv_file, batch_size=4):
        self.batch_size = batch_size
        # spacy english tokenizer
        self.spacy_en = spacy.load('en_core_web_sm')

        # parsing fields
        self.comment_parser = Field(sequential=True, use_vocab=True, lower=True, tokenize=self.tokenize, init_token="<sos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
        self.video_name_parser = Field(sequential=False, use_vocab=True, tokenize=self.tokenize)
        self.channel_parser = Field(sequential=False, use_vocab=True)
        self.user_parser = Field(sequential=True, use_vocab=True, tokenize=lambda x: [char for char in x], init_token="<sos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
        self.date_parser = Field(sequential=True, use_vocab=False, tokenize=self.date_tokenizer)
        self.label_parser = Field(sequential=False, use_vocab=False)

        field = {'Comment': ('c', self.comment_parser),
                 'Video Name': ('vn', self.video_name_parser),
                 'Channel Name': ('cn', self.channel_parser),
                 'User Name': ('un', self.user_parser),
                 'Date': ('d', self.date_parser),
                 'Likes': ('y', self.label_parser)
                 }

        # dataset
        self.sets = TabularDataset(path=csv_file, format='csv', fields=field)

        for exp in self.sets.examples:
            # change exact like number to bins (output)
            for b in range(6):
                if int(exp.y) // 10**b == 0:
                    exp.y = torch.tensor(b)
                    break
            if type(exp.y) == str:
                exp.y = torch.tensor(6)

        train, test = self.sets.split(split_ratio=0.7)
        test, validation = test.split(split_ratio=0.66)
        self.comment_parser.build_vocab(self.sets, min_freq=2)
        self.video_name_parser.build_vocab(self.sets)
        self.channel_parser.build_vocab(self.sets)
        self.user_parser.build_vocab(self.sets)
        self.comment_dict = self.comment_parser.vocab.stoi
        self.channel_dict = self.channel_parser.vocab.stoi
        self.video_dict = self.video_name_parser.vocab.stoi
        self.user_dict = self.user_parser.vocab.stoi

        self.train_iter, self.valid_iter, self.test_iter = BucketIterator.splits((train, validation, test), batch_size=self.batch_size, sort_key=lambda x: len(x.c), sort=False, sort_within_batch=True)
        self.train_iter.create_batches()
        self.valid_iter.create_batches()
        self.test_iter.create_batches()

    def tokenize(self, text):
        return [token.text for token in self.spacy_en.tokenizer(text)]

    def date_tokenizer(self, d):
        arr = re.split(r"[-T:Z]+", d)[:-1]
        return [int(x) for x in arr]

    def index_to_video_name(self, index):
        return list(self.video_dict)[index]

    def index_to_channel(self, index):
        return list(self.channel_dict)[index]



data_file = "data/youtube_dataset.csv"

