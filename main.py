from model import YoutubeCommentsClassifier as Model
from trainer import train
from dataset import CommentDataset

if __name__ == '__main__':
    data_file = "data/youtube_dataset.csv"
    dataset = CommentDataset(data_file, batch_size=32)
    model = Model(dataset)

    train(dataset, model, "train.pt")
