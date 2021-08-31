from trainer import train
from dataset import CommentDataset
from model import YoutubeCommentsClassifier as Model
import torch


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_file = "data/youtube_dataset.csv"
    dataset = CommentDataset(data_file, batch_size=32, device=device)
    model = Model(dataset, device).to(device)

    train(dataset, model, "train.pt", device=device)
