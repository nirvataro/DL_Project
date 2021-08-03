import torch
import torch.nn as nn

other_features = 4

class YoutubeCommentsClassifier(nn.Module):
    def __init__(self, embedding_dim=128, LSTM_hidden=64, LSTM_layers=3, LSTM_drop=0.2, LSTM_out=16, FC1_out=128, FC2_out=32, out_dim=6):
        super(YoutubeCommentsClassifier, self).__init__()

        # comments LSTM section
        self.comment_embedding = nn.Embedding(num_embeddings=, embedding_dim=embedding_dim)
        self.comment_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=LSTM_hidden, num_layers=LSTM_layers, dropout=LSTM_drop)
        self.comment_linear = nn.Linear(in_features=LSTM_hidden, out_features=LSTM_out)

        # non-time variant layers
        self.fc_1 = nn.Linear(in_features=(LSTM_out + other_features), out_features=FC1_out)
        self.fc_2 = nn.Linear(in_features=FC1_out, out_features=FC2_out)
        self.fc_3 = nn.Linear(in_features=FC2_out, out_features=out_dim)

        # activation function
        self.activation = nn.LogSoftmax()

    def forward(self, x, context):
        comment, video_inputs = x
        embed = self.comment_embedding(comment)
        lstm, context = self.comment_LSTM(embed, context)
        comment_output = self.comment_linear(lstm)

        FC_input = torch.cat((comment_output, video_inputs))
        y_hat = self.fc_1(FC_input)
        y_hat = self.activation(y_hat)
        y_hat = self.fc_2(y_hat)
        y_hat = self.activation(y_hat)
        y_hat = self.fc_3(y_hat)
        return y_hat

    def init_context(self, batch_size):
        hidden = torch.zeros(self.layers, batch_size, self.hidden)
        cell = torch.zeros(self.layers, batch_size, self.hidden)
        return hidden, cell
