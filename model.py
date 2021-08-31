import torch
import torch.nn as nn
import gensim.downloader as api


# other features  details:
# video name represented by 1 parameter, channel name represented by 1 parameter, date represented by 6 parameters
other_features = 8
w2v_model = api.load("glove-twitter-100")


class YoutubeCommentsClassifier(nn.Module):
    def __init__(self, dataset, device, comment_embedding_dim=100, un_embedding_dim=16, c_LSTM_hidden=256, un_LSTM_hidden=32, c_LSTM_layers=6, un_LSTM_layers=3, LSTM_drop=0.4, LSTM_out=32, FC1_out=128, FC2_out=32, out_dim=6):
        super(YoutubeCommentsClassifier, self).__init__()

        self.batch_size = dataset.batch_size
        self.c_layers = c_LSTM_layers
        self.un_layers = un_LSTM_layers
        self.c_hidden = c_LSTM_hidden
        self.un_hidden = un_LSTM_hidden
        self.device = device

        self.embed_weights = torch.FloatTensor(w2v_model.vectors)

        # comments LSTM section
        self.comment_embedding = nn.Embedding.from_pretrained(self.embed_weights) #(num_embeddings=len(dataset.comment_dict), embedding_dim=comment_embedding_dim)
        self.comment_LSTM = nn.LSTM(input_size=comment_embedding_dim, hidden_size=c_LSTM_hidden, num_layers=c_LSTM_layers, dropout=LSTM_drop)
        self.comment_linear = nn.Linear(in_features=c_LSTM_hidden, out_features=LSTM_out)

        # user-name LSTM section
        self.user_embedding = nn.Embedding(num_embeddings=len(dataset.user_dict), embedding_dim=un_embedding_dim)
        self.user_LSTM = nn.LSTM(input_size=un_embedding_dim, hidden_size=un_LSTM_hidden, num_layers=un_LSTM_layers, dropout=LSTM_drop)
        self.user_linear = nn.Linear(in_features=un_LSTM_hidden, out_features=4)

        # non-time variant layers
        self.fc_1 = nn.Linear(in_features=(LSTM_out + 4 + other_features), out_features=FC1_out)
        self.batch_norm1 = nn.BatchNorm1d(LSTM_out + 4 + other_features)
        self.fc_2 = nn.Linear(in_features=FC1_out, out_features=FC2_out)
        self.batch_norm2 = nn.BatchNorm1d(FC1_out)
        self.fc_3 = nn.Linear(in_features=FC2_out, out_features=out_dim)
        self.batch_norm3 = nn.BatchNorm1d(out_dim)
        self.dropout_fc = nn.Dropout(p=0.4)

        # activation function
        self.activation = nn.LogSoftmax(dim=0)

    def forward(self, x, context_com, context_un):
        embed = self.comment_embedding(x.c)
        lstm, context_com = self.comment_LSTM(embed, context_com)
        comment_output = self.comment_linear(lstm)

        embed_un = self.user_embedding(x.un)
        lstm_un, context_un = self.user_LSTM(embed_un, context_un)
        un_output = self.user_linear(lstm_un)

        FC_input = torch.cat((torch.transpose(comment_output[-1, :, :].squeeze(), 0, 1), torch.transpose(un_output[-1, :, :].squeeze(), 0 , 1), x.vn.unsqueeze(dim=0), x.cn.unsqueeze(dim=0), x.d), dim=0)
        FC_input = torch.transpose(FC_input, 0, 1)

        y_hat = self.batch_norm1(FC_input)
        y_hat = self.fc_1(y_hat)
        y_hat = self.activation(y_hat)
        y_hat = self.dropout_fc(y_hat)
        y_hat = self.batch_norm2(y_hat)
        y_hat = self.fc_2(y_hat)
        y_hat = self.activation(y_hat)
        y_hat = self.dropout_fc(y_hat)
        y_hat = self.fc_3(y_hat)

        return y_hat

    def init_context(self):
        c_hidden = torch.zeros(self.c_layers, self.batch_size, self.c_hidden)
        c_cell = torch.zeros(self.c_layers, self.batch_size, self.c_hidden)
        un_hidden = torch.zeros(self.un_layers, self.batch_size, self.un_hidden)
        un_cell = torch.zeros(self.un_layers, self.batch_size, self.un_hidden)
        return (c_hidden.to(self.device), c_cell.to(self.device)), (un_hidden.to(self.device), un_cell.to(self.device))