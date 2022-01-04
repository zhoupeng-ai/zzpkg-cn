import torch
import torch.nn as nn


class KbqaTextCnn(nn.Module):
    def __init__(self, vocab_size, embed_size, output_size, padding_idx):
        super(KbqaTextCnn, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=[3, embed_size])
        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=[4, embed_size])
        self.cnn3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=[5, embed_size])

        self.pooled_1 = nn.MaxPool1d(kernel_size=8)
        self.pooled_2 = nn.MaxPool1d(kernel_size=7)
        self.pooled_3 = nn.MaxPool1d(kernel_size=6)

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(300, output_size)

    def forward(self, text):
        # text: batch_size, seq_length
        # batch_size, seq_length, embedding_size
        input_embed = self.embed(text)
        # batch_size, v_size, embed_size
        input_embed = self.dropout(input_embed).unsqueeze(1)
        # batch_size, 1, v_size, embed_size
        # cnn 需要的维度是（N, C, H, W）
        # 即（batch_size, channel_size, hidden_size, embed_size）
        cnn_output_1 = self.cnn1(input_embed).squeeze(-1)
        cnn_output_2 = self.cnn2(input_embed).squeeze(-1)
        cnn_output_3 = self.cnn3(input_embed).squeeze(-1)

        out_1 = self.pooled_1(cnn_output_1)
        out_2 = self.pooled_2(cnn_output_2)
        out_3 = self.pooled_3(cnn_output_3)

        out = torch.cat([out_1, out_2, out_3], dim=1).squeeze(-1)

        out = self.dropout(out)

        out = self.fc(out)

        return out