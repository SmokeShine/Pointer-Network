import torch
import torch.nn as nn
import tqdm

START_TOKEN = 0
END_TOKEN = 999


# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class EncoderRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=512, dropout=0.0):
        super(EncoderRNN, self).__init__()
        self.LSTMCell = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

    def forward(self, input):
        # https://i.stack.imgur.com/SjnTl.png
        # output - output for depth 1 for all time steps (history)
        # (hidden,cellstate) - hidden,cellstate by rolling over all timesteps (final)

        output, (hidden, cellstate) = self.LSTMCell(input)
        ## concatenate 0 to output for end token -> this is required for convex example in figure 1 as there is possibility of skipping a point.
        # end_token_pad = torch.zeros_like(hidden.permute(1, 0, 2))
        # padded_output = torch.cat((end_token_pad, output), dim=1)
        return output, hidden, cellstate


class Attention(nn.Module):
    def __init__(
        self,
        encoder_hidden_size=512,
        decoder_hidden_size=512,
    ):
        super(Attention, self).__init__()
        self.tanh = torch.tanh
        self.softmax = torch.softmax
        # 5 is dimensionality reduction. Paper does not talk apart from V being learnable and becomes a vector
        self.W1_encoder = nn.Linear(encoder_hidden_size, 5).to("mps")
        self.W2_decoder = nn.Linear(decoder_hidden_size, 5).to("mps")
        self.V = nn.Linear(5, 1).to(
            "mps"
        )  # V is a vector as encoder and decoder hidden state have same dimensions

    def forward(self, decoder_output, encoder_outputs):
        # https://miro.medium.com/v2/resize:fit:2000/format:webp/1*TPlS-uko-n3uAxbAQY_STQ.png
        """
        encoder_outputs
        encoder_hidden
        V
        W1
        W2
        """
        sum_ = self.W1_encoder(encoder_outputs) + self.W2_decoder(
            decoder_output
        )  # j belongs 1...n for encoder hidden states
        tanh_ = self.tanh(sum_)
        # torch.nn.Linear(in_features, out_features)
        logits = self.V(tanh_).squeeze(-1)
        softmax_ = self.softmax(logits, axis=1)  # CCE requires unnormalized logits
        maxvalue, pointer = softmax_.max(
            axis=1
        )  # categorical cross entropy can handle labels. No need to one hot encode. or apply softmax
        # *** RuntimeError: Expected floating point type for target with class probabilities, got Long
        # https://stackoverflow.com/questions/74541568/why-does-the-pytorch-crossentropyloss-use-label-encoding-instead-of-one-hot-enc
        return logits, pointer


class DecoderRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=512, dropout=0.0):
        super(DecoderRNN, self).__init__()
        self.LSTMCell = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

    def forward(
        self, input_coordinates, encoder_outputs, encoder_hidden, encoder_cellstate
    ):
        batch_size, n_steps, input_dimenions = encoder_outputs.shape
        # Input the start token
        decoder_input = torch.empty(batch_size, 2, device=torch.device("mps")).fill_(
            START_TOKEN
        )  # 2 for xy coordinates
        decoder_input = decoder_input.unsqueeze(
            1
        )  # 1 time step with 2coordinates (4,1,2)
        decoder_hidden = (encoder_hidden, encoder_cellstate)  # pass the hidden state
        decoder_outputs = []

        for i in tqdm.tqdm(range(n_steps + 1)):
            # Decode next
            (decoder_output, decoder_hidden) = self.LSTMCell(
                decoder_input, decoder_hidden
            )
            attention = Attention()  # length will come from broadcasting of 1..n for j
            logits, batch_pointers = attention(decoder_output, encoder_outputs)
            decoder_outputs.append(logits)

            next_input = torch.zeros((batch_size, 2)).to("mps")
            next_input = next_input.unsqueeze(
                1
            )  # torch.Size([4, 1, 2]) placeholder for storing next input
            for i in range(len(next_input)):
                next_input[i, :] = input_coordinates[i][batch_pointers[i]]
            decoder_input = next_input

            ## categorical cross entropy - how to use labels?
            # Use pointer as input for next roll

            # https://stackoverflow.com/questions/61187520/should-decoder-prediction-be-detached-in-pytorch-training
            # decoder_input = topi.squeeze(
            #     -1
            # ).detach()  # detach from history as input for next roll
        decoder_outputs = torch.stack(
            decoder_outputs, axis=1
        )  # time step representation. 4 batches choose 1 of 2 cities, repeat 3 times (last one for coming back)
        # https://discuss.pytorch.org/t/how-to-use-batch-size-with-crossentropyloss/101194
        decoder_outputs = decoder_outputs.permute(0, 2, 1)
        return decoder_outputs


class PointerNetwork(nn.Module):
    # Single LSTM Layer
    # - 256/512 hidden units
    # - SGD
    #     - learning rate 1.0
    # - Gradient Clipping 2.0
    # - Batch Size 128
    # - Random Uniform weight initialization (-0.08 to 0.08)
    # Question - how to create batch with variable length that includes start and end token
    # Question - how to calculate loss function?
    # if tokens are appended, how will this work in batch? because batch is created based on max length which is a function of input length

    def __init__(self):
        super(PointerNetwork, self).__init__()
        self.encoder = EncoderRNN()
        self.decoder = DecoderRNN()

    def forward(self, x):
        """
        x -> torch.Size([4, 2, 2]) > 4 batches of 2 cities with (x,y) coordinates , i.e. 2 time steps [batch,cities,coordinates]
        encoder_outputs - output from each time step for depth 1-> torch.Size([4, 2, 256]) > history of outputs for each time step
        encoder_hidden - state at final time step -> torch.Size([1, 4, 256]) > property of cell, does not depend on number of steps
        encoder_cellstate - state at final time step -> torch.Size([1, 4, 256]) > property of cell, does not depend on number of steps
        attention_output ->
        attention_hidden ->
        out ->
        """
        out = None
        encoder_outputs, encoder_hidden, encoder_cellstate = self.encoder(x)
        out = self.decoder(
            x, encoder_outputs, encoder_hidden, encoder_cellstate
        )  # need to pass for getting the coordinates of next pointer
        return out
