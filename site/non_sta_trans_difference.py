import numpy as np
import torch
import torch.nn as nn
from Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from SelfAttention_Family import DSAttention, AttentionLayer
import positional_encoder as pos
from Embed import DataEmbedding


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''

    def __init__(self, input_size,seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
                        # 1,   96,   p_hidden_dim[256,256],    2        1
        super(Projector, self).__init__()
        print('.......................',input_size)

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.kernal_size = kernel_size
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)
        layers = [nn.Linear(2 * input_size, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        # print('.......................', input_size)
        # print(layers)
        # exit()
        #[Linear(in_features=2, out_features=256, bias=True), ReLU(), Linear(in_features=256, out_features=256, bias=True), ReLU(), Linear(in_features=256, out_features=1, bias=False)]

        # print(layers)
        # exit()

        self.backbone = nn.Sequential(*layers)  # runs layers sequentially

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O

        batch_size = x.shape[0]  # 32

        x = self.series_conv(x)

        x = torch.cat([x, stats], dim=1)  # B x 2 x E  #stats = standard deviation.
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O
        # print('y.shape',y,y.shape)
        # exit()
        return y


class TimeSeriesTransformer(nn.Module):
    """
    Non-stationary Transformer
    """

    def __init__(self, input_size,
                 batch_first: bool,

                 seq_len,
                 pred_len,
                 label_len,
                 dim_val: int = 512,
                 n_encoder_layers: int = 2,
                 n_decoder_layers: int = 1,
                 n_heads: int = 8,
                 dropout_encoder: float = 0.2,
                 dropout_decoder: float = 0.2,
                 dropout_pos_enc: float = 0.1,
                 dim_feedforward_encoder: int = 2048,
                 dim_feedforward_decoder: int = 2048,
                 num_predicted_features: int = 1,
                 activation='relu'):
        super(TimeSeriesTransformer, self).__init__()

        # Embedding
        self.seq_len = seq_len
        self.label_len=label_len
        self.pred_len=pred_len


        # print("input_size is: {}".format(input_size))
        # print("dim_val is: {}".format(dim_val))

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
        )

        self.decoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
        )

        # Create positional encoder
        self.positional_encoding_layer = pos.PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc,
            batch_first=True
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, attention_dropout=dropout_encoder,output_attention=False),
                        dim_val,
                        n_heads   ),
                    dim_val,
                    dim_feedforward_encoder,
                    dropout=dropout_encoder,
                    activation=activation
                ) for l in range(n_encoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(dim_val)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, attention_dropout=dropout_decoder, output_attention=False),
                        dim_val, n_heads),
                    AttentionLayer(
                        DSAttention(False, attention_dropout=dropout_decoder, output_attention=False),
                        dim_val, n_heads),
                    dim_val,
                    dim_feedforward_decoder,
                    dropout=dropout_decoder,
                    activation=activation,
                )
                for l in range(n_decoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(dim_val),
            projection=nn.Linear(dim_val, num_predicted_features, bias=True)
        )

        self.tau_learner = Projector(input_size=input_size, seq_len=seq_len, hidden_dims=[256,256],
                                     hidden_layers=2, output_dim=1)

        self.delta_learner = Projector(input_size=input_size, seq_len=seq_len, hidden_dims=[256,256],
                                     hidden_layers=2, output_dim=seq_len)

    def forward(self, x_enc,y_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # print("x_enc",x_enc)


        x_raw = x_enc.clone().detach()  # torch.Size([32, 96, 7])
        x_raw_lagged = torch.cat([x_raw[:, 0:1, :], x_raw[:, :-1, :]], dim=1)

        #normalization module:
        x_dash=torch.diff(x_raw,dim=1).to(x_enc.device).clone()
        x_dash=  torch.cat([torch.zeros_like(x_dash[:,0:1,:]),x_dash],dim=1).to(x_enc.device).clone()
        # print(" fixed_x_dash",x_dash)
        # exit()



        # mean
        mean_enc = x_raw.mean(1, keepdim=True).clone().detach()  # B x 1 x E
        # x_enc = x_enc - mean_enc
        #standard deviation
        std_enc = torch.sqrt(torch.var(x_raw, dim=1, keepdim=True, unbiased=False) + 1e-5).to(x_enc.device).clone()  # B x 1 x E


        x_dash_dec = torch.cat([x_dash[:, -self.label_len:, :], y_dec],
                              dim=1).to(x_enc.device).clone()
        # print("x_dash_dec",x_dash_dec.shape)
        # exit()

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar

        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        # print('delta',delta)
        # exit()
        # Model Inference
        enc_out = self.encoder_input_layer(x_dash)
        enc_out = self.positional_encoding_layer(enc_out)
        # print('enc-out',enc_out.shape)
        # exit()
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)
        # print('enc-out', enc_out)
        # exit()


        dec_out = self.decoder_input_layer(x_dash_dec)
        # print('dec_out)',dec_out)
        dec_out=self.positional_encoding_layer(dec_out)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)
        # print('dec_out)', dec_out)
        # exit()
        # De-normalization
        dec_out=dec_out[:, -self.pred_len:, :]
        # print('dec_out)', dec_out)
        # exit()

        dec_out = dec_out + x_raw_lagged
        # print('dec_out)', dec_out,dec_out.shape)

        # exit()

        # if self.output_attention:
        #     return dec_out[:, -self.dec_seq_len::, :], attns
        # else:
        return dec_out # [B, L, D]