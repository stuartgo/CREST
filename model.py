import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_tcn import TCN
from mamba_ssm import Mamba2

from TransformerEncoder import TransformerEncoder

torch.manual_seed(42)


def positional_embeddings(shape):
    batch_size, seq_len, embed_dim = shape
    positions = torch.arange(0, seq_len).unsqueeze(1)
    denominators_sin = torch.pow(10000.0, 2 * torch.arange(0, embed_dim, 2) / embed_dim)
    denominators_cos = torch.pow(10000.0, 2 * torch.arange(1, embed_dim, 2) / embed_dim)

    embeddings = torch.zeros(seq_len, embed_dim)
    embeddings[:, 0::2] = torch.sin(positions / denominators_sin)
    embeddings[:, 1::2] = torch.cos(positions / denominators_cos)

    return embeddings.repeat(batch_size, 1, 1)


class CircadianModel(pl.LightningModule):
    def __init__(self, input_size=10, output_size=1, model_type="transformer", params={}):
        super().__init__()
        self.save_hyperparameters()

        params = params.copy()
        self.params = params
        self.model_type = model_type
        self.lr = params.pop("learning_rate")
        hidden_size = params.pop("d_model")

        if model_type != "moment":
            self.embedder = nn.Linear(input_size, hidden_size)

        if model_type == "lstm":
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, **params)
            lstm_out_size = hidden_size * 2 if params.get("bidirectional") else hidden_size
            self.classifier = nn.Linear(lstm_out_size, output_size)

        elif model_type == "transformer":
            self.encoder = TransformerEncoder(input_dim=hidden_size, **params)
            self.classifier = nn.Linear(hidden_size, output_size)
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        elif model_type == "tcn":
            self.tcn = TCN(num_inputs=hidden_size, input_shape="NLC", **params)
            self.classifier = nn.Linear(hidden_size, output_size)

        elif model_type == "mamba":
            self.mamba = Mamba2(d_model=hidden_size, **params)
            self.classifier = nn.Linear(hidden_size, output_size)

        elif model_type == "moment":
            self.classifier = nn.Linear(input_size, output_size)

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        if self.model_type == "lstm":
            x = self.embedder(x)
            _, (h_n, _) = self.lstm(x)
            if self.params.get("bidirectional"):
                h_n = torch.cat((h_n[0], h_n[1]), dim=1)
            else:
                h_n = h_n[0]
            return self.classifier(h_n)

        elif self.model_type == "transformer":
            x = self.embedder(x)
            x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = x + positional_embeddings(x.shape).to(x.device)
            x = self.encoder(x)
            return self.classifier(x[:, 0, :])

        elif self.model_type == "tcn":
            x = self.embedder(x)
            x = self.tcn(x)
            return self.classifier(x.mean(dim=1))

        elif self.model_type == "mamba":
            x = self.embedder(x)
            x = self.mamba(x)
            return self.classifier(x[:, 0, :])

        elif self.model_type == "moment":
            return self.classifier(x)

    def get_attention_weights(self, x):
        x = self.embedder(x)
        x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + positional_embeddings(x.shape).to(x.device)
        return self.encoder.get_attention_maps(x)

    def circular_mae_loss(self, y_pred, y_true):
        hour_pred = torch.atan2(y_pred[:, 0], y_pred[:, 1]) * 24 / (2 * torch.pi)
        hour_true = torch.atan2(y_true[:, 0], y_true[:, 1]) * 24 / (2 * torch.pi)
        diff = torch.abs(hour_true - hour_pred)
        return torch.minimum(diff, 24 - diff).mean()

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        mse = self.loss_fn(y_hat, y)
        mae = self.circular_mae_loss(y_hat, y)
        return mse, mae

    def training_step(self, batch, batch_idx):
        mse, mae = self._shared_step(batch)
        self.log("train_loss", mse, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_hour_loss", mae, on_epoch=True, prog_bar=True, sync_dist=True)
        return mse

    def validation_step(self, batch, batch_idx):
        mse, mae = self._shared_step(batch)
        self.log("val_loss", mse, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_hour_loss", mae, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        mse, mae = self._shared_step(batch)
        self.log("test_loss", mse, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_hour_loss", mae, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
