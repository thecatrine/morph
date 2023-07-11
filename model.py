import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from . import model_utils as utils

log = logging.getLogger(__name__)

OUT_CHANNELS = 16

class Residual(utils.TimestepBlock):
    def __init__(self, in_channels, out_channels, embedding_channels, dropout_rate, normalization_groups):
        super(Residual, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_channels = embedding_channels
        self.dropout_rate = dropout_rate
        self.normalization_groups = normalization_groups

        self.norm1 = nn.GroupNorm(normalization_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Map embeddings into correct dimension
        # Why do we add rather than concatenate? something something information something?
        self.emb_mapping = nn.Linear(embedding_channels, out_channels)
        self.norm2 = nn.GroupNorm(normalization_groups, out_channels)

        self.droupout = nn.Dropout(p=dropout_rate)
        self.conv2 = utils.zero_module(
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if in_channels == out_channels:
            self.residual_connection = nn.Identity()
        else:
            self.residual_connection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, orig_batch, timesteps):
        # batch is [N x C x H x W]
        batch = self.norm1(orig_batch)
        batch = F.silu(batch)
        batch = self.conv1(batch)

        # Map timesteps N x C
        timesteps = self.emb_mapping(F.silu(timesteps))

        # Add timesteps into batch
        timesteps = timesteps.unsqueeze(-1)
        timesteps = timesteps.unsqueeze(-1)

        batch = batch + timesteps

        batch = F.silu(self.norm2(batch))

        batch = self.droupout(batch)
        batch = self.conv2(batch)

        resid = self.residual_connection(orig_batch)

        return batch + resid

class Attention(nn.Module):
    def __init__(self, channels, normalization_groups, num_head_channels):
        super().__init__()
        self.channels = channels
        self.num_head_channels = num_head_channels

        log.debug(channels, num_head_channels)
        assert channels % num_head_channels == 0
        self.num_heads = channels // num_head_channels


        self.norm1 = nn.GroupNorm(normalization_groups, channels)

        # Triple number of channels for q,k,v
        self.conv1 = nn.Conv1d(channels, channels*3, 1)

        # Do we want dropout here?
        self.attention = utils.QKVAttention(n_heads=self.num_heads)

        # Why do we have this
        self.conv2 = utils.zero_module(
            nn.Conv1d(channels, channels, 1)
        )

    def forward(self, orig_batch):
        # Why don't we do silu here too?
        batch = self.norm1(orig_batch)

        assert(self.channels == batch.shape[1])
        batch = batch.reshape(batch.shape[0], self.channels, -1)

        qkv = self.conv1(batch)
        batch = self.attention(qkv)

        batch = self.conv2(batch)

        return batch.reshape(orig_batch.shape) + orig_batch

class Diffuser(torch.nn.Module):
    def __init__(self,
                 dropout_rate=0.1,
                 normalization_groups=32,
                 channels=192,
                 num_head_channels=64,
                 num_residuals=3,
                 in_channels=3,
                 out_channels=3,
                 interior_attention=True,
                 channel_multiple_schedule=[1, 2, 3]):
        super(Diffuser, self).__init__()
        # input is b x in_channels x 28 x 28
        self.time_embed = None
        self.in_layer = None

        self.down_layers = nn.ModuleList([])
        self.middle_layer = None
        self.up_layers = nn.ModuleList([])

        self.out_layer = None

        self.channel_multiple_schedule = channel_multiple_schedule
        self.skip_sizes = []

        self.interior_attention = interior_attention

        # State used in forward
        self.channels = channels

        # TODO: Why do we use this number of timestamp channels
        timestamp_channels = 4 * self.channels
        self.timestamp_channels = timestamp_channels

        # Learned linear layers for timestamp embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.channels, self.timestamp_channels),
            nn.SiLU(),
            nn.Linear(self.timestamp_channels, self.timestamp_channels),
        )
        log.debug("Timestamps ", self.channels, self.timestamp_channels)

        # Initial convolution layer to higher channels
        self.in_layer = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        log.debug("Convolution ", in_channels, channels)

        # Do layers on the way down
        # (Res -> Attn) -> Downsample
        cur_channels = self.channels
        for i, channel_multiple in enumerate(self.channel_multiple_schedule):
            is_last_block = (i == len(self.channel_multiple_schedule) - 1)
            channels_out = self.channels*channel_multiple

            for j in range(num_residuals):
                sub_layers = []

                sub_layers.append(Residual(cur_channels, channels_out, timestamp_channels, dropout_rate, normalization_groups))
                log.debug("Adding residual ", cur_channels, channels_out)
                cur_channels = channels_out

                if channel_multiple >= self.interior_attention:
                    sub_layers.append(Attention(cur_channels, normalization_groups, num_head_channels))
                    log.debug("Adding Attention ", cur_channels, channels_out)

                self.skip_sizes.append(cur_channels)
                self.down_layers.append(utils.TimestepEmbedSequential(*sub_layers))
            log.debug("---")
            # We downsample except right before the middle layers
            if not is_last_block:
                self.skip_sizes.append(cur_channels)
                self.down_layers.append(
                    utils.TimestepEmbedSequential(nn.AvgPool2d(kernel_size=2, stride=2))
                )
                log.debug("<< Downsample ", cur_channels, channels_out)

        # Middle layers
        self.middle_layer = utils.TimestepEmbedSequential(
            Residual(cur_channels, cur_channels, timestamp_channels, dropout_rate, normalization_groups),
            Attention(cur_channels, normalization_groups, num_head_channels),
            Residual(cur_channels, cur_channels, timestamp_channels, dropout_rate, normalization_groups),
        )

        log.debug("---")
        log.debug("Middle layer ", cur_channels, cur_channels)
        log.debug("---")
        # Do layers on the way up
        # (Res -> Attn) -> Res -> Attn -> Upsample
        for i, channel_multiple in reversed(list(enumerate(self.channel_multiple_schedule))):
            is_last_block = (i == 0)
            channels_out = self.channels*channel_multiple

            # We need extra channels here because we add in skip connections
            for j in range(num_residuals):
                skip_channels = cur_channels + self.skip_sizes.pop()
                log.debug ("         Skip size was: ", skip_channels)
                sub_layers = []
                sub_layers.append(Residual(skip_channels, channels_out, timestamp_channels, dropout_rate, normalization_groups))
                log.debug("Adding residual ", cur_channels, channels_out)
                cur_channels = channels_out

                if channel_multiple >= self.interior_attention:
                    sub_layers.append(Attention(cur_channels, normalization_groups, num_head_channels))
                    log.debug("Adding Attention ", cur_channels, channels_out)
                self.up_layers.append(utils.TimestepEmbedSequential(*sub_layers))

            if not is_last_block:
                skip_channels = cur_channels + self.skip_sizes.pop()
                log.debug ("Skip size was: ", skip_channels)
                # We need extra channels here because we add in skip connections
                arr = []
                arr.append(Residual(skip_channels, cur_channels, timestamp_channels,
                                    dropout_rate, normalization_groups))
                if channel_multiple >= self.interior_attention:
                    arr.append(Attention(cur_channels, normalization_groups, num_head_channels))
                arr.append(utils.InvokeFunction(F.interpolate, scale_factor=2, mode='nearest'))
                self.up_layers.append(utils.TimestepEmbedSequential(*arr))
                log.debug(">> Adding Upsample ", cur_channels, cur_channels)

        self.out_layer = nn.Sequential(
            nn.GroupNorm(normalization_groups, cur_channels),
            nn.SiLU(),
            utils.zero_module(
                nn.Conv2d(in_channels=cur_channels, out_channels=out_channels, kernel_size=3, padding=1)
            ),
        )

    def forward_with_cond_scale(self, *a, **kw):
        return self.forward(*a, **kw)

    def forward(self, orig_batch, timesteps, *a, **kw):
        if timesteps.dtype == torch.long:
            timesteps = timesteps.to(torch.float32)
        skip_connections = []

        # Get embedded timesteps by blowing up with a linear layer
        if timesteps.dtype == torch.long:
            timesteps = timesteps.to(torch.float32)
        embedded_timesteps = utils.timestep_embedding(timesteps, self.channels)
        embedded_timesteps = embedded_timesteps.to(timesteps.dtype)

        embedded_timesteps = self.time_embed(embedded_timesteps)

        # Do input layer
        batch = self.in_layer(orig_batch)

        #log.debug(batch)
        # Do downsamplings
        for layer in self.down_layers:
            batch = layer(batch, embedded_timesteps)
            skip_connections.append(batch)

        # Do middle layer
        batch = self.middle_layer(batch, embedded_timesteps)

        # Do upsamplings
        for layer in self.up_layers:
            batch = torch.cat([batch, skip_connections.pop()], dim=1)
            batch = layer(batch, embedded_timesteps)

        # Do output layer
        batch = self.out_layer(batch)

        return batch


def get_model(training_config):
    model_config = training_config['model']

    return Diffuser(**model_config)