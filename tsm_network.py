import torch
import torch.nn as nn
from mlp_mixer import MixerBlock

LONG_LAYER_NUM = 3
DEFAULT_ENCODER_HIDDEN = 1024
CHANNEL_NUM = 4


def pairwise_cosine_similarity(x, y):
    """Compute self-pairwise cosine similarity.

    Args:
        x: torch.FloatTensor [B, C, L, E']
        y: torch.FloatTensor [B, C, L, E']

    Returns:
        Cosine similarity matrix [B, C, L, L].
    """
    x = x.detach()
    y = y.permute(0, 1, 3, 2)
    dot = torch.matmul(x, y)
    x_dist = torch.norm(x, p=2, dim=3, keepdim=True)
    y_dist = torch.norm(y, p=2, dim=2, keepdim=True)
    dist = x_dist * y_dist
    cos = dot / (dist + 1e-8)
    return cos


def pairwise_minus_l2_distance(x, y):
    """Compute pairwise negative L2 distance.

    Args:
        x: torch.FloatTensor [B, C, L, E']
        y: torch.FloatTensor [B, C, L, E']

    Returns:
        Negative L2 distance matrix [B, C, L, L].
    """
    x = x.unsqueeze(3).detach()
    y = y.unsqueeze(2)
    l2_dist = torch.sqrt(torch.sum((x - y) ** 2, dim=-1) + 1e-8)
    return -l2_dist


class CustomEncoder(nn.Module):
    def __init__(self, hidden, frame_size):
        super().__init__()
        self.feature_reduction = nn.Sequential(
            nn.Linear(2048, hidden),
            nn.GELU()
        )
        self.short_layer = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 1),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 1),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 1),
        )
        self.middle_layer_1 = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=1),
        )
        self.middle_layer_2 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
        )
        self.mixer_layers = nn.ModuleList(
            [MixerBlock(frame_size, hidden, hidden // 2, hidden * 2) for _ in range(LONG_LAYER_NUM)]
        )
        self.layernorm = nn.LayerNorm(hidden)

    def forward(self, x):
        """Encode frame features through multi-scale temporal branches.

        Args:
            x: torch.FloatTensor [B, L, E] where E=2048.

        Returns:
            Multi-channel encoded features [B, C, L, E'].
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        reduced_feature = self.feature_reduction(x).permute(0, 2, 1)  # [B, E', L]
        short = self.short_layer(reduced_feature) + reduced_feature
        middle_1 = self.middle_layer_1(reduced_feature) + reduced_feature
        middle_2 = self.middle_layer_2(middle_1) + middle_1

        long = reduced_feature.permute(0, 2, 1)  # [B, L, E']
        for layer in self.mixer_layers:
            long = layer(long)
        long = self.layernorm(long)
        long = long.permute(0, 2, 1)

        out_list = [short, middle_1, middle_2, long]
        out = torch.stack(out_list, dim=1)  # [B, C, L, E']

        assert CHANNEL_NUM % out.size(1) == 0
        group_num = CHANNEL_NUM // out.size(1)
        out = out.view(out.size(0), out.size(1), group_num, out.size(2) // group_num, out.size(3))
        out = out.view(out.size(0), -1, out.size(3), out.size(4)).permute(0, 1, 3, 2)
        return out


class SJNET(nn.Module):
    def __init__(self, frame_size, encoder_hidden=DEFAULT_ENCODER_HIDDEN, channel_num=4, decoder_hidden=256):
        super().__init__()
        self.encoder = CustomEncoder(encoder_hidden, frame_size)
        self.normalize_tsm = nn.InstanceNorm2d(channel_num, affine=False)
        self.opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x):
        out = self.encoder(x)
        tsm = self.normalize_tsm(pairwise_cosine_similarity(out, out))
        return tsm

    def get_tsm(self, x):
        """Compute the averaged temporal self-similarity matrix.

        Args:
            x: Input features [B, L, E].

        Returns:
            TSM as numpy array [B, L, L, 1].
        """
        with torch.no_grad():
            out = self.encoder(x)
            tsm = self.normalize_tsm(pairwise_cosine_similarity(out, out))
            tsm = torch.mean(tsm, dim=1).unsqueeze(1)
            tsm = tsm.permute(0, 2, 3, 1)
            tsm = tsm.detach().cpu().numpy()
        return tsm
