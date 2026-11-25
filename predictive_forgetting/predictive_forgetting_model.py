#!/usr/bin/env python3
"""
Predictive-forgetting autoencoder.

Directly implements the encoder/decoder architecture described in
"Why the Brain Consolidates: Predictive Forgetting for Optimal
Generalisation" (Fountas et al.), Section 3.1/3.2 and Methods
("Cortical Refinement (Autoencoder) Model").

This module defines only the generative model g_Phi (encoder) and
Dec_Theta (decoder): a small deterministic convolutional autoencoder,
g_Phi : X -> Z and Dec_Theta : Z -> X, matching the paper's own scale
(a handful of conv layers, a low-dimensional latent).

Trained via reconstruction plus successor-state prediction (Phase 1 in
the paper's Methods, with Y defined per Section 2.2 as the successor
state X_{t+1} for unsupervised/observational settings), this becomes
the frozen generative model that a later wake/sleep script uses to
"dream" -- decode(z) -- and consolidate frames via free-energy
inference (Eq. 16-17 in the paper). That inference logic is
intentionally NOT implemented here; this file is the model only.

Also includes a small successor predictor h_Psi: Z -> Z, used to give
the encoder pressure to retain what predicts the near future (Y) and
discard what doesn't -- the "predictive" half of predictive forgetting.
It reuses Dec_Theta to decode its predicted latent, so no separate
generative capacity is needed.
"""

from typing import Sequence, Tuple, Union

# torch is the go to ML library, i.e. has a lot of predefined functions for how to train a model
import torch
import torch.nn as nn

# Either a single int (square frames, backward-compatible with old checkpoints)
# or an explicit (height, width) pair for non-square frames.
Size = Union[int, Sequence[int]]


def _to_hw(size: Size) -> Tuple[int, int]:
    """Normalize an int (square) or (height, width) pair to a (height, width) tuple."""
    if isinstance(size, int):
        return size, size
    h, w = size
    return int(h), int(w)


class Encoder(nn.Module):
    """g_Phi: X -> Z. Deterministic conv stack + linear projection to latent_dim."""

    def __init__(self, in_channels: int, channels: Sequence[int], latent_dim: int, input_size: Size):
        super().__init__()
        n_down = len(channels)
        h, w = _to_hw(input_size)
        assert h % (2 ** n_down) == 0 and w % (2 ** n_down) == 0, (
            f"input_size ({h}x{w}) must be divisible by 2^{n_down} = {2 ** n_down} "
            f"in both dimensions for {n_down} stride-2 downsampling layers"
        )
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)

        self.final_spatial = (h // (2 ** n_down), w // (2 ** n_down))
        final_h, final_w = self.final_spatial
        flat_dim = channels[-1] * final_h * final_w
        self.to_latent = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        return self.to_latent(h)


class Decoder(nn.Module):
    """Dec_Theta: Z -> X. Mirrors the encoder via transposed convs. Outputs raw logits."""

    def __init__(self, out_channels: int, channels: Sequence[int], latent_dim: int, final_spatial: Tuple[int, int]):
        super().__init__()
        self.channels = list(channels)
        self.final_spatial = final_spatial
        final_h, final_w = final_spatial
        flat_dim = channels[-1] * final_h * final_w
        self.from_latent = nn.Linear(latent_dim, flat_dim)

        rev_channels = list(reversed(channels))
        layers = []
        for i in range(len(rev_channels) - 1):
            c_in, c_out = rev_channels[i], rev_channels[i + 1]
            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
        # Final layer: no BatchNorm/ReLU -- raw logits for BCEWithLogitsLoss.
        layers += [nn.ConvTranspose2d(rev_channels[-1], out_channels, kernel_size=4, stride=2, padding=1)]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        final_h, final_w = self.final_spatial
        h = h.view(-1, self.channels[-1], final_h, final_w)
        return self.deconv(h)


class SuccessorPredictor(nn.Module):
    """h_Psi: Z -> Z. Predicts the latent of the successor state Y (Section 2.2)."""

    def __init__(self, latent_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PredictiveForgettingAutoencoder(nn.Module):
    """
    g_Phi (encoder) + Dec_Theta (decoder), matching the paper's Methods
    ("Cortical Refinement (Autoencoder) Model" / Section 3.2 backbone).

    Deterministic: z = g_Phi(x) is a single point, not a distribution --
    the paper's stochasticity/prior pressure is introduced later, during
    wake/sleep *inference* (Langevin dynamics on z), not here.

    forward(x) returns reconstruction LOGITS; train against x in [0, 1]
    with BCEWithLogitsLoss, as in the paper's Methods.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 64, 64),
        latent_dim: int = 64,
        input_size: Size = 64,
    ):
        super().__init__()
        channels = list(channels)
        self.in_channels = in_channels
        self.channels = channels
        self.latent_dim = latent_dim
        self.input_size = input_size

        self.encoder = Encoder(in_channels, channels, latent_dim, input_size)
        self.decoder = Decoder(in_channels, channels, latent_dim, self.encoder.final_spatial)
        self.successor_predictor = SuccessorPredictor(latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def predict_successor(self, z: torch.Tensor) -> torch.Tensor:
        """h_Psi(z) decoded through Dec_Theta -- predicted logits for Y."""
        return self.decode(self.successor_predictor(z))
