# -*- coding: utf-8 -*-
"""
PyTorch implementation of FCN_AE — Fully-Connected Network Autoencoder
for anomalous sound detection.

Architecture matches the original Chainer-based model exactly:
  - Mel-filterbank features (NumFB dim) with frame concatenation
  - Encoder: BN → frame_concat → Linear → ... → bottleneck (z_dim)
  - Decoder: bottleneck → Linear → ... → reconstruction
  - Anomaly score: per-frame MSE between input and reconstruction

Reference:
  Y. Koizumi, et al., "ToyADMOS: A Dataset of Miniature-Machine Operating
  Sounds for Anomalous Sound Detection," WASPAA 2019.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Frame concatenation (context window) ─────────────────────────────────────

def frame_concat(h: torch.Tensor, bw: int, fw: int) -> torch.Tensor:
    """
    Concatenate past (bw) and future (fw) frames for each time step.
    
    Args:
        h:  Input features, shape (K, O) where K = time frames, O = feature dim
        bw: Number of backward (past) frames to include
        fw: Number of forward (future) frames to include
    
    Returns:
        Concatenated features, shape (K, O * (1 + bw + fw))
    """
    p = h                           # (K, O)
    K, O = h.shape
    # Past frames (shift up, pad zeros at bottom)
    for ii in range(bw):
        z = torch.zeros((ii, O), dtype=h.dtype, device=h.device)
        q = torch.cat((p[ii:, :], z), dim=0)
        h = torch.cat((h, q), dim=1)
    # Future frames (shift down, pad zeros at top)
    for ii in range(fw):
        z = torch.zeros((ii + 1, O), dtype=h.dtype, device=h.device)
        q = torch.cat((z, p[0:K - (ii + 1), :]), dim=0)
        h = torch.cat((h, q), dim=1)
    return h


# ── FCN Autoencoder ──────────────────────────────────────────────────────────

class FCN_AE(nn.Module):
    """
    Fully-Connected Network Autoencoder for anomalous sound detection.
    
    Architecture:
        Input  → BN  → FrameConcat → Encoder (FC layers) → Bottleneck (z_dim)
              → Decoder (FC layers) → Reconstruction → MSE anomaly score
    
    Args:
        in_dim:   Input feature dimension (NumFB = number of mel filter banks)
        hid_dim:  Hidden layer dimension
        z_dim:    Bottleneck (latent) dimension
        num_hid:  Number of hidden layers in encoder AND decoder
        num_fw:   Number of future frames for context concatenation
        num_bw:   Number of past frames for context concatenation
    """
    
    def __init__(self, in_dim, hid_dim, z_dim, num_hid, num_fw, num_bw):
        super(FCN_AE, self).__init__()
        
        self.num_fw = num_fw
        self.num_bw = num_bw
        self.num_hid = num_hid
        
        # Effective input dim after frame concatenation
        concat_dim = in_dim * (1 + num_fw + num_bw)
        
        # ── Input BatchNorm (no affine params, matches original) ─────────
        # Note: original Chainer BN uses running stats in eval mode.
        # We use default track_running_stats=True to match this behaviour.
        self.in_BN = nn.BatchNorm1d(in_dim, affine=False)
        
        # ── Encoder ──────────────────────────────────────────────────────
        encoder_layers = []
        encoder_layers.append(nn.Linear(concat_dim, hid_dim))
        for _ in range(num_hid):
            encoder_layers.append(nn.Linear(hid_dim, hid_dim))
        encoder_layers.append(nn.Linear(hid_dim, z_dim))
        self.encoder = nn.ModuleList(encoder_layers)
        
        # ── Decoder ──────────────────────────────────────────────────────
        decoder_layers = []
        decoder_layers.append(nn.Linear(z_dim, hid_dim))
        for _ in range(num_hid):
            decoder_layers.append(nn.Linear(hid_dim, hid_dim))
        decoder_layers.append(nn.Linear(hid_dim, concat_dim))
        self.decoder = nn.ModuleList(decoder_layers)
        
        # ── Weight initialisation (GlorotNormal ≈ xavier_normal) ─────────
        self._init_weights()
        
        # Report
        num_params = sum(p.numel() for p in self.parameters())
        print(f'Number of parameters: {num_params}')
    
    def _init_weights(self):
        """Initialise Linear layers with GlorotNormal (xavier_normal)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x_data):
        """
        Forward pass.
        
        Args:
            x_data: Mel-filterbank features, shape (NumFB, T)
        
        Returns:
            score: Per-frame anomaly scores, shape (T,)
            x:     Input after frame concatenation, shape (T, concat_dim)
            y:     Reconstructed output, shape (T, concat_dim)
        """
        # ── Input BN (transpose → BN → transpose back) ───────────────────
        # x_data: (NumFB, T) → transpose → (T, NumFB) → BN → (T, NumFB)
        x = x_data.t()                              # (T, NumFB)
        x = self.in_BN(x)                           # (T, NumFB)
        
        # ── Frame concatenation (context window) ─────────────────────────
        # Need (K, O) where K=T, O=NumFB. frame_concat expects (K, O).
        # But frame_concat operates on (K, O) → (K, O*(1+bw+fw))
        # in_BN output is (T, NumFB), but frame_concat uses K,T dims differently.
        # Let's transpose to match: h is (O, K) → frame_concat → (O, K*(1+bw+fw))
        # Actually, looking at the original: h has shape (K, O) where K = time frames.
        # in_BN returns (T, NumFB) = (K, O). This is the correct shape.
        x = frame_concat(x, self.num_bw, self.num_fw)   # (T, concat_dim)
        h = x
        
        # ── Encoder ──────────────────────────────────────────────────────
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if i < len(self.encoder) - 1:          # ReLU on all but last encoder layer
                h = F.relu(h)
        z = F.relu(h)                               # Bottleneck with ReLU
        
        # ── Decoder ──────────────────────────────────────────────────────
        h = z
        for i, layer in enumerate(self.decoder):
            h = layer(h)
            if i < len(self.decoder) - 1:          # ReLU on all but last decoder layer
                h = F.relu(h)
        y = h                                       # Reconstruction (T, concat_dim)
        
        # ── Anomaly score ────────────────────────────────────────────────
        # Per-frame MSE, averaged over the concatenated feature dims
        score = torch.sum((x - y) ** 2, dim=1) / (1 + self.num_fw + self.num_bw)
        
        return score, x, y


# ── Debug ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick sanity check
    model = FCN_AE(in_dim=64, hid_dim=512, z_dim=128, num_hid=4, num_fw=10, num_bw=10)
    dummy_input = torch.randn(64, 100)  # (NumFB=64, T=100)
    score, x, y = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print(f"Score shape: {score.shape}")
    print(f"Mean score: {score.mean().item():.6f}")
