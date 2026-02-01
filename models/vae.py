"""
VAE Model Architectures for BHAI Paper

Unsupervised VAE (v2.6.7) and Semi-Supervised VAE (v2.14)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler


class DistributionAwareScaler:
    """
    Scaler that applies log transforms to specific columns before StandardScaler.
    
    - Columns 1, 2 (Mag susc, NGR): signed log transform
    - Columns 3, 4, 5 (R, G, B): log1p transform
    - All columns: StandardScaler
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.signed_log_indices = [1, 2]
        self.log_indices = [3, 4, 5]
    
    def signed_log_transform(self, x):
        return np.sign(x) * np.log1p(np.abs(x))
    
    def inverse_signed_log_transform(self, x):
        return np.sign(x) * (np.exp(np.abs(x)) - 1)
    
    def fit(self, X):
        X_t = self._transform_logs(X)
        self.scaler.fit(X_t)
        return self
    
    def transform(self, X):
        X_t = self._transform_logs(X)
        return self.scaler.transform(X_t)
    
    def fit_transform(self, X):
        X_t = self._transform_logs(X)
        return self.scaler.fit_transform(X_t)
    
    def inverse_transform(self, X):
        X_t = self.scaler.inverse_transform(X)
        return self._inverse_transform_logs(X_t)
    
    def _transform_logs(self, X):
        X_t = X.copy()
        for idx in self.signed_log_indices:
            X_t[:, idx] = self.signed_log_transform(X_t[:, idx])
        for idx in self.log_indices:
            X_t[:, idx] = np.log1p(X_t[:, idx])
        return X_t
    
    def _inverse_transform_logs(self, X):
        X_t = X.copy()
        for idx in self.signed_log_indices:
            X_t[:, idx] = self.inverse_signed_log_transform(X_t[:, idx])
        for idx in self.log_indices:
            X_t[:, idx] = np.expm1(X_t[:, idx])
        return X_t


class VAE(nn.Module):
    """
    Unsupervised Variational Autoencoder (v2.6.7 architecture).
    
    Architecture:
        Encoder: input(6) -> 64 -> 32 -> latent(10)
        Decoder: latent(10) -> 32 -> 64 -> output(6)
    
    Uses BatchNorm after each hidden layer.
    """
    def __init__(self, input_dim=6, latent_dim=10, hidden_dims=[64, 32]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[0], input_dim)
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction."""
        h = self.decoder(z)
        return self.fc_out(h)
    
    def forward(self, x):
        """Forward pass: encode -> sample -> decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def get_embeddings(self, x):
        """Get deterministic embeddings (mean of latent distribution)."""
        mu, _ = self.encode(x)
        return mu
    
    def loss_function(self, recon, x, mu, logvar, beta=1.0):
        """
        VAE loss = Reconstruction loss + beta * KL divergence
        """
        recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


class SemiSupervisedVAE(nn.Module):
    """
    Semi-Supervised Variational Autoencoder (v2.14 architecture).
    
    Same encoder/decoder as unsupervised VAE, plus a classification head
    that predicts lithology from the latent representation.
    
    Architecture:
        Encoder: input(6) -> 64 -> 32 -> latent(10)
        Decoder: latent(10) -> 32 -> 64 -> output(6)
        Classifier: latent(10) -> 64 -> n_classes
    """
    def __init__(self, input_dim=6, latent_dim=10, hidden_dims=[64, 32], n_classes=139):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        
        # Encoder
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[0], input_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction."""
        h = self.decoder(z)
        return self.fc_out(h)
    
    def classify(self, z):
        """Classify from latent representation."""
        return self.classifier(z)
    
    def forward(self, x):
        """Forward pass: encode -> sample -> decode + classify."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classify(z)
        return recon, mu, logvar, logits
    
    def get_embeddings(self, x):
        """Get deterministic embeddings (mean of latent distribution)."""
        mu, _ = self.encode(x)
        return mu
    
    def loss_function(self, recon, x, mu, logvar, logits, labels, 
                     beta=1.0, alpha=0.1):
        """
        Semi-supervised VAE loss:
            L = Recon + beta*KL + alpha*Classification
        
        Parameters
        ----------
        alpha : float
            Weight for classification loss (default 0.1)
        """
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Classification loss
        class_loss = nn.functional.cross_entropy(logits, labels, reduction='sum')
        
        total_loss = recon_loss + beta * kl_loss + alpha * class_loss
        return total_loss, recon_loss, kl_loss, class_loss


# Feature column names
FEATURE_COLS = [
    'Bulk density (GRA)',
    'Magnetic susceptibility (instr. units)',
    'NGR total counts (cps)',
    'R', 'G', 'B'
]
