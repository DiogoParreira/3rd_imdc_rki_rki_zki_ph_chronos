"""
PyTorch blocks for an LSTM + Bayesian output layer.

This file does not train a model and does not create submissions.
It only contains reusable components for experimentation.

Architecture idea:

    sequence of state-week features
        -> LSTM encoder
        -> last hidden state
        -> Bayesian linear layer
        -> mean and log-variance
        -> Monte Carlo predictive samples
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def softplus_sigma(raw_sigma: torch.Tensor, min_sigma: float = 1e-4) -> torch.Tensor:
    """
    Convert an unconstrained raw sigma value into a strictly positive sigma.

    Parameters
    ----------
    raw_sigma:
        Raw model output.
    min_sigma:
        Small positive floor to avoid numerical problems.

    Returns
    -------
    torch.Tensor
        Positive standard deviation.
    """
    return F.softplus(raw_sigma) + min_sigma


def gaussian_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Gaussian negative log likelihood.

    This is useful when the model predicts both:
        - mu: expected value
        - sigma: predictive uncertainty
    Otherwise, we could potentially use MSE if we want to focus on mu. 

    For dengue/mosquito count prediction, we will usually use this on the
    log1p-transformed target:

        y_log = log1p(cases)

    Parameters
    ----------
    y:
        Observed target.
    mu:
        Predicted mean.
    sigma:
        Predicted standard deviation.
    reduction:
        "mean", "sum", or "none".

    Returns
    -------
    torch.Tensor
        Negative log likelihood.
    """
    var = sigma.pow(2)

    nll = 0.5 * (
        torch.log(2.0 * torch.pi * var)
        + (y - mu).pow(2) / var
    )

    if reduction == "mean":
        return nll.mean()

    if reduction == "sum":
        return nll.sum()

    if reduction == "none":
        return nll

    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


class BayesianLinear(nn.Module):
    """
    Variational Bayesian linear layer.

    Instead of learning one deterministic weight matrix W, this layer learns a
    distribution over weights:

        W ~ Normal(weight_mu, weight_sigma)

    During training and Monte Carlo prediction, weights are sampled using the
    reparameterization trick:

        W = weight_mu + weight_sigma * epsilon

    The layer also computes the KL divergence between the variational posterior
    q(W) and a standard Normal prior p(W).

    This gives us epistemic uncertainty from parameter uncertainty.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        posterior_rho_init: float = -5.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Mean parameters
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(
            torch.empty(out_features)
        )

        # Rho parameters. Sigma is obtained through softplus(rho).
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias_rho = nn.Parameter(
            torch.empty(out_features)
        )

        self.reset_parameters(posterior_rho_init)

    def reset_parameters(self, posterior_rho_init: float) -> None:
        """
        Initialize means like a normal Linear layer and initialize posterior
        uncertainty to be small.
        """
        bound = 1.0 / math.sqrt(self.in_features)

        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

        nn.init.constant_(self.weight_rho, posterior_rho_init)
        nn.init.constant_(self.bias_rho, posterior_rho_init)

    @property
    def weight_sigma(self) -> torch.Tensor:
        return F.softplus(self.weight_rho)

    @property
    def bias_sigma(self) -> torch.Tensor:
        return F.softplus(self.bias_rho)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape [batch, in_features].
        sample:
            If True, sample weights from the variational posterior.
            If False, use posterior mean weights.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch, out_features].
        """
        if sample:
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)

            weight = self.weight_mu + self.weight_sigma * weight_eps
            bias = self.bias_mu + self.bias_sigma * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """
        KL divergence between q(W) and p(W).

        q(W) = Normal(mu, sigma)
        p(W) = Normal(0, prior_sigma)

        Returns
        -------
        torch.Tensor
            Scalar KL divergence.
        """
        weight_kl = self._normal_kl(
            self.weight_mu,
            self.weight_sigma,
            self.prior_sigma,
        )

        bias_kl = self._normal_kl(
            self.bias_mu,
            self.bias_sigma,
            self.prior_sigma,
        )

        return weight_kl + bias_kl

    @staticmethod
    def _normal_kl(
        posterior_mu: torch.Tensor,
        posterior_sigma: torch.Tensor,
        prior_sigma: float,
    ) -> torch.Tensor:
        """
        KL divergence KL(q || p) where:

            q = Normal(posterior_mu, posterior_sigma)
            p = Normal(0, prior_sigma)
        """
        prior_var = prior_sigma ** 2
        posterior_var = posterior_sigma.pow(2)

        kl = torch.log(torch.tensor(prior_sigma, device=posterior_mu.device))
        kl = kl - torch.log(posterior_sigma)
        kl = kl + (posterior_var + posterior_mu.pow(2)) / (2.0 * prior_var)
        kl = kl - 0.5

        return kl.sum()


class LSTMBayesianHead(nn.Module):
    """
    LSTM encoder with Bayesian output layer.

    This is the architecture to experiment with first:

        X sequence
            -> LSTM
            -> last hidden state
            -> BayesianLinear
            -> mu and raw_sigma

    The model outputs a predictive mean and standard deviation.

    Input shape:
        X: [batch, lookback, n_features]

    Output:
        mu:    [batch]
        sigma: [batch]
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        prior_sigma: float = 1.0,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        # Output 2 values:
        #   1. mu
        #   2. raw_sigma
        self.head = BayesianLinear(
            in_features=hidden_size,
            out_features=2,
            prior_sigma=prior_sigma,
        )

    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input sequence with shape [batch, lookback, n_features].
        sample:
            Whether to sample Bayesian head weights.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            mu and sigma, each with shape [batch].
        """
        lstm_out, _ = self.lstm(x)

        # Last time step representation
        h_last = lstm_out[:, -1, :]

        out = self.head(h_last, sample=sample)

        mu = out[:, 0]
        raw_sigma = out[:, 1]

        sigma = softplus_sigma(raw_sigma)

        return mu, sigma

    def kl_divergence(self) -> torch.Tensor:
        """
        KL term from the Bayesian output layer.
        """
        return self.head.kl_divergence()


@dataclass
class BayesianLossOutput:
    total_loss: torch.Tensor
    nll: torch.Tensor
    kl: torch.Tensor


def bayesian_lstm_loss(
    model: LSTMBayesianHead,
    x: torch.Tensor,
    y: torch.Tensor,
    beta: float = 1e-4,
) -> BayesianLossOutput:
    """
    Compute Bayesian LSTM loss.

    total_loss = Gaussian NLL + beta * KL

    Parameters
    ----------
    model:
        LSTMBayesianHead.
    x:
        Input tensor [batch, lookback, n_features].
    y:
        Target tensor [batch].
    beta:
        Weight on the KL divergence.

    Returns
    -------
    BayesianLossOutput
        total loss, NLL component, and KL component.
    """
    mu, sigma = model(x, sample=True)

    nll = gaussian_nll(y, mu, sigma, reduction="mean")
    kl = model.kl_divergence()

    total = nll + beta * kl

    return BayesianLossOutput(
        total_loss=total,
        nll=nll.detach(),
        kl=kl.detach(),
    )


@torch.no_grad()
def mc_predict_log_scale(
    model: LSTMBayesianHead,
    x: torch.Tensor,
    n_samples: int = 500,
) -> torch.Tensor:
    """
    Draw Monte Carlo predictive samples on the model scale.

    If the model is trained on log1p(cases), the output samples are on the
    log1p scale.

    Parameters
    ----------
    model:
        Trained LSTMBayesianHead.
    x:
        Input tensor [batch, lookback, n_features].
    n_samples:
        Number of Monte Carlo samples.

    Returns
    -------
    torch.Tensor
        Samples with shape [n_samples, batch].
    """
    model.eval()

    samples = []

    for _ in range(n_samples):
        mu, sigma = model(x, sample=True)

        eps = torch.randn_like(mu)
        sample = mu + sigma * eps

        samples.append(sample)

    return torch.stack(samples, dim=0)


@torch.no_grad()
def mc_predict_cases(
    model: LSTMBayesianHead,
    x: torch.Tensor,
    n_samples: int = 500,
) -> torch.Tensor:
    """
    Draw Monte Carlo samples and convert from log1p scale back to cases.

    Returns
    -------
    torch.Tensor
        Non-negative samples with shape [n_samples, batch].
    """
    log_samples = mc_predict_log_scale(
        model=model,
        x=x,
        n_samples=n_samples,
    )

    case_samples = torch.expm1(log_samples)
    case_samples = torch.clamp(case_samples, min=0.0)

    return case_samples


def prediction_intervals_from_samples(samples: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Convert Monte Carlo samples into point predictions and intervals.

    Parameters
    ----------
    samples:
        Tensor with shape [n_samples, batch].

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with pred/lower/upper tensors.
    """
    q = torch.quantile(
        samples,
        torch.tensor(
            [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975],
            device=samples.device,
        ),
        dim=0,
    )

    return {
        "lower_95": q[0],
        "lower_90": q[1],
        "lower_80": q[2],
        "lower_50": q[3],
        "pred": q[4],
        "upper_50": q[5],
        "upper_80": q[6],
        "upper_90": q[7],
        "upper_95": q[8],
    }