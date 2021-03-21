import torch
import torch.distributions
import torch.nn as nn


class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.crit(pred, true).sum() / pred.shape[1]


class OneDimensionalKLD(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1),
            dim=0,
        )
        return kld_loss * self.beta


class KLDTwoGaussians(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        mu_1: torch.Tensor,
        logvar_1: torch.Tensor,
        mu_2: torch.Tensor,
        logvar_2: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = mu_1.shape[1]

        sigma1 = logvar_1.mul(0.5).exp()
        sigma2 = logvar_2.mul(0.5).exp()

        q = torch.distributions.Normal(mu_1, sigma1)
        p = torch.distributions.Normal(mu_2, sigma2)

        kld = torch.distributions.kl_divergence(q, p)

        return kld.sum() / batch_size * self.beta


class KLD(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        batch_size = mu.shape[1]

        sigma = logvar.mul(0.5).exp()

        q = torch.distributions.Normal(mu, sigma)
        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(mu)
        )

        kld = torch.distributions.kl_divergence(q, p)

        return kld.sum() / batch_size * self.beta
