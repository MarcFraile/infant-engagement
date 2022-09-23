import torch


class LinearDataset:
    """
    Dummy test dataset. Separates space along a hyperplane, and classifies points as 1 or 0 depending on which side they belong to.
    """

    def __init__(self, dims: int, samples_per_epoch: int):
        self.dims = dims
        self.samples_per_epoch = samples_per_epoch

        self.bias = torch.rand(1)

        self.normal = torch.rand(dims)
        N = self.normal.norm()
        self.normal /= N

    def __len__(self):
        return self.samples_per_epoch

    def _sample(self, n):
        x     = 10 * torch.randn(n, self.dims)
        score = (x * self.normal[None, :]).sum(dim=1) + self.bias
        # score = self.normal.dot(x)
        y     = (score > 0).float()

        return (x.squeeze(dim=0), y.squeeze(dim=0))

    def __getitem__(self, _idx):
        return self._sample(1)

    def all(self):
        return self._sample(self.samples_per_epoch)
