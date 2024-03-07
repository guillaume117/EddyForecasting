import torch


def MSEWeightedLoss(input, target, weights=None):
    if weights is None:
        weights = torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.4, 0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4], device=input.device)

    squared_diff = (input - target)**2
    weighted_squared_diff = squared_diff * weights.view(1, -1, 1, 1)
    loss = torch.sum(weighted_squared_diff, dim=1) 
    return torch.mean(loss)




