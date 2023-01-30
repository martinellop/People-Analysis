import torch

def L1_distance(x:torch.Tensor, y: torch.Tensor):
    """
    Args:
        x: must be shaped [k1,d] 
        y: must be shaped [k2,d]

    Returns:
        torch.Tensor shaped [k1,k2]
    """
    assert x.dim() == 2 and y.dim() == 2
    assert x.shape[1] == y.shape[1]

    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.sum(torch.abs(x-y), -1)

def L2_distance(x:torch.Tensor, y: torch.Tensor, squared_distance:bool=False):
    """
    Args:
        x: must be shaped [k1,d] 
        y: must be shaped [k2,d]
        squared_distance: if set to true, final square root is not computed.

    Returns:
        torch.Tensor shaped [k1,k2]
    """
    assert x.dim() == 2 and y.dim() == 2
    assert x.shape[1] == y.shape[1]

    x = x.reshape(x.shape[0], 1, x.shape[1])
    y = y.reshape(1, y.shape[0], y.shape[1])
    if squared_distance:
        return torch.sum(torch.pow(x-y,2), -1)
    else:
        return torch.sqrt(torch.sum(torch.pow(x-y,2), -1).clamp(min=1e-12))

def Mahalanobis_distance(x:torch.Tensor, y:torch.Tensor, C:torch.Tensor):
    """
    Args:
        x: must be shaped [k1,d] 
        y: must be shaped [k2,d]
        C: is the inverse of the Covariance Matrix (d x d)

    Returns:
        torch.Tensor shaped [k1,k2]
    """
    assert x.dim() == 2 and y.dim() == 2 and C.dim() == 2
    assert x.shape[1] == y.shape[1] == C.shape[0] == C.shape[1]

    assert False , "THIS CODE MUST BE REFACTORED"   #to be checked!
    x = x.unsqueeze(0)
    y = y.unsqueeze(1)
    diff = (x - y).to(dtype=torch.float)            # (n x d)
    mat1 = torch.matmul(diff, C)                    # (n x d)
    dist = torch.matmul(mat1,diff.t())              # (n x n)
    dist = dist.diag()                              # (n x 1)
    return torch.sqrt(dist)

def Cosine_distance(x: torch.Tensor, y:torch.Tensor):
    """
    Args:
        x: must be shaped [k1,d] 
        y: must be shaped [k2,d]

    Returns:
        torch.Tensor shaped [k1,k2]
    """
    assert x.dim() == 2 and y.dim() == 2
    assert x.shape[1] == y.shape[1]

    x = x.reshape(x.shape[0], 1, x.shape[1])
    y = y.reshape(1, y.shape[0], y.shape[1])

    den = x.norm(dim=-1) * y.norm(dim=-1)
    similarities = (x * y).sum(-1) / den 
    return 1 - similarities