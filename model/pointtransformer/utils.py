import torch


def list2tensor2d(list_p):
    """
    list_p: a size-b list storing (sample, c) tensors
    return: Tensor of shape (sum(sample), c)
    """

    if len(list_p) == 0:
        return torch.empty(0)
    
    target = torch.device('cuda:0')
    list_p = [t.to(target) for t in list_p]
    n_p = torch.cat(list_p, dim=0)  # (sum(sample), c)
    return n_p.clone()


def list2tensor3d(list_p):
    """
    list_p: 长度为 B 的 list
            每个元素是一个 list，存若干 (c, d) 的 Tensor
    return:
        n_p: (sum_all c, d)
    """
    flat_list = []

    for t in list_p:   
        if not t: continue
        target = torch.device('cuda:0')
        t = [t1.to(target) for t1 in t]
        flat = torch.cat(t, dim=0)     # (sum_j c_j, d)
        flat_list.append(flat)

    n_p = torch.cat(flat_list, dim=0)  # (total, d)
    return n_p.clone()

def list2offset(list_p):
    """
    list_p: 长度为 B 的 list
            每个元素是一个 list，存若干 (c, d) 的 Tensor
    return:
        offsets: (B+1,) 累积行数
    """
    offsets = []
    total = 0

    for t in list_p:   
        if not t: continue
        target = torch.device('cuda:0')
        t = [t1.to(target) for t1 in t]
        flat = torch.cat(t, dim=0)     # (sum_j c_j, d)
        total += flat.shape[0]
        offsets.append(total)

    offsets = torch.tensor(offsets, dtype=torch.int32)
    return offsets

def list2CandidateOffset(list_p):
    """
    list_p: 长度为 B 的 list
            每个元素是一个 list，存若干 (c, d) 的 Tensor
    return:
        offsets: (B+1,) 每个采样点均对应一个候选域区间
    """
    offsets = []
    total = 0

    for t in list_p:   
        if not t: continue
        for m in t:
            total += m.shape[0]
            offsets.append(total)

    offsets = torch.tensor(offsets, dtype=torch.int32)
    return offsets