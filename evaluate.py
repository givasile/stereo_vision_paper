import torch


def mean_error(pred, gtL, maskL):
    assert gtL.size() == pred.size()
    assert gtL.size() == maskL.size()
    valid_pred = torch.masked_select(pred, maskL)
    valid_gt = torch.masked_select(gtL, maskL)
    m_e = torch.mean(valid_pred - valid_gt).item()
    return m_e

def std_error(pred, gtL, maskL):
    assert gtL.size() == pred.size()
    assert gtL.size() == maskL.size()
    valid_pred = torch.masked_select(pred, maskL)
    valid_gt = torch.masked_select(gtL, maskL)
    std_e = torch.std(valid_pred - valid_gt).item()
    return std_e

def mean_absolute_error(pred, gtL, maskL):
    assert gtL.size() == pred.size()
    assert gtL.size() == maskL.size()
    valid_pred = torch.masked_select(pred, maskL)
    valid_gt = torch.masked_select(gtL, maskL)
    m_ae = torch.mean(torch.abs(valid_pred - valid_gt)).item()
    return m_ae

def std_absolute_error(pred, gtL, maskL):
    assert gtL.size() == pred.size()
    assert gtL.size() == maskL.size()
    valid_pred = torch.masked_select(pred, maskL)
    valid_gt = torch.masked_select(gtL, maskL)
    std_ae = torch.std(torch.abs(valid_pred - valid_gt)).item()
    return std_ae

def image_absolute_error(pred, gtL, maskL):
    assert gtL.size() == pred.size()
    assert gtL.size() == maskL.size()
    im = torch.abs(pred - gtL)
    mask_inv = maskL.clone()
    mask_inv[mask_inv == 1] = 2
    mask_inv[mask_inv == 0] = 1
    mask_inv[mask_inv == 2] = 0
    im[mask_inv] = 0
    return im


def percentage_over_limit(pred, gtL, maskL, threshold):
    assert gtL.size() == pred.size()
    assert gtL.size() == maskL.size()
    valid_pred = torch.masked_select(pred, maskL)
    valid_gt = torch.masked_select(gtL, maskL)
    error = torch.abs(valid_pred - valid_gt)
    error[error <= threshold] = 0
    error[error > threshold] = 1
    n_error = torch.sum(error).item()
    return n_error/(valid_gt.size()[0])*100


def image_percentage_over_limit(pred, gtL, maskL, threshold):
    assert gtL.size() == pred.size()
    assert gtL.size() == maskL.size()
    mask_inv = maskL.clone()
    mask_inv[mask_inv == 1] = 2
    mask_inv[mask_inv == 0] = 1
    mask_inv[mask_inv == 2] = 0
    error = torch.abs(pred - gtL)
    error[error <= threshold] = 0
    error[error > threshold] = 1
    error[mask_inv] = 0.5
    return error
