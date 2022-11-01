import torch


def generate(dim=1024):
    v = torch.randn(dim)
    return v / torch.linalg.norm(v)


def make_unitary(v):
    fft_val = torch.fft.fft(v)
    fft_imag = fft_val.imag
    fft_real = fft_val.real
    fft_norms = torch.sqrt(fft_imag ** 2 + fft_real ** 2)
    invalid = fft_norms <= 0.0
    fft_val[invalid] = 1.0
    fft_norms[invalid] = 1.0
    fft_unit = fft_val / fft_norms
    return (torch.fft.ifft(fft_unit, n=len(v))).real


def bind(v1, v2):
    out = torch.fft.irfft(torch.fft.rfft(v1) * torch.fft.rfft(v2), dim=-1)
    return out
