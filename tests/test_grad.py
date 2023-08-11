import torch
from torch.autograd.gradcheck import gradcheck

import dcnv3


def test_dcnv3(dtype=torch.float64,
               device='cuda'):
    torch.manual_seed(12)
    device = torch.device(device)

    input = torch.randn(1, 10, 10, 4,
                        dtype=dtype, device=device, requires_grad=True)
    kernel_h, kernel_w = 3, 3
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dilation_h, dilation_w = 1, 1
    group = 2
    group_channels = input.size(-1) // group
    offset_scale = 1.0
    im2col_step = 256
    remove_center = 0

    offset = torch.randn(*input.shape[:-1], group * 2 * (kernel_h * kernel_w - remove_center),
                         dtype=dtype, device=device, requires_grad=True)
    mask = torch.rand(*input.shape[:-1], group * (kernel_h * kernel_w - remove_center),
                      dtype=dtype, device=device, requires_grad=True)

    optim = torch.optim.Optimizer([input, offset, mask], {})

    c_output = dcnv3.ops.dcnv3(input, offset, mask,
                               kernel_h, kernel_w,
                               stride_h, stride_w,
                               pad_h, pad_w,
                               dilation_h, dilation_w,
                               group, group_channels,
                               offset_scale,
                               im2col_step,
                               remove_center)
    c_output.sum().backward()
    c_input_grad = input.grad.clone()
    c_offset_grad = offset.grad.clone()
    c_mask_grad = mask.grad.clone()
    optim.zero_grad()
    print(c_output)

    grad_ok = gradcheck(
        lambda inp, off, msk: dcnv3.ops.dcnv3(inp, off, msk,
                                              kernel_h, kernel_w,
                                              stride_h, stride_w,
                                              pad_h, pad_w,
                                              dilation_h, dilation_w,
                                              group, group_channels,
                                              offset_scale,
                                              im2col_step, remove_center),
        (input, offset, mask), nondet_tol=1e-5)
    print('grad_check:', grad_ok)


if __name__ == '__main__':
    test_dcnv3()
