from typing import Tuple

from torch import Tensor


def _cuda_version() -> int: ...


def dcnv3_forward(input: Tensor, offset: Tensor, mask: Tensor,
                  kernel_h: int, kernel_w: int,
                  stride_h: int, stride_w: int,
                  pad_h: int, pad_w: int,
                  dilation_h: int, dilation_w: int,
                  group: int, group_channels: int,
                  offset_scale: float,
                  im2col_step: int, remove_center: int) -> Tensor: ...


def dcnv3_backward(input: Tensor, offset: Tensor, mask: Tensor,
                   kernel_h: int, kernel_w: int,
                   stride_h: int, stride_w: int,
                   pad_h: int, pad_w: int,
                   dilation_h: int, dilation_w: int,
                   group: int, group_channels: int,
                   offset_scale: float, grad_output: Tensor,
                   im2col_step: int, remove_center: int) -> Tuple[Tensor, Tensor, Tensor]: ...
