from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from dcnv3._C import dcnv3_forward, dcnv3_backward

__all__ = [
    'DCNv3Function',
    'dcnv3',
]


class DCNv3Function(Function):
    @staticmethod
    @custom_fwd
    def forward(
            ctx, input, offset, mask,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w,
            group, group_channels, offset_scale, im2col_step, remove_center):
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center

        output = dcnv3_forward(input, offset, mask, kernel_h,
                               kernel_w, stride_h, stride_w, pad_h,
                               pad_w, dilation_h, dilation_w, group,
                               group_channels, offset_scale, im2col_step,
                               remove_center)
        ctx.save_for_backward(input, offset, mask)

        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset, mask = ctx.saved_tensors

        grad_input, grad_offset, grad_mask = dcnv3_backward(
            input, offset, mask, ctx.kernel_h,
            ctx.kernel_w, ctx.stride_h, ctx.stride_w, ctx.pad_h,
            ctx.pad_w, ctx.dilation_h, ctx.dilation_w, ctx.group,
            ctx.group_channels, ctx.offset_scale, grad_output.contiguous(), ctx.im2col_step,
            ctx.remove_center)

        return grad_input, grad_offset, grad_mask, \
            None, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, input, offset, mask, kernel_h, kernel_w, stride_h,
                 stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                 group_channels, offset_scale, im2col_step, remove_center):
        """Symbolic function for mmdeploy::DCNv3.

        Returns:
            DCNv3 op for onnx.
        """
        return g.op(
            'mmdeploy::TRTDCNv3',
            input,
            offset,
            mask,
            kernel_h_i=int(kernel_h),
            kernel_w_i=int(kernel_w),
            stride_h_i=int(stride_h),
            stride_w_i=int(stride_w),
            pad_h_i=int(pad_h),
            pad_w_i=int(pad_w),
            dilation_h_i=int(dilation_h),
            dilation_w_i=int(dilation_w),
            group_i=int(group),
            group_channels_i=int(group_channels),
            offset_scale_f=float(offset_scale),
            im2col_step_i=int(im2col_step),
            remove_center=int(remove_center),
        )


def dcnv3(
        input: Tensor, offset: Tensor, mask: Tensor,
        kernel_h: int, kernel_w: int,
        stride_h: int, stride_w: int,
        pad_h: int, pad_w: int,
        dilation_h: int, dilation_w: int,
        group: int, group_channels: int,
        offset_scale: float,
        im2col_step: int, remove_center: int):
    return DCNv3Function.apply(input, offset, mask, kernel_h,
                               kernel_w, stride_h, stride_w, pad_h,
                               pad_w, dilation_h, dilation_w, group,
                               group_channels, offset_scale, im2col_step, remove_center)
