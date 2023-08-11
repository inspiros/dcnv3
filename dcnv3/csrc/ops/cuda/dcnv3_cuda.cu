/*!
**************************************************************************************************
* Int64_ternImage
* Copyright (c) 2022 OpenGVLab
* Licensed under The MIT License [see LICENSE for details]
**************************************************************************************************
* Modified from
* https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
*
* Modified from
* https://github.com/OpenGVLab/Int64_ternImage/blob/master/classification/ops_dcnv3/src
**************************************************************************************************
*/

#include "dcnv3_im2col_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

namespace dcnv3 {
    namespace ops {
        namespace cuda {
            at::Tensor dcnv3_forward(
                    const at::Tensor &input, const at::Tensor &offset,
                    const at::Tensor &mask, const int64_t kernel_h,
                    const int64_t kernel_w, const int64_t stride_h,
                    const int64_t stride_w, const int64_t pad_h,
                    const int64_t pad_w, const int64_t dilation_h,
                    const int64_t dilation_w, const int64_t group,
                    const int64_t group_channels,
                    const double_t offset_scale, const int64_t im2col_step, const int64_t remove_center) {
                TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous")
                TORCH_CHECK(offset.is_contiguous(), "offset tensor has to be contiguous")
                TORCH_CHECK(mask.is_contiguous(), "mask tensor has to be contiguous")

                TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor")
                TORCH_CHECK(offset.is_cuda(), "offset must be a CUDA tensor")
                TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor")

                const int64_t batch = input.size(0);
                const int64_t height_in = input.size(1);
                const int64_t width_in = input.size(2);
                const int64_t channels = input.size(3);
                const int64_t height_out =
                        (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
                const int64_t width_out =
                        (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
                const int64_t im2col_step_ = std::min(batch, im2col_step);

                TORCH_CHECK(batch % im2col_step_ == 0,
                            "batch(",
                            batch,
                            ") must divide im2col_step(",
                            im2col_step_,
                            ")")
                TORCH_CHECK(
                        channels == (group * group_channels),
                        "Input channels and group times group channels wont match: (",
                        channels,
                        " vs ",
                        group * group_channels,
                        ")")

                auto output =
                        at::zeros({batch, height_out, width_out, group * group_channels},
                                  input.options());

                const int64_t batch_n = im2col_step_;
                auto output_n = output.view({batch / batch_n, batch_n, height_out,
                                             width_out, group * group_channels});
                auto per_input_size = height_in * width_in * group * group_channels;
                auto per_offset_size =
                        height_out * width_out * group * (kernel_h * kernel_w - remove_center) * 2;
                auto per_mask_size = height_out * width_out * group * (kernel_h * kernel_w - remove_center);
                for (int64_t n = 0; n < batch / im2col_step_; ++n) {
                    auto columns = output_n.select(0, n);
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                            input.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
                        detail::dcnv3_im2col_cuda(
                                at::cuda::getCurrentCUDAStream(),
                                input.data_ptr<scalar_t>() + n * im2col_step_ * per_input_size,
                                offset.data_ptr<scalar_t>() +
                                n * im2col_step_ * per_offset_size,
                                mask.data_ptr<scalar_t>() + n * im2col_step_ * per_mask_size,
                                columns.data_ptr<scalar_t>(), kernel_h, kernel_w, stride_h,
                                stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                                group_channels, batch_n, height_in, width_in, height_out,
                                width_out, offset_scale, remove_center);
                    }));
                }

                return output;
            }

            std::tuple<at::Tensor, at::Tensor, at::Tensor> dcnv3_backward(
                    const at::Tensor &input, const at::Tensor &offset,
                    const at::Tensor &mask, const int64_t kernel_h,
                    const int64_t kernel_w, const int64_t stride_h, const int64_t stride_w,
                    const int64_t pad_h, const int64_t pad_w, const int64_t dilation_h,
                    const int64_t dilation_w, const int64_t group,
                    const int64_t group_channels, const double_t offset_scale,
                    const at::Tensor &grad_output, const int64_t im2col_step, const int64_t remove_center) {
                TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous")
                TORCH_CHECK(offset.is_contiguous(), "offset tensor has to be contiguous")
                TORCH_CHECK(mask.is_contiguous(), "mask tensor has to be contiguous")
                TORCH_CHECK(grad_output.is_contiguous(), "grad_output tensor has to be contiguous")

                TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor")
                TORCH_CHECK(offset.is_cuda(), "offset must be a CUDA tensor")
                TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor")
                TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor")

                const int64_t batch = input.size(0);
                const int64_t height_in = input.size(1);
                const int64_t width_in = input.size(2);
                const int64_t channels = input.size(3);
                const int64_t height_out =
                        (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
                const int64_t width_out =
                        (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
                const int64_t im2col_step_ = std::min(batch, im2col_step);

                TORCH_CHECK(batch % im2col_step_ == 0,
                            "batch(",
                            batch,
                            ") must divide im2col_step(",
                            im2col_step_,
                            ")")
                TORCH_CHECK(
                        channels == (group * group_channels),
                        "Input channels and group times group channels wont match: (",
                        channels,
                        " vs ",
                        group * group_channels,
                        ")")

                auto dtype = input.dtype();
                if (dtype == at::kHalf) {
                    dtype = at::kFloat;
                }

                auto grad_input = at::zeros_like(input, dtype);
                auto grad_offset = at::zeros_like(offset, dtype);
                auto grad_mask = at::zeros_like(mask, dtype);

                const int64_t batch_n = im2col_step_;
                auto per_input_size = height_in * width_in * group * group_channels;
                auto per_offset_size =
                        height_out * width_out * group * (kernel_h * kernel_w - remove_center) * 2;
                auto per_mask_size = height_out * width_out * group * (kernel_h * kernel_w - remove_center);
                auto grad_output_n =
                        grad_output.view({batch / im2col_step_, batch_n, height_out * width_out,
                                          group, group_channels});

                for (int64_t n = 0; n < batch / im2col_step_; ++n) {
                    auto grad_output_g = grad_output_n.select(0, n);
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                            input.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
                        detail::dcnv3_col2im_cuda(
                                at::cuda::getCurrentCUDAStream(),
                                grad_output_g.data_ptr<scalar_t>(),
                                input.data_ptr<scalar_t>() + n * im2col_step_ * per_input_size,
                                offset.data_ptr<scalar_t>() +
                                n * im2col_step_ * per_offset_size,
                                mask.data_ptr<scalar_t>() + n * im2col_step_ * per_mask_size,
                                kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                                dilation_h, dilation_w, group, group_channels, batch_n,
                                height_in, width_in, height_out, width_out, offset_scale, remove_center,
                                grad_input.data_ptr<opmath_t>() +
                                n * im2col_step_ * per_input_size,
                                grad_offset.data_ptr<opmath_t>() +
                                n * im2col_step_ * per_offset_size,
                                grad_mask.data_ptr<opmath_t>() +
                                n * im2col_step_ * per_mask_size);
                    }));
                }

                if (input.dtype() == at::kHalf) {
                    grad_input = grad_input.to(at::kHalf);
                    grad_offset = grad_offset.to(at::kHalf);
                    grad_mask = grad_mask.to(at::kHalf);
                }
                return std::make_tuple(grad_input, grad_offset, grad_mask);
            }
        }
    }
}
