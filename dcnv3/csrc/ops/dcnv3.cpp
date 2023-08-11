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

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#ifdef WITH_CUDA

#include "cuda/dcnv3_cuda.h"

#endif

namespace dcnv3 {
    namespace ops {
        at::Tensor dcnv3_forward(
                const at::Tensor &input, const at::Tensor &offset,
                const at::Tensor &mask, const int64_t kernel_h,
                const int64_t kernel_w, const int64_t stride_h,
                const int64_t stride_w, const int64_t pad_h, const int64_t pad_w,
                const int64_t dilation_h, const int64_t dilation_w,
                const int64_t group, const int64_t group_channels,
                const double_t offset_scale, const int64_t im2col_step, const int64_t remove_center) {
            C10_LOG_API_USAGE_ONCE("dcnv3.csrc.ops.dcnv3.dcnv3")
            if (input.is_cuda()) {
#ifdef WITH_CUDA
                return cuda::dcnv3_forward(input, offset, mask, kernel_h, kernel_w,
                                           stride_h, stride_w, pad_h, pad_w, dilation_h,
                                           dilation_w, group, group_channels,
                                           offset_scale, im2col_step, remove_center);
#else
                AT_ERROR("Not compiled with GPU support");
#endif
            }
            AT_ERROR("Not implemented on the CPU");
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor, at::Tensor> _dcnv3_backward(
                    const at::Tensor &input, const at::Tensor &offset,
                    const at::Tensor &mask, const int64_t kernel_h, const int64_t kernel_w,
                    const int64_t stride_h, const int64_t stride_w, const int64_t pad_h,
                    const int64_t pad_w, const int64_t dilation_h, const int64_t dilation_w,
                    const int64_t group, const int64_t group_channels,
                    const double_t offset_scale, const at::Tensor &grad_output,
                    const int64_t im2col_step, const int64_t remove_center) {
                if (input.is_cuda()) {
#ifdef WITH_CUDA
                    return cuda::dcnv3_backward(input, offset, mask, kernel_h, kernel_w,
                                                stride_h, stride_w, pad_h, pad_w, dilation_h,
                                                dilation_w, group, group_channels,
                                                offset_scale, grad_output, im2col_step, remove_center);
#else
                    AT_ERROR("Not compiled with GPU support");
#endif
                }
                AT_ERROR("Not implemented on the CPU");
            }
        }

        TORCH_LIBRARY_FRAGMENT(dcnv3, m) {
            m.def("dcnv3::dcnv3_forward", &dcnv3_forward);
            m.def("dcnv3::_dcnv3_backward", &detail::_dcnv3_backward);
        }
    }
}
