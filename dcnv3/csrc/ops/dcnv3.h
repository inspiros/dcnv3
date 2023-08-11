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

#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace dcnv3 {
    namespace ops {
        DCNV3_API at::Tensor dcnv3_forward(
                const at::Tensor &input, const at::Tensor &offset,
                const at::Tensor &mask, const int64_t kernel_h,
                const int64_t kernel_w, const int64_t stride_h,
                const int64_t stride_w, const int64_t pad_h, const int64_t pad_w,
                const int64_t dilation_h, const int64_t dilation_w,
                const int64_t group, const int64_t group_channels,
                const double_t offset_scale, const int64_t im2col_step, const int64_t remove_center);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor, at::Tensor> _dcnv3_backward(
                    const at::Tensor &input, const at::Tensor &offset,
                    const at::Tensor &mask, int64_t kernel_h, int64_t kernel_w,
                    int64_t stride_h, int64_t stride_w, int64_t pad_h,
                    int64_t pad_w, int64_t dilation_h, int64_t dilation_w,
                    int64_t group, int64_t group_channels,
                    double_t offset_scale, const at::Tensor &grad_output,
                    int64_t im2col_step, int64_t remove_center);
        }
    }
}
