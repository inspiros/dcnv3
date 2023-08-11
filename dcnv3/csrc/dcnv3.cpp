#include <torch/extension.h>
#include "dcnv3.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

namespace dcnv3 {
    int64_t cuda_version() {
#ifdef WITH_CUDA
        return CUDA_VERSION;
#else
        return -1;
#endif
    }

    TORCH_LIBRARY_FRAGMENT(dcnv3, m) {
        m.def("_cuda_version", &cuda_version);
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("_cuda_version", &cuda_version, "get cuda version");
        m.def("dcnv3_forward", &ops::dcnv3_forward, "dcnv3_forward");
        m.def("dcnv3_backward", &ops::detail::_dcnv3_backward, "dcnv3_backward");
    }
}
