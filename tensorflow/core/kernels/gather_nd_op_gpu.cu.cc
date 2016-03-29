/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/gather_nd_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace generator {

template <typename T, typename Index, int NDIM>
class GatherNdGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  GatherNdGenerator(typename TTypes<const Index, 2>::Tensor32Bit Tindices,
                    typename TTypes<const T, NDIM>::Tensor32Bit Tparams)
      : Tindices_(Tindices), Tparams_(Tparams) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<int, 1>& loc_array) const {
    int loc = loc_array[0];
    Eigen::array<int, NDIM> ix;
    bool out_of_bounds = false;
    for (int i = 0; i < NDIM; ++i) {
      int ix_i = Tindices_(loc, i);
      ix[i] = ix_i;
      out_of_bounds |= !FastBoundsCheck(ix_i, Tparams_.dimension(i));
    }
    if (out_of_bounds) {
      return T(0);  // TODO(ebrevdo): Pass error back to host.
    } else {
      return Tparams_(ix);
    }
  }

 private:
  typename TTypes<const Index, 2>::Tensor32Bit Tindices_;
  typename TTypes<const T, NDIM>::Tensor32Bit Tparams_;
};

}  // namespace generator

namespace functor {

template <typename T, typename Index, int NDIM>
struct GatherNd<GPUDevice, T, Index, NDIM> {
  Index operator()(const GPUDevice& d,
                   typename TTypes<T, NDIM>::ConstTensor Tparams,
                   typename TTypes<Index>::ConstMatrix Tindices,
                   typename TTypes<T>::Flat Tout) {
    generator::GatherNdGenerator<T, Index, NDIM> gather_nd_generator(
        To32Bit(Tindices), To32Bit(Tparams));
    To32Bit(Tout).device(d) = To32Bit(Tout).generate(gather_nd_generator);

    // TODO(ebrevdo): enable indices validation on GPU.
    // Right now checking for indicies out of bound in the kernel would
    // require copying code between GPU/CPU, and is too slow.
    return -1;
  }
};

}  // namespace functor

#define DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, NDIM) \
  template struct functor::GatherNd<GPUDevice, T, Index, NDIM>;

#define DEFINE_GPU_SPECS_INDEX(T, Index)    \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 1); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 2); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 3); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 4); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 5);

#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, int32); \
  DEFINE_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
