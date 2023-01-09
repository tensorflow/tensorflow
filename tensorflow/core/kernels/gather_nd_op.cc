/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/gather_nd_op.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class GatherNdOp : public OpKernel {
 public:
  explicit GatherNdOp(OpKernelConstruction* c) : OpKernel(c) {
    const DataType dt = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<Index>::v();
    OP_REQUIRES_OK(c, c->MatchSignature({dt, index_t}, {dt}));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& params = c->input(0);
    const Tensor& indices = c->input(1);

    Tensor out;
    OP_REQUIRES_OK(
        c, functor::DoGatherNd<Device, T, Index>(c, params, indices, &out));
    c->set_output(0, out);
  }
};

#define REGISTER_GATHER_ND_FULL(dev, type, index_type)                 \
  REGISTER_KERNEL_BUILDER(Name("GatherNd")                             \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices"), \
                          GatherNdOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ND_ALL_INDICES(dev, type) \
  REGISTER_GATHER_ND_FULL(dev, type, int32);      \
  REGISTER_GATHER_ND_FULL(dev, type, int64_t)

#define REGISTER_GATHER_ND_CPU(type) REGISTER_GATHER_ND_ALL_INDICES(CPU, type)

// TODO(ebrevdo): This is a pure data-movement kernel. It shouldn't be
// instantiated for all different types. Instead, all the types should
// be coalesced. So we should only have int8, int16, int32, int64 support.
// And float is redirected to int32, double is redirected to int64,
// and complex<float> is redirected to int32 with twice the number of
// entries, similarly for complex<double>.
//
// Same for the GPU kernel.
TF_CALL_ALL_TYPES(REGISTER_GATHER_ND_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_ND_CPU);

#undef REGISTER_GATHER_ND_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, NDIM)          \
  template <>                                                 \
  Index GatherNdSlice<GPUDevice, T, Index, NDIM>::operator()( \
      const GPUDevice& d, const Index slice_size,             \
      typename TTypes<int32>::Scalar Tscratch,                \
      typename TTypes<T, NDIM + 1>::ConstTensor Tparams,      \
      typename TTypes<Index>::ConstMatrix Tindices,           \
      typename TTypes<T>::Matrix Tout);                       \
  extern template struct GatherNdSlice<GPUDevice, T, Index, NDIM>;

#define DECLARE_GPU_SPECS_INDEX(T, Index)    \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 0); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 1); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 2); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 3); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 4); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 5); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 6); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 7);

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64_t)

TF_CALL_int32(DECLARE_GPU_SPECS);
TF_CALL_int64(DECLARE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_bfloat16(DECLARE_GPU_SPECS);
TF_CALL_COMPLEX_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GATHER_ND_GPU(type) REGISTER_GATHER_ND_ALL_INDICES(GPU, type)

TF_CALL_int32(REGISTER_GATHER_ND_GPU);
TF_CALL_int64(REGISTER_GATHER_ND_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHER_ND_GPU);
TF_CALL_bfloat16(REGISTER_GATHER_ND_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GATHER_ND_GPU);

#undef REGISTER_GATHER_ND_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_GATHER_ND_ALL_INDICES
#undef REGISTER_GATHER_ND_FULL

}  // namespace tensorflow
