/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/gather_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class GatherOp : public OpKernel {
 public:
  //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8,
  //   etc. here for the type of the second input argument.  Should
  //   we have the framework do some sort of integer promotion
  //   automatically, or should that be something that users have to
  //   do explicitly with a conversion operator in the graph?
  explicit GatherOp(OpKernelConstruction* c) : OpKernel(c) {
    const DataType dt = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<Index>::v();
    OP_REQUIRES_OK(c, c->MatchSignature({dt, index_t}, {dt}));
    // We used to grab the validate_indices attribute here, but now we
    // always validate indices since the speed difference was only 1.5%.
    // TODO(irving): Remove the validate_indices attribute once we have
    // support for removing attrs in a backwards compatible way.
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& params = c->input(0);
    const Tensor& indices = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    // Check that we have enough index space
    const int64 N = indices.NumElements();
    OP_REQUIRES(
        c, params.dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));

    // The result shape is indices.shape + params.shape[1:].
    TensorShape result_shape = indices.shape();
    for (int i = 1; i < params.dims(); i++) {
      result_shape.AddDim(params.dim_size(i));
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));
    if (N > 0) {
      auto params_flat = params.flat_outer_dims<T>();
      auto indices_flat = indices.flat<Index>();
      auto out_flat = out->shaped<T, 2>({N, out->NumElements() / N});

      functor::Gather<Device, T, Index> functor;
      int64 bad_i = functor(c->eigen_device<Device>(), params_flat,
                            indices_flat, out_flat);

      OP_REQUIRES(
          c, bad_i < 0,
          errors::InvalidArgument(
              "indices", SliceDebugString(indices.shape(), bad_i), " = ",
              indices_flat(bad_i), " is not in [0, ", params.dim_size(0), ")"));
    }
  }
};

namespace functor {

// Helper method to copy using memcpy.
template <typename T, typename Index, typename SliceIndex,
          SliceIndex static_slice_elems>
SliceIndex HandleCopies(typename TTypes<T>::ConstMatrix params,
                        typename TTypes<Index>::ConstFlat indices,
                        SliceIndex slice_elems,
                        typename TTypes<T>::Matrix out) {
  const SliceIndex first_dim_size =
      static_cast<SliceIndex>(indices.dimension(0));
  const Index limit = static_cast<Index>(params.dimension(0));
  T* out_base = &out(0, 0);
  const T* params_base = &params(0, 0);
  if (static_slice_elems >= 0) {
    // Give compiler static knowledge of the number of elements/bytes
    CHECK_EQ(static_slice_elems, slice_elems);
    slice_elems = static_slice_elems;
  }
  // Compute slice_bytes here so that static knowledge is available
  const size_t slice_bytes = slice_elems * sizeof(T);
  for (SliceIndex i = 0; i < first_dim_size; i++) {
    const SliceIndex j = i + 1;
    if (j < first_dim_size) {
      port::prefetch<port::PREFETCH_HINT_T0>(&params(indices(j), 0));
      port::prefetch<port::PREFETCH_HINT_T0>(&out(j, 0));
    }
    // Grab the index and check its validity.  An earlier version of the
    // code checked it and then grabbed it from memory a second time, which
    // was a security risk since it could have changed in between.
    const Index index = internal::SubtleMustCopy(indices(i));
    if (!FastBoundsCheck(index, limit)) return i;
    // Copy using memcpy if possible, otherwise an Eigen loop
    if (Allocator::is_simple<T>::value) {
      memcpy(out_base + i * slice_elems, params_base + index * slice_elems,
             slice_bytes);
    } else {
      out.template chip<0>(i) = params.template chip<0>(index);
    }
  }
  return -1;
}

// Specialization gather functor for CPU.
template <typename T, typename Index>
struct Gather<CPUDevice, T, Index> {
  int64 operator()(const CPUDevice& d, typename TTypes<T>::ConstMatrix params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T>::Matrix out) {
    const int64 N = indices.size();
    const int64 slice_size = out.size() / N;
    int64 bad_i;

    bool use_large = (slice_size > std::numeric_limits<int32>::max() ||
                      params.size() > std::numeric_limits<int32>::max() ||
                      N > std::numeric_limits<int32>::max());
#define CALL(elems)                                                   \
  do {                                                                \
    if (use_large) {                                                  \
      bad_i = HandleCopies<T, Index, int64, elems>(params, indices,   \
                                                   slice_size, out);  \
    } else {                                                          \
      const int32 small_slice = static_cast<int32>(slice_size);       \
      bad_i = HandleCopies<T, Index, int32, elems>(params, indices,   \
                                                   small_slice, out); \
    }                                                                 \
  } while (0)

    if (slice_size == 10)
      CALL(10);
    else if (slice_size == 20)
      CALL(20);
    else
      CALL(-1);
#undef CALL

    return bad_i;
  }
};
}  // namespace functor

#define REGISTER_GATHER_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("Gather")                               \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices"), \
                          GatherOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, type, int32);      \
  REGISTER_GATHER_FULL(dev, type, int64)

#define REGISTER_GATHER_CPU(type) REGISTER_GATHER_ALL_INDICES(CPU, type)

TF_CALL_ALL_TYPES(REGISTER_GATHER_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_CPU);

#undef REGISTER_GATHER_CPU

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPECS_INDEX(T, Index)                          \
  template <>                                                      \
  Index Gather<GPUDevice, T, Index>::operator()(                   \
      const GPUDevice& d, typename TTypes<T>::ConstMatrix Tparams, \
      typename TTypes<Index>::ConstFlat Tindices,                  \
      typename TTypes<T>::Matrix Tout);                            \
  extern template struct Gather<GPUDevice, T, Index>;

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHER_GPU);

#undef REGISTER_GATHER_GPU

#endif  // GOOGLE_CUDA

#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

}  // namespace tensorflow
