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

// See docs in ../ops/array_ops.cc.
#define EIGEN_USE_THREADS

#include <atomic>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/gather_nd_op.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

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
    OP_REQUIRES(c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
                errors::InvalidArgument("params must be at least a vector"));
    OP_REQUIRES(c, TensorShapeUtils::IsMatrixOrHigher(indices.shape()),
                errors::InvalidArgument("indices must be at least a matrix"));
    OP_REQUIRES(
        c, indices.dim_size(indices.dims() - 1) == params.dims(),
        errors::InvalidArgument(
            "index innermost dimension length must equal params rank; saw: ",
            indices.dim_size(indices.dims() - 1), " vs. ", params.dims()));

    // Check that we have enough index space
    const int64 N_big = indices.NumElements() / params.dims();
    OP_REQUIRES(c, N_big <= std::numeric_limits<int>::max(),
                errors::InvalidArgument(
                    "indices has too many elements for int indexing: ", N_big,
                    " > ", std::numeric_limits<int>::max()));
    const int N = indices.NumElements() / params.dims();
    OP_REQUIRES(
        c, params.NumElements() <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.NumElements() too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.NumElements(), " > ",
                                std::numeric_limits<Index>::max()));

    // The result shape is indices.shape[:-1]
    TensorShape result_shape(indices.shape());
    result_shape.RemoveDim(result_shape.dims() - 1);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));
    if (N > 0) {
      auto indices_mat = indices.flat_inner_dims<Index>();
      auto out_flat = out->flat<T>();

      Index bad_i = -1;
      switch (params.dims()) {
#define PARAMS_CASE(NDIM)                                                  \
  case NDIM: {                                                             \
    functor::GatherNd<Device, T, Index, NDIM> functor;                     \
    auto params_tensor = params.tensor<T, NDIM>();                         \
    bad_i = functor(c->eigen_device<Device>(), params_tensor, indices_mat, \
                    out_flat);                                             \
  } break

        PARAMS_CASE(1);
        PARAMS_CASE(2);
        PARAMS_CASE(3);
        PARAMS_CASE(4);
        PARAMS_CASE(5);
        default:
          OP_REQUIRES(c, false,
                      errors::InvalidArgument(
                          "Only param tensors with ranks between 1 and 5 "
                          "are currently supported.  Tensor rank: ",
                          params.dims()));
      }

      OP_REQUIRES(c, bad_i < 0,
                  errors::InvalidArgument(
                      "flat indices[", bad_i, ", :] = [",
                      str_util::Join(gtl::ArraySlice<Index>(
                                         &indices_mat(bad_i, 0), params.dims()),
                                     ", "),
                      "] does not index into param (shape: ",
                      params.shape().DebugString(), ")."));
    }
  }
};

// Specialization of GatherNd to CPU
namespace generator {

template <typename T, typename Index, int NDIM>
class GatherNdGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  GatherNdGenerator(typename TTypes<Index>::ConstMatrix Tindices,
                    typename TTypes<T, NDIM>::ConstTensor Tparams,
                    Index* error_loc)
      : Tindices_(Tindices), Tparams_(Tparams), error_loc_(*error_loc) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, 1>& loc_array) const {
    Index loc = loc_array[0];
    Eigen::array<Eigen::DenseIndex, NDIM> ix;
    bool out_of_bounds = false;
    for (int i = 0; i < NDIM; ++i) {
      Index ix_i = Tindices_(loc, i);
      ix[i] = ix_i;
      out_of_bounds |= !FastBoundsCheck(ix_i, Tparams_.dimension(i));
    }
    if (out_of_bounds) {
      error_loc_ = loc;
      return T();  // Return 0, 0.0, or '', etc.
    } else {
      return Tparams_(ix);
    }
  }

 private:
  typename TTypes<Index>::ConstMatrix Tindices_;
  typename TTypes<T, NDIM>::ConstTensor Tparams_;
  Index& error_loc_;
};

}  // namespace generator

namespace functor {

template <typename T, typename Index, int NDIM>
struct GatherNd<CPUDevice, T, Index, NDIM> {
  Index operator()(const CPUDevice& d,
                   typename TTypes<T, NDIM>::ConstTensor Tparams,
                   typename TTypes<Index>::ConstMatrix Tindices,
                   typename TTypes<T>::Flat Tout) {
    Index error_loc(-1);
    generator::GatherNdGenerator<T, Index, NDIM> gather_nd_generator(Tindices,
                                                                     Tparams,
                                                                     &error_loc);
    Tout.device(d) = Tout.generate(gather_nd_generator);

    // error_loc() returns -1 if there's no out-of-bounds index,
    // otherwise it returns the location of an OOB index in Tindices.
    return error_loc;
  }
};

}  // namespace functor

#define REGISTER_GATHER_ND_FULL(dev, type, index_type)                 \
  REGISTER_KERNEL_BUILDER(Name("GatherNd")                             \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices"), \
                          GatherNdOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ND_ALL_INDICES(dev, type) \
  REGISTER_GATHER_ND_FULL(dev, type, int32);      \
  REGISTER_GATHER_ND_FULL(dev, type, int64)

#define REGISTER_GATHER_ND_CPU(type) REGISTER_GATHER_ND_ALL_INDICES(CPU, type)

TF_CALL_ALL_TYPES(REGISTER_GATHER_ND_CPU);

#undef REGISTER_GATHER_ND_CPU

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, NDIM)                     \
  template <>                                                            \
  Index GatherNd<GPUDevice, T, Index, NDIM>::operator()(                 \
      const GPUDevice& d, typename TTypes<T, NDIM>::ConstTensor Tparams, \
      typename TTypes<Index>::ConstMatrix Tindices,                      \
      typename TTypes<T>::Flat Tout);                                    \
  extern template struct GatherNd<GPUDevice, T, Index, NDIM>;

#define DECLARE_GPU_SPECS_INDEX(T, Index)    \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 1); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 2); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 3); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 4); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 5)

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GATHER_ND_GPU(type) REGISTER_GATHER_ND_ALL_INDICES(GPU, type)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHER_ND_GPU);

#undef REGISTER_GATHER_ND_GPU

#endif  // GOOGLE_CUDA

#undef REGISTER_GATHER_ND_ALL_INDICES
#undef REGISTER_GATHER_ND_FULL

}  // namespace tensorflow
