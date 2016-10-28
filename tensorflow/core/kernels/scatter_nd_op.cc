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

// See docs in ../ops/state_ops.cc.
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/scatter_nd_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Check whether updates.shape = indices.shape[0] + params.shape[IXDIM:]
static bool ValidUpdateShape(const TensorShape& params_shape,
                             const Tensor& indices, const Tensor& updates) {
  int64 indices_nd = 1;
  if (indices.dims() > 1) {
    indices_nd = indices.dim_size(1);
  }
  for (int d = indices_nd; d < params_shape.dims(); d++) {
    if (updates.dim_size(d - indices_nd + 1) != params_shape.dim_size(d)) {
      return false;
    }
  }
  return true;
}

template <typename Index>
static void PrepareAndValidateInputs(OpKernelContext* c,
                                     const TensorShape& params_shape,
                                     const Tensor& indices,
                                     const Tensor& updates, int64* indices_nd,
                                     Index* num_updates, Index* slice_size) {
  const TensorShape& indices_shape(indices.shape());
  const TensorShape& updates_shape(updates.shape());

  OP_REQUIRES(
      c, TensorShapeUtils::IsVectorOrHigher(params_shape),
      errors::InvalidArgument("Output must be at least 1-D, ", "got shape: ",
                              params_shape.DebugString()));

  OP_REQUIRES(c, params_shape.num_elements() >= 0 ||
                     (indices.NumElements() == 0 && updates.NumElements() == 0),
              errors::InvalidArgument(
                  "Indices and updates specified for empty output", " shape"));

  OP_REQUIRES(c, updates.dim_size(0) == indices.dim_size(0),
              errors::InvalidArgument(
                  "The outermost dimension of updates and indices ",
                  "must match. Got indices.shape ", indices_shape.DebugString(),
                  ", updates.shape ", updates_shape.DebugString()));
  OP_REQUIRES(
      c, ValidUpdateShape(params_shape, indices, updates),
      errors::InvalidArgument(
          "Must have updates.shape = indices.shape[0] + params_shape[IXDIM:], ",
          "got updates.shape ", updates_shape.DebugString(), ", indices.shape ",
          indices_shape.DebugString(), ", params_shape ",
          params_shape.DebugString()));
  // Check that we have enough index space
  const int64 N_big = indices.NumElements();
  OP_REQUIRES(c, N_big <= std::numeric_limits<Index>::max(),
              errors::InvalidArgument(
                  "indices has too many elements for ",
                  DataTypeString(DataTypeToEnum<Index>::v()), " indexing: ",
                  N_big, " > ", std::numeric_limits<Index>::max()));
  OP_REQUIRES(
      c, params_shape.dim_size(0) <= std::numeric_limits<Index>::max(),
      errors::InvalidArgument("params_shape[0] too large for ",
                              DataTypeString(DataTypeToEnum<Index>::v()),
                              " indexing: ", params_shape.dim_size(0), " > ",
                              std::numeric_limits<Index>::max()));

  // Calculate the number of dimensions in indices
  *indices_nd = 1;
  if (indices_shape.dims() > 1) {
    *indices_nd = indices_shape.dim_size(indices_shape.dims() - 1);
  }

  // Calculate the number of elements that make up each slice of our updated
  // tensor. This allows us to work with flattened tensors and copy over whole
  // slices at a time.
  Index total_nd = params_shape.dims();

  int64 slice_size_big = 1;
  for (int64 i = *indices_nd; i < total_nd; ++i) {
    slice_size_big *= params_shape.dim_size(i);
  }

  OP_REQUIRES(c, slice_size_big <= std::numeric_limits<Index>::max(),
              errors::InvalidArgument("slice size is too large for indexing: ",
                                      slice_size_big, " > ",
                                      std::numeric_limits<Index>::max()));

  *slice_size = static_cast<Index>(slice_size_big);

  const int64 safe_indices_nd = (*indices_nd < 1) ? 1 : *indices_nd;
  *num_updates = indices_shape.num_elements() / safe_indices_nd;
}

template <typename Device, typename T, typename Index>
class ScatterNdOp : public OpKernel {
 public:
  explicit ScatterNdOp(OpKernelConstruction* c) : OpKernel(c) {
    const DataType dt = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<Index>::v();
    OP_REQUIRES_OK(c, c->MatchSignature({index_t, dt, index_t}, {dt}));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& indices = c->input(0);
    const Tensor& updates = c->input(1);
    const Tensor& shape_input = c->input(2);

    OP_REQUIRES(c, shape_input.dims() == 1,
                errors::InvalidArgument("Shape must be a vector"));
    auto vec = shape_input.flat<Index>();
    TensorShape shape;
    TensorShapeUtils::MakeShape(vec.data(), vec.size(), &shape);

    int64 indices_nd;
    Index num_updates;
    Index slice_size;
    PrepareAndValidateInputs<Index>(c, shape, indices, updates, &indices_nd,
                                    &num_updates, &slice_size);
    if (!c->status().ok()) return;

    Tensor scratch;
    OP_REQUIRES_OK(c, c->allocate_temp(DT_INT32, TensorShape(), &scratch));

    auto scratch_scalar = scratch.scalar<Index>();
    auto indices_flat = indices.flat_inner_dims<Index>();
    auto updates_flat = updates.shaped<T, 2>({num_updates, slice_size});

    Index bad_i = -1;
    switch (indices_nd) {
#define PARAMS_CASE(IXDIM)                                                   \
  case IXDIM: {                                                              \
    Tensor* out = nullptr;                                                   \
    OP_REQUIRES_OK(c, c->allocate_output(0, shape, &out));                   \
    functor::SetZeroFunctor<Device, T> fill;                                 \
    fill(c->eigen_device<Device>(), out->flat<T>());                         \
    if (shape.num_elements() > 0) {                                          \
      auto output_flat = out->flat_outer_dims<T, (IXDIM) + 1>();             \
      functor::ScatterNdFunctor<Device, T, Index,                            \
                                scatter_nd_op::UpdateOp::ADD, (IXDIM)>       \
          functor;                                                           \
      bad_i = functor(c->eigen_device<Device>(), slice_size, scratch_scalar, \
                      output_flat, indices_flat, updates_flat, output_flat); \
    }                                                                        \
  } break
      PARAMS_CASE(0);
      PARAMS_CASE(1);
      PARAMS_CASE(2);
      PARAMS_CASE(3);
      PARAMS_CASE(4);
      PARAMS_CASE(5);
#undef PARAMS_CASE
      default:
        OP_REQUIRES(c, false,
                    errors::InvalidArgument(
                        "Only indices.shape[-1] values between 0 and 5 "
                        "are currently supported.  Requested rank: ",
                        indices_nd));
    }
    OP_REQUIRES(
        c, bad_i < 0,
        errors::InvalidArgument(
            "Invalid indices: ", SliceDebugString(indices.shape(), bad_i),
            " = [", str_util::Join(gtl::ArraySlice<Index>(
                                       &indices_flat(bad_i, 0), indices_nd),
                                   ", "),
            "] does not index into ", shape.DebugString()));
  }
};

template <typename Device, typename T, typename Index,
          scatter_nd_op::UpdateOp op>
class ScatterNdUpdateOp : public OpKernel {
 public:
  explicit ScatterNdUpdateOp(OpKernelConstruction* c) : OpKernel(c) {
    const DataType dt = DataTypeToEnum<T>::v();
    const DataType dt_ref = DataTypeToEnum<T>::ref();
    const DataType index_t = DataTypeToEnum<Index>::v();
    OP_REQUIRES_OK(c, c->MatchSignature({dt_ref, index_t, dt}, {dt_ref}));
    OP_REQUIRES_OK(c, c->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* c) override {
    if (use_exclusive_lock_) {
      // Hold mutex while we apply updates
      mutex_lock l(*c->input_ref_mutex(0));
      DoCompute(c);
    } else {
      DoCompute(c);
    }
  }

 private:
  bool use_exclusive_lock_;

  void DoCompute(OpKernelContext* c) {
    Tensor params = c->mutable_input(0, use_exclusive_lock_);
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);
    const TensorShape& params_shape(params.shape());

    int64 indices_nd;
    Index num_updates;
    Index slice_size;

    OP_REQUIRES(c, params.IsInitialized(),
                errors::FailedPrecondition("Null ref for params"));
    PrepareAndValidateInputs<Index>(c, params_shape, indices, updates,
                                    &indices_nd, &num_updates, &slice_size);
    if (!c->status().ok()) return;

    Tensor scratch;
    OP_REQUIRES_OK(c, c->allocate_temp(DT_INT32, TensorShape(), &scratch));

    auto scratch_scalar = scratch.scalar<Index>();
    auto indices_flat = indices.flat_inner_dims<Index>();
    auto updates_flat = updates.shaped<T, 2>({num_updates, slice_size});

    Index bad_i = -1;
    c->forward_ref_input_to_ref_output(0, 0);
    switch (indices_nd) {
#define PARAMS_CASE(IXDIM)                                                 \
  case IXDIM: {                                                            \
    auto params_flat = params.flat_outer_dims<T, (IXDIM) + 1>();           \
    functor::ScatterNdFunctor<Device, T, Index, op, IXDIM> functor;        \
    bad_i = functor(c->eigen_device<Device>(), slice_size, scratch_scalar, \
                    params_flat, indices_flat, updates_flat, params_flat); \
  } break
      PARAMS_CASE(0);
      PARAMS_CASE(1);
      PARAMS_CASE(2);
      PARAMS_CASE(3);
      PARAMS_CASE(4);
      PARAMS_CASE(5);
#undef PARAMS_CASE
      default:
        OP_REQUIRES(c, false,
                    errors::InvalidArgument(
                        "Only indices.shape[-1] values between 1 and 5 "
                        "are currently supported.  Requested rank: ",
                        indices_nd));
    }
    OP_REQUIRES(
        c, bad_i < 0,
        errors::InvalidArgument(
            "Invalid indices: ", SliceDebugString(indices.shape(), bad_i),
            " = [", str_util::Join(gtl::ArraySlice<Index>(
                                       &indices_flat(bad_i, 0), indices_nd),
                                   ", "),
            "] is not in [0, ", params.dim_size(0), ")"));
  }
};

// Specialization of ScatterNdSliceGenerator to CPU
namespace generator {

template <typename T, typename Index, scatter_nd_op::UpdateOp op>
class UpdateExecutor {
 public:
  static void Update(T* input, const T* updates, T* output, Index slice_size);
};

template <typename T, typename Index>
class UpdateExecutor<T, Index, scatter_nd_op::UpdateOp::ASSIGN> {
 public:
  static void Update(T* /* unused */, const T* updates, T* output,
                     Index slice_size) {
    std::copy_n(updates, slice_size, output);
  }
};

template <typename T, typename Index>
class UpdateExecutor<T, Index, scatter_nd_op::UpdateOp::ADD> {
 public:
  static void Update(T* input, const T* updates, T* output, Index slice_size) {
    std::transform(input, input + slice_size, updates, output, std::plus<T>());
  }
};

template <typename T, typename Index>
class UpdateExecutor<T, Index, scatter_nd_op::UpdateOp::SUB> {
 public:
  static void Update(T* input, const T* updates, T* output, Index slice_size) {
    std::transform(input, input + slice_size, updates, output, std::minus<T>());
  }
};

template <typename T, typename Index>
class UpdateExecutor<T, Index, scatter_nd_op::UpdateOp::MUL> {
 public:
  static void Update(T* input, const T* updates, T* output, Index slice_size) {
    std::transform(input, input + slice_size, updates, output,
                   std::multiplies<T>());
  }
};

template <typename T, typename Index>
class UpdateExecutor<T, Index, scatter_nd_op::UpdateOp::DIV> {
 public:
  static void Update(T* input, const T* updates, T* output, Index slice_size) {
    std::transform(input, input + slice_size, updates, output,
                   std::divides<T>());
  }
};

template <typename T, typename Index, scatter_nd_op::UpdateOp op, int IXDIM>
class ScatterNdSliceGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE ScatterNdSliceGenerator(
      const Index slice_size, typename TTypes<T, IXDIM + 1>::Tensor Tparams,
      typename TTypes<Index, 2>::ConstTensor Tindices,
      typename TTypes<T, 2>::ConstTensor Tupdates,
      typename TTypes<T, IXDIM + 1>::Tensor Toutput,
      std::atomic<Index>* error_loc)
      : slice_size_(slice_size),
        Tparams_(Tparams),
        Tindices_(Tindices),
        Tupdates_(Tupdates),
        Toutput_(Toutput),
        error_loc_(error_loc) {}

  EIGEN_DEVICE_FUNC bool GenerateIndices(
      const Index loc, Eigen::array<Eigen::DenseIndex, IXDIM + 1>* ix) const {
    (*ix)[IXDIM] = 0;
    bool out_of_bounds = false;
    for (int i = 0; i < IXDIM; ++i) {
      const Index ix_i = internal::SubtleMustCopy(Tindices_(loc, i));
      (*ix)[i] = ix_i;
      out_of_bounds |= !FastBoundsCheck(ix_i, Tparams_.dimension(i));
    }
    return out_of_bounds;
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int32
  operator()(const Eigen::array<Eigen::DenseIndex, 1>& loc_array) const {
    auto loc = loc_array[0];
    Eigen::array<Eigen::DenseIndex, IXDIM + 1> ix_params;
    Eigen::array<Eigen::DenseIndex, 2> ix_updates;
    ix_updates[0] = loc;
    ix_updates[1] = 0;
    const bool out_of_bounds = GenerateIndices(loc, &ix_params);
    if (TF_PREDICT_FALSE(out_of_bounds)) {
      error_loc_->store(loc);
    } else {
      UpdateExecutor<T, Index, op>::Update(&Tparams_(ix_params),
                                           &Tupdates_(ix_updates),
                                           &Toutput_(ix_params), slice_size_);
    }
    return static_cast<int32>(0);  // Return something...
  }

 protected:
  const Index slice_size_;
  mutable typename TTypes<T, IXDIM + 1>::Tensor Tparams_;
  const typename TTypes<Index, 2>::ConstTensor Tindices_;
  const typename TTypes<T, 2>::ConstTensor Tupdates_;
  mutable typename TTypes<T, IXDIM + 1>::Tensor Toutput_;
  std::atomic<Index>* error_loc_;
};

}  // namespace generator

namespace functor {
// Implementation of update functor for CPU.
template <typename T, typename Index, scatter_nd_op::UpdateOp op, int IXDIM>
struct ScatterNdFunctor<CPUDevice, T, Index, op, IXDIM> {
  Index operator()(const CPUDevice& d, const Index slice_size,
                   typename TTypes<Index>::Scalar Tscratch,
                   typename TTypes<T, IXDIM + 1>::Tensor Tparams,
                   typename TTypes<Index, 2>::ConstTensor Tindices,
                   typename TTypes<T, 2>::ConstTensor Tupdates,
                   typename TTypes<T, IXDIM + 1>::Tensor Toutput) {
    std::atomic<Index> error_loc(-1);

    const Eigen::DenseIndex batch_size = Tindices.dimension(0);
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::Tensor<Eigen::DenseIndex, 1>::Dimensions reshape_dims{{ 1 }};
    Eigen::array<Eigen::DenseIndex, 1> broadcast_dims{{ batch_size }};
#else
    Eigen::IndexList<Eigen::type2index<1> > reshape_dims;
    Eigen::IndexList<Eigen::DenseIndex> broadcast_dims;
    broadcast_dims.set(0, batch_size);
#endif

    generator::ScatterNdSliceGenerator<T, Index, op, IXDIM> generator(
        slice_size, Tparams, Tindices, Tupdates, Toutput, &error_loc);
    Tscratch.device(d) = Tscratch.reshape(reshape_dims)
                             .broadcast(broadcast_dims)
                             .generate(generator)
                             .sum();

    // error_loc() returns -1 if there's no out-of-bounds index,
    // otherwise it returns the location of an OOB index in Tindices.
    return error_loc.load();
  }
};
}  // namespace functor

#define REGISTER_SCATTER_ND_KERNEL_INDEX(type, index_type, dev, name)  \
  REGISTER_KERNEL_BUILDER(Name(name)                                   \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          ScatterNdOp<dev##Device, type, index_type>)

#define REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX(type, index_type, dev, name, \
                                                op)                          \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name(name)                                                             \
          .Device(DEVICE_##dev)                                              \
          .TypeConstraint<type>("T")                                         \
          .TypeConstraint<index_type>("Tindices"),                           \
      ScatterNdUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_SCATTER_ND_KERNEL(type, dev, name)         \
  REGISTER_SCATTER_ND_KERNEL_INDEX(type, int32, dev, name); \
  REGISTER_SCATTER_ND_KERNEL_INDEX(type, int64, dev, name)

#define REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, name, op)         \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int32, dev, name, op); \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int64, dev, name, op)

#define REGISTER_SCATTER_ND_ADD_SUB(type, dev)                     \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdAdd",     \
                                    scatter_nd_op::UpdateOp::ADD); \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdSub",     \
                                    scatter_nd_op::UpdateOp::SUB); \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdMul",     \
                                    scatter_nd_op::UpdateOp::MUL); \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdDiv",     \
                                    scatter_nd_op::UpdateOp::DIV);

#define REGISTER_SCATTER_ND(type, dev) \
  REGISTER_SCATTER_ND_KERNEL(type, dev, "ScatterNd");

#define REGISTER_SCATTER_ND_UPDATE(type, dev)                     \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdUpdate", \
                                    scatter_nd_op::UpdateOp::ASSIGN);

// Registers CPU kernels.
#define REGISTER_SCATTER_ND_ADD_SUB_CPU(type) \
  REGISTER_SCATTER_ND_ADD_SUB(type, CPU);

#define REGISTER_SCATTER_ND_UPDATE_CPU(type) \
  REGISTER_SCATTER_ND_UPDATE(type, CPU);

#define REGISTER_SCATTER_ND_CPU(type) REGISTER_SCATTER_ND(type, CPU);

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_ADD_SUB_CPU);
TF_CALL_ALL_TYPES(REGISTER_SCATTER_ND_UPDATE_CPU);
TF_CALL_ALL_TYPES(REGISTER_SCATTER_ND_CPU);

// Registers GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_SCATTER_ND_ADD_SUB_GPU(type) \
  REGISTER_SCATTER_ND_ADD_SUB(type, GPU);

#define REGISTER_SCATTER_ND_UPDATE_GPU(type) \
  REGISTER_SCATTER_ND_UPDATE(type, GPU);

// TODO(simister): Re-enable when GPU support is working.
// TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ND_ADD_SUB_GPU);
// TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ND_UPDATE_GPU);

#endif  // GOOGLE_CUDA

#undef REGISTER_SCATTER_ND_ADD
#undef REGISTER_SCATTER_ND_ADD_SUB
#undef REGISTER_SCATTER_ND_ADD_SUB_CPU
#undef REGISTER_SCATTER_ND_ADD_SUB_GPU
#undef REGISTER_SCATTER_ND_UPDATE
#undef REGISTER_SCATTER_ND_UPDATE_CPU
#undef REGISTER_SCATTER_ND_UPDATE_GPU
#undef REGISTER_SCATTER_ND_KERNEL
#undef REGISTER_SCATTER_ND_KERNEL_INDEX

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {

#define DECLARE_GPU_SPECS_OP(T, Index, op, NDIM)                     \
  template <>                                                        \
  Index ScatterNdFunctor<GPUDevice, T, Index, op, NDIM>::operator()( \
      OpKernelContext* c, const GPUDevice& d,                        \
      typename TTypes<T, IXDIM>::Tensor params,                      \
      typename TTypes<Index, 2>::ConstTensor indices,                \
      typename TTypes<T, 2>::ConstTensor updates);                   \
  extern template struct ScatterNdFunctor<GPUDevice, T, Index, op>;

#define DECLARE_GPU_SPECS_OPS(T, Index, op) \
  DECLARE_GPU_SPECS_OP(T, Index, op, 0);    \
  DECLARE_GPU_SPECS_OP(T, Index, op, 1);    \
  DECLARE_GPU_SPECS_OP(T, Index, op, 2);    \
  DECLARE_GPU_SPECS_OP(T, Index, op, 3);    \
  DECLARE_GPU_SPECS_OP(T, Index, op, 4);    \
  DECLARE_GPU_SPECS_OP(T, Index, op, 5)

#define DECLARE_GPU_SPECS_INDEX(T, Index)                           \
  DECLARE_GPU_SPECS_OPS(T, Index, scatter_nd_op::UpdateOp::ASSIGN); \
  DECLARE_GPU_SPECS_OPS(T, Index, scatter_nd_op::UpdateOp::ADD);    \
  DECLARE_GPU_SPECS_OPS(T, Index, scatter_nd_op::UpdateOp::SUB);    \
  DECLARE_GPU_SPECS_OPS(T, Index, scatter_nd_op::UpdateOp::MUL);    \
  DECLARE_GPU_SPECS_OPS(T, Index, scatter_nd_op::UpdateOp::DIV);

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64);

// TODO(simister): Re-enable when GPU support is working.
// TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
#undef DECLARE_GPU_SPECS_OPS
#undef DECLARE_GPU_SPECS_OP

}  // namespace functor
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
