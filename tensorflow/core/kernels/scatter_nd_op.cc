/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Check whether updates.shape = indices.shape[:batch_dim] +
// params_shape[slice_dim:]
static Status ValidateUpdateShape(const TensorShape& params_shape,
                                  const Tensor& indices,
                                  const Tensor& updates) {
  const int64 slice_dim =
      (indices.dims() > 1) ? indices.dim_size(indices.dims() - 1) : 1;
  const int64 batch_dim = (indices.dims() > 1) ? indices.dims() - 1 : 1;

#define SHAPE_ERR                                               \
  errors::InvalidArgument(                                      \
      "Must have updates.shape = indices.shape[:batch_dim] + ", \
      "params_shape[slice_dim:], got updates.shape: ",          \
      updates.shape().DebugString(),                            \
      ", indices.shape: ", indices.shape().DebugString(),       \
      ", params_shape: ", params_shape.DebugString(),           \
      ", slice_dim: ", slice_dim, ", and batch_dim: ", batch_dim)

  if (updates.dims() < batch_dim) return SHAPE_ERR;
  if (params_shape.dims() < slice_dim + (updates.dims() - batch_dim)) {
    return SHAPE_ERR;
  }
  if (updates.dims() != batch_dim + params_shape.dims() - slice_dim) {
    return SHAPE_ERR;
  }
  for (int d = 0; d < batch_dim; ++d) {
    if (updates.dim_size(d) != indices.dim_size(d)) return SHAPE_ERR;
  }
  for (int d = 0; d < updates.dims() - batch_dim; ++d) {
    if (updates.dim_size(d + batch_dim) !=
        params_shape.dim_size(d + slice_dim)) {
      return SHAPE_ERR;
    }
  }
#undef SHAPE_ERR
  return Status::OK();
}

template <typename Index>
static void PrepareAndValidateInputs(OpKernelContext* c,
                                     const TensorShape& params_shape,
                                     const Tensor& indices,
                                     const Tensor& updates, int64* slice_dim,
                                     Index* num_updates, Index* slice_size) {
  const TensorShape& indices_shape(indices.shape());
  const TensorShape& updates_shape(updates.shape());

  OP_REQUIRES(
      c, TensorShapeUtils::IsVectorOrHigher(params_shape),
      errors::InvalidArgument("Output must be at least 1-D, ",
                              "got shape: ", params_shape.DebugString()));

  OP_REQUIRES(c,
              params_shape.num_elements() >= 0 ||
                  (indices.NumElements() == 0 && updates.NumElements() == 0),
              errors::InvalidArgument(
                  "Indices and updates specified for empty output", " shape"));

  OP_REQUIRES(c, updates.dim_size(0) == indices.dim_size(0),
              errors::InvalidArgument(
                  "The outermost dimension of updates and indices ",
                  "must match. Got indices.shape ", indices_shape.DebugString(),
                  ", updates.shape ", updates_shape.DebugString()));
  OP_REQUIRES_OK(c, ValidateUpdateShape(params_shape, indices, updates));

  // Check that we have enough index space
  const int64 N_big = indices.NumElements();
  OP_REQUIRES(
      c, N_big <= std::numeric_limits<Index>::max(),
      errors::InvalidArgument("indices has too many elements for ",
                              DataTypeString(DataTypeToEnum<Index>::v()),
                              " indexing: ", N_big, " > ",
                              std::numeric_limits<Index>::max()));
  OP_REQUIRES(
      c, params_shape.dim_size(0) <= std::numeric_limits<Index>::max(),
      errors::InvalidArgument("params_shape[0] too large for ",
                              DataTypeString(DataTypeToEnum<Index>::v()),
                              " indexing: ", params_shape.dim_size(0), " > ",
                              std::numeric_limits<Index>::max()));

  // Calculate the number of dimensions in indices
  *slice_dim = (indices_shape.dims() > 1)
                   ? indices_shape.dim_size(indices_shape.dims() - 1)
                   : 1;

  // Calculate the number of elements that make up each slice of our updated
  // tensor. This allows us to work with flattened tensors and copy over whole
  // slices at a time.
  Index total_nd = params_shape.dims();

  int64 slice_size_big = 1;
  for (int64 i = *slice_dim; i < total_nd; ++i) {
    slice_size_big *= params_shape.dim_size(i);
  }

  OP_REQUIRES(c, slice_size_big <= std::numeric_limits<Index>::max(),
              errors::InvalidArgument(
                  "slice size is too large for indexing: ", slice_size_big,
                  " > ", std::numeric_limits<Index>::max()));

  *slice_size = static_cast<Index>(slice_size_big);

  const int64 safe_slice_dim = (*slice_dim < 1) ? 1 : *slice_dim;
  *num_updates = indices_shape.num_elements() / safe_slice_dim;
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
    OP_REQUIRES_OK(c,
                   TensorShapeUtils::MakeShape(vec.data(), vec.size(), &shape));

    int64 slice_dim;
    Index num_updates;
    Index slice_size;
    PrepareAndValidateInputs<Index>(c, shape, indices, updates, &slice_dim,
                                    &num_updates, &slice_size);
    if (!c->status().ok()) return;

    auto indices_flat = indices.flat_inner_dims<Index>();
    auto updates_flat = updates.shaped<T, 2>({num_updates, slice_size});

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, shape, &out));
    functor::SetZeroFunctor<Device, T> fill;
    fill(c->eigen_device<Device>(), out->flat<T>());
    auto output_matrix = out->template shaped<T, 2>(
        {shape.num_elements() / slice_size, slice_size});

    Index bad_i = -1;

    if (shape.num_elements() > 0) {
      switch (slice_dim) {
#define PARAMS_CASE(IXDIM)                                                    \
  case IXDIM: {                                                               \
    typename Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix;      \
    for (int i = 0; i < IXDIM; ++i) {                                         \
      output_shape_prefix[i] = shape.dim_size(i);                             \
    }                                                                         \
    functor::ScatterNdFunctor<Device, T, Index, scatter_nd_op::UpdateOp::ADD, \
                              IXDIM>                                          \
        functor;                                                              \
    bad_i =                                                                   \
        functor(c->eigen_device<Device>(), slice_size, output_shape_prefix,   \
                output_matrix, indices_flat, updates_flat, output_matrix);    \
  } break
        // TODO(simister): Re-enable this once binary size is under control.
        //      PARAMS_CASE(0);
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
                          slice_dim));
      }
    }
    OP_REQUIRES(
        c, bad_i < 0,
        errors::InvalidArgument(
            "Invalid indices: ", SliceDebugString(indices.shape(), bad_i),
            " = [",
            str_util::Join(
                gtl::ArraySlice<Index>(&indices_flat(bad_i, 0), slice_dim),
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

    int64 slice_dim;
    Index num_updates;
    Index slice_size;

    OP_REQUIRES(c, params.IsInitialized(),
                errors::FailedPrecondition("Null ref for params"));
    PrepareAndValidateInputs<Index>(c, params_shape, indices, updates,
                                    &slice_dim, &num_updates, &slice_size);
    if (!c->status().ok()) return;

    auto indices_flat = indices.flat_inner_dims<Index>();
    auto updates_flat = updates.shaped<T, 2>({num_updates, slice_size});
    auto params_matrix = params.template shaped<T, 2>(
        {params_shape.num_elements() / slice_size, slice_size});
    Index bad_i = -1;
    c->forward_ref_input_to_ref_output(0, 0);

    switch (slice_dim) {
#define PARAMS_CASE(IXDIM)                                                  \
  case IXDIM: {                                                             \
    typename Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix;    \
    for (int i = 0; i < IXDIM; ++i) {                                       \
      output_shape_prefix[i] = params_shape.dim_size(i);                    \
    }                                                                       \
    functor::ScatterNdFunctor<Device, T, Index, op, IXDIM> functor;         \
    bad_i =                                                                 \
        functor(c->eigen_device<Device>(), slice_size, output_shape_prefix, \
                params_matrix, indices_flat, updates_flat, params_matrix);  \
  } break
      // TODO(simister): Re-enable this once binary size is under control.
      //      PARAMS_CASE(0);
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
                        slice_dim));
    }
    OP_REQUIRES(
        c, bad_i < 0,
        errors::InvalidArgument(
            "Invalid indices: ", SliceDebugString(indices.shape(), bad_i),
            " = [",
            str_util::Join(
                gtl::ArraySlice<Index>(&indices_flat(bad_i, 0), slice_dim),
                ", "),
            "] is not in [0, ", params.dim_size(0), ")"));
  }
};

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
                                    scatter_nd_op::UpdateOp::SUB);
// TODO(simister): Find a way to reduce amount of templated generated code
// to reduce build size, then re-enable these additional operations.
// REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdMul",
//                                   scatter_nd_op::UpdateOp::MUL);
// REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdDiv",
//                                   scatter_nd_op::UpdateOp::DIV);

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
// TODO(simister): Re-enable all types after binary size is under control.
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_UPDATE_CPU);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_CPU);

// Registers GPU kernels.
#if GOOGLE_CUDA

#define REGISTER_SCATTER_ND_ADD_SUB_GPU(type) \
  REGISTER_SCATTER_ND_ADD_SUB(type, GPU);

#define REGISTER_SCATTER_ND_UPDATE_GPU(type) \
  REGISTER_SCATTER_ND_UPDATE(type, GPU);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ND_ADD_SUB_GPU);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ND_UPDATE_GPU);

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, IXDIM)           \
  template <>                                                           \
  Index ScatterNdFunctor<GPUDevice, T, Index, op, IXDIM>::operator()(   \
      const GPUDevice& d, const Index slice_size,                       \
      const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix, \
      typename TTypes<T, 2>::Tensor Tparams,                            \
      typename TTypes<Index, 2>::ConstTensor Tindices,                  \
      typename TTypes<T, 2>::ConstTensor Tupdates,                      \
      typename TTypes<T, 2>::Tensor Toutput);                           \
  extern template struct ScatterNdFunctor<GPUDevice, T, Index, op, IXDIM>;

#define DECLARE_GPU_SPECS_INDEX_OP(T, Index, op)     \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 1); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 2); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 3); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 4); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 5)

#define DECLARE_GPU_SPECS_INDEX(T, Index)                                \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::ASSIGN); \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::ADD);    \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::SUB)

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
#undef DECLARE_GPU_SPECS_INDEX_OP
}  // namespace functor

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

}  // namespace tensorflow
