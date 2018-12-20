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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/scatter_nd_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

#ifdef TENSORFLOW_USE_SYCL
#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#endif  // TENSORFLOW_USE_SYCL

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

// Returns true if the three tensors have valid number of elements
// If shape_input has 0 elements, then we need to have indices and updates with
// exactly 0 elements too, otherwise we should error. If indices has 0 elements
// then updates should also have 0 elements, otherwise we should error.
bool ValidEmptyOutputShape(int64 num_inputs, int64 num_indices,
                           int64 num_updates) {
  if (num_indices == 0 && num_updates == 0) {
    return true;  // regardless of num_inputs ?= 0, covers both cases
  }
  // now we want all 3 tensors to have values
  return (num_inputs != 0 && num_indices != 0 && num_updates != 0);
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

    OP_REQUIRES(c, indices.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Indices shape must have rank at least one. Found:",
                    indices.shape().DebugString()));
    OP_REQUIRES(c, updates.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Updates shape must have rank at least one. Found:",
                    updates.shape().DebugString()));

    auto vec = shape_input.flat<Index>();
    TensorShape shape;
    OP_REQUIRES_OK(c,
                   TensorShapeUtils::MakeShape(vec.data(), vec.size(), &shape));

    OP_REQUIRES(c,
                ValidEmptyOutputShape(shape_input.NumElements(),
                                      indices.shape().num_elements(),
                                      updates.shape().num_elements()),
                errors::InvalidArgument(
                    "Indices and updates specified for empty output shape"));

    const int64 outer_dims = indices.shape().dims() - 1;

    for (int i = 0; i < outer_dims; ++i) {
      OP_REQUIRES(c, indices.shape().dim_size(i) == updates.shape().dim_size(i),
                  errors::InvalidArgument(
                      "Outer dimensions of indices and update must match. "
                      "Indices shape: ",
                      indices.shape().DebugString(),
                      ", updates shape:", updates.shape().DebugString()));
    }

    const int64 ix = indices.shape().dim_size(outer_dims);
    OP_REQUIRES(
        c, updates.shape().dims() - outer_dims == shape.dims() - ix,
        errors::InvalidArgument("Inner dimensions of output shape must match "
                                "inner dimensions of updates shape. Output: ",
                                shape.DebugString(),
                                " updates: ", updates.shape().DebugString()));
    for (int i = 0; i + outer_dims < updates.shape().dims(); ++i) {
      OP_REQUIRES(
          c, updates.shape().dim_size(i + outer_dims) == shape.dim_size(ix + i),
          errors::InvalidArgument(
              "The inner ", shape.dims() - ix,
              " dimensions of output.shape=", shape.DebugString(),
              " must match the inner ", updates.shape().dims() - outer_dims,
              " dimensions of updates.shape=", updates.shape().DebugString()));
    }
    OP_REQUIRES(c, shape_input.dims() == 1,
                errors::InvalidArgument("Shape must be a vector"));

    Tensor out;
    OP_REQUIRES_OK(
        c, functor::DoScatterNd<Device, T, Index, scatter_nd_op::UpdateOp::ADD>(
               c, indices, updates, shape, &out, true /*allocate*/));
    c->set_output(0, out);
  }
};

template <typename Device, typename T, typename Index,
          scatter_nd_op::UpdateOp op>
class TensorScatterOp : public OpKernel {
 public:
  explicit TensorScatterOp(OpKernelConstruction* c) : OpKernel(c) {
    const DataType dt = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<Index>::v();
    OP_REQUIRES_OK(c, c->MatchSignature({dt, index_t, dt}, {dt}));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& input = c->input(0);
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);

    OP_REQUIRES(c, indices.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Indices shape must have rank at least one. Found:",
                    indices.shape().DebugString()));
    OP_REQUIRES(c, updates.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Updates shape must have rank at least one. Found:",
                    updates.shape().DebugString()));

    TensorShape shape = input.shape();

    OP_REQUIRES(c,
                ValidEmptyOutputShape(shape.num_elements(),
                                      indices.shape().num_elements(),
                                      updates.shape().num_elements()),
                errors::InvalidArgument(
                    "Indices and updates specified for empty output shape"));

    const int64 outer_dims = indices.shape().dims() - 1;

    for (int i = 0; i < outer_dims; ++i) {
      OP_REQUIRES(c, indices.shape().dim_size(i) == updates.shape().dim_size(i),
                  errors::InvalidArgument(
                      "Outer dimensions of indices and update must match. "
                      "Indices shape: ",
                      indices.shape().DebugString(),
                      ", updates shape:", updates.shape().DebugString()));
    }

    const int64 ix = indices.shape().dim_size(outer_dims);
    OP_REQUIRES(
        c, updates.shape().dims() - outer_dims == shape.dims() - ix,
        errors::InvalidArgument("Inner dimensions of output shape must match "
                                "inner dimensions of updates shape. Output: ",
                                shape.DebugString(),
                                " updates: ", updates.shape().DebugString()));
    for (int i = 0; i + outer_dims < updates.shape().dims(); ++i) {
      OP_REQUIRES(
          c, updates.shape().dim_size(i + outer_dims) == shape.dim_size(ix + i),
          errors::InvalidArgument(
              "The inner ", shape.dims() - ix,
              " dimensions of output.shape=", shape.DebugString(),
              " must match the inner ", updates.shape().dims() - outer_dims,
              " dimensions of updates.shape=", updates.shape().DebugString()));
    }

    std::unique_ptr<Tensor> forwarded_input = c->forward_input(
        2, 0, input.dtype(), shape, DEVICE_MEMORY, AllocatorAttributes());

    if (forwarded_input == nullptr) {
      // We were not able to forward the input, so we deep copy the tensor and
      // set the output.
      Tensor* out;
      OP_REQUIRES_OK(c, c->allocate_output(0, input.shape(), &out));

      OP_REQUIRES_OK(c, tensorflow::functor::DoCopy(c->eigen_device<Device>(),
                                                    input, out));
      OP_REQUIRES_OK(c,
                     functor::DoScatterNd<Device, T, Index, op>(
                         c, indices, updates, shape, out, false /*allocate*/));
    } else {
      // Output forwarded, so simply perform the scatter.
      OP_REQUIRES_OK(c, functor::DoScatterNd<Device, T, Index, op>(
                            c, indices, updates, shape, forwarded_input.get(),
                            false /*allocate*/));
    }
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
    dtype_ = c->input_type(0);
    if (c->input_type(0) == DT_RESOURCE) {
      // TODO(apassos): what to validate here?
    } else if (IsRefType(c->input_type(0))) {
      OP_REQUIRES_OK(c, c->MatchSignature({dt_ref, index_t, dt}, {dt_ref}));
      OP_REQUIRES_OK(c, c->GetAttr("use_locking", &use_exclusive_lock_));
    } else {
      OP_REQUIRES_OK(c, c->MatchSignature({dt, index_t, dt}, {dt}));
      use_exclusive_lock_ = false;
    }
  }

  void Compute(OpKernelContext* c) override {
    if (dtype_ == DT_RESOURCE) {
      Var* v;
      OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
      core::ScopedUnref scoped_unref(v);
      OP_REQUIRES_OK(c, EnsureSparseVariableAccess<Device, T>(c, v));
      mutex_lock m(*v->mu());
      DoCompute(c);
    } else if (use_exclusive_lock_) {
      // If we're here, it means the input type is a ref.
      DCHECK(IsRefType(c->input_dtype(0)));
      // Hold mutex while we apply updates
      mutex_lock l(*c->input_ref_mutex(0));
      DoCompute(c);
    } else {
      DoCompute(c);
    }
  }

 private:
  DataType dtype_;
  bool use_exclusive_lock_;

  void DoCompute(OpKernelContext* c) {
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);
    Tensor params;
    TensorShape params_shape;

    if (dtype_ == DT_RESOURCE) {
      Var* v;
      OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
      Tensor* t = v->tensor();
      params = *t;
      params_shape = params.shape();
    } else if (IsRefType(c->input_dtype(0))) {
      params = c->mutable_input(0, use_exclusive_lock_);
      params_shape = params.shape();
      c->forward_ref_input_to_ref_output(0, 0);
      OP_REQUIRES(c, params.IsInitialized(),
                  errors::FailedPrecondition("Null ref for params"));
    } else {
      Tensor* params_ptr;
      params_shape = c->input(0).shape();
      if (!c->forward_input_to_output_with_shape(0, 0, params_shape,
                                                 &params_ptr)) {
        // We weren't able to forward the input to output, so just
        // allocate a new output tensor and copy the values over.
        OP_REQUIRES_OK(c, c->allocate_output(0, params_shape, &params_ptr));
        params = *params_ptr;
        functor::DenseUpdate<Device, T, ASSIGN> copy;
        const Tensor& input_copy = c->input(0);
        copy(c->eigen_device<Device>(), params.flat<T>(), input_copy.flat<T>());
      } else {
        params = *params_ptr;
      }
    }

    OP_REQUIRES_OK(
        c, functor::DoScatterNd<Device, T, Index, op>(
               c, indices, updates, params_shape, &params, false /*allocate*/));
  }
};

#define REGISTER_SCATTER_ND_KERNEL_INDEX(type, index_type, dev, name) \
  REGISTER_KERNEL_BUILDER(Name(name)                                  \
                              .Device(DEVICE_##dev)                   \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<index_type>("Tindices") \
                              .HostMemory("shape"),                   \
                          ScatterNdOp<dev##Device, type, index_type>)

#define REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX(type, index_type, dev, name, \
                                                op)                          \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name(name)                                                             \
          .Device(DEVICE_##dev)                                              \
          .TypeConstraint<type>("T")                                         \
          .TypeConstraint<index_type>("Tindices"),                           \
      ScatterNdUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX(type, index_type, \
                                                         dev, name, op)    \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name(name)                                                           \
          .Device(DEVICE_##dev)                                            \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<index_type>("Tindices")                          \
          .HostMemory("ref"),                                              \
      ScatterNdUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_SCATTER_ND_KERNEL(type, dev, name)         \
  REGISTER_SCATTER_ND_KERNEL_INDEX(type, int32, dev, name); \
  REGISTER_SCATTER_ND_KERNEL_INDEX(type, int64, dev, name)

#define REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, name, op)         \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int32, dev, name, op); \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int64, dev, name, op)

#define REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(type, dev, name, op)    \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int32, dev, name, \
                                                   op);                    \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int64, dev, name, op)

#define REGISTER_SCATTER_ND_ADD_SUB(type, dev)                            \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdAdd",            \
                                    scatter_nd_op::UpdateOp::ADD);        \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdNonAliasingAdd", \
                                    scatter_nd_op::UpdateOp::ADD);        \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdSub",            \
                                    scatter_nd_op::UpdateOp::SUB);        \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(                             \
      type, dev, "ResourceScatterNdAdd", scatter_nd_op::UpdateOp::ADD);

#define REGISTER_SCATTER_ND(type, dev) \
  REGISTER_SCATTER_ND_KERNEL(type, dev, "ScatterNd");

#define REGISTER_SCATTER_ND_UPDATE(type, dev)                         \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdUpdate",     \
                                    scatter_nd_op::UpdateOp::ASSIGN); \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(                         \
      type, dev, "ResourceScatterNdUpdate", scatter_nd_op::UpdateOp::ASSIGN);

// Registers CPU kernels.
#define REGISTER_SCATTER_ND_ADD_SUB_CPU(type) \
  REGISTER_SCATTER_ND_ADD_SUB(type, CPU);

#define REGISTER_SCATTER_ND_UPDATE_CPU(type) \
  REGISTER_SCATTER_ND_UPDATE(type, CPU);

#define REGISTER_SCATTER_ND_CPU(type) REGISTER_SCATTER_ND(type, CPU);
#define REGISTER_SCATTER_ND_GPU(type) REGISTER_SCATTER_ND(type, GPU);

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_ADD_SUB_CPU);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_UPDATE_CPU);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_CPU);
TF_CALL_string(REGISTER_SCATTER_ND_CPU);
TF_CALL_bool(REGISTER_SCATTER_ND_ADD_SUB_CPU);
TF_CALL_bool(REGISTER_SCATTER_ND_UPDATE_CPU);
TF_CALL_bool(REGISTER_SCATTER_ND_CPU);

#define REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, index_type, \
                                                          dev)              \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterUpdate")                       \
                              .Device(DEVICE_##dev)                         \
                              .TypeConstraint<type>("T")                    \
                              .TypeConstraint<index_type>("Tindices"),      \
                          TensorScatterOp<dev##Device, type, index_type,    \
                                          scatter_nd_op::UpdateOp::ASSIGN>)

#define REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, index_type, dev) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterAdd")                            \
                              .Device(DEVICE_##dev)                           \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<index_type>("Tindices"),        \
                          TensorScatterOp<dev##Device, type, index_type,      \
                                          scatter_nd_op::UpdateOp::ADD>)

#define REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, index_type, dev) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterSub")                            \
                              .Device(DEVICE_##dev)                           \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<index_type>("Tindices"),        \
                          TensorScatterOp<dev##Device, type, index_type,      \
                                          scatter_nd_op::UpdateOp::SUB>)

#define REGISTER_SCATTER_ND_TENSOR_UPDATE_CPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, int32, CPU); \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, int64, CPU);

#define REGISTER_SCATTER_ND_TENSOR_ADD_CPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, int32, CPU); \
  REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, int64, CPU);

#define REGISTER_SCATTER_ND_TENSOR_SUB_CPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, int32, CPU); \
  REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, int64, CPU);

#define REGISTER_SCATTER_ND_TENSOR_CPU(type)   \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_CPU(type); \
  REGISTER_SCATTER_ND_TENSOR_ADD_CPU(type);    \
  REGISTER_SCATTER_ND_TENSOR_SUB_CPU(type);

// Register TensorScatterUpdate/Add/Sub for all number types.
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_TENSOR_CPU);
// Register only TensorScatterUpdate for string/bool types as well.
TF_CALL_string(REGISTER_SCATTER_ND_TENSOR_UPDATE_CPU);
TF_CALL_bool(REGISTER_SCATTER_ND_TENSOR_UPDATE_CPU);

#undef REGISTER_SCATTER_ND_TENSOR_CPU

// Registers GPU kernels.
#if GOOGLE_CUDA

#define REGISTER_SCATTER_ND_ADD_SUB_GPU(type) \
  REGISTER_SCATTER_ND_ADD_SUB(type, GPU);

#define REGISTER_SCATTER_ND_UPDATE_GPU(type) \
  REGISTER_SCATTER_ND_UPDATE(type, GPU);

#define REGISTER_SCATTER_ND_ALL_GPU(type) \
  REGISTER_SCATTER_ND_ADD_SUB_GPU(type);  \
  REGISTER_SCATTER_ND_UPDATE_GPU(type);   \
  REGISTER_SCATTER_ND_GPU(type);

TF_CALL_int32(REGISTER_SCATTER_ND_ALL_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ND_ALL_GPU);
TF_CALL_complex64(REGISTER_SCATTER_ND_ALL_GPU);
TF_CALL_complex128(REGISTER_SCATTER_ND_ALL_GPU);

#undef REGISTER_SCATTER_ND_ALL_GPU

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SCATTER_ND_ADD_SUB_SYCL(type) \
  REGISTER_SCATTER_ND_ADD_SUB(type, SYCL);

#define REGISTER_SCATTER_ND_UPDATE_SYCL(type) \
  REGISTER_SCATTER_ND_UPDATE(type, SYCL);

TF_CALL_int32(REGISTER_SCATTER_ND_ADD_SUB_SYCL);
TF_CALL_int32(REGISTER_SCATTER_ND_UPDATE_SYCL);
TF_CALL_bool(REGISTER_SCATTER_ND_UPDATE_SYCL);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ND_ADD_SUB_SYCL);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ND_UPDATE_SYCL);
#undef REGISTER_SCATTER_ND_ADD_SUB_SYCL
#undef REGISTER_SCATTER_ND_UPDATE_SYCL
#endif  // TENSORFLOW_USE_SYCL

#define REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, int64, GPU);

#define REGISTER_SCATTER_ND_TENSOR_ADD_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, int64, GPU);

#define REGISTER_SCATTER_ND_TENSOR_SUB_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, int64, GPU);

#define REGISTER_SCATTER_ND_TENSOR_GPU(type)   \
  REGISTER_SCATTER_ND_TENSOR_ADD_GPU(type);    \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU(type); \
  REGISTER_SCATTER_ND_TENSOR_SUB_GPU(type);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ND_TENSOR_GPU);

#undef REGISTER_SCATTER_ND_ADD
#undef REGISTER_SCATTER_ND_ADD_SUB
#undef REGISTER_SCATTER_ND_ADD_SUB_CPU
#undef REGISTER_SCATTER_ND_ADD_SUB_GPU
#undef REGISTER_SCATTER_ND_UPDATE
#undef REGISTER_SCATTER_ND_UPDATE_CPU
#undef REGISTER_SCATTER_ND_UPDATE_GPU
#undef REGISTER_SCATTER_ND_KERNEL
#undef REGISTER_SCATTER_ND_KERNEL_INDEX
#undef REGISTER_SCATTER_ND_TENSOR_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_CPU
#undef REGISTER_SCATTER_ND_TENSOR_GPU
#undef REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU
#undef REGISTER_SCATTER_ND_TENSOR_ADD_GPU
#undef REGISTER_SCATTER_ND_TENSOR_SUB_GPU
#undef REGISTER_SCATTER_ND_TENSOR_GPU

#endif  // GOOGLE_CUDA

namespace functor {
// Check whether updates.shape = indices.shape[:batch_dim] +
// params_shape[slice_dim:]
Status ValidateUpdateShape(const TensorShape& params_shape,
                           const Tensor& indices, const Tensor& updates) {
  const int64 slice_dim =
      (indices.dims() > 1) ? indices.dim_size(indices.dims() - 1) : 1;
  const int64 batch_dim = (indices.dims() > 1) ? indices.dims() - 1 : 1;

  auto shape_err = [&]() {
    return errors::InvalidArgument(
        "Must have updates.shape = indices.shape[:batch_dim] + ",
        "params_shape[slice_dim:], got updates.shape: ",
        updates.shape().DebugString(),
        ", indices.shape: ", indices.shape().DebugString(),
        ", params_shape: ", params_shape.DebugString(),
        ", slice_dim: ", slice_dim, ", and batch_dim: ", batch_dim);
  };

  if (updates.dims() < batch_dim) return shape_err();
  if (params_shape.dims() < slice_dim + (updates.dims() - batch_dim)) {
    return shape_err();
  }
  if (updates.dims() != batch_dim + params_shape.dims() - slice_dim) {
    return shape_err();
  }
  for (int d = 0; d < batch_dim; ++d) {
    if (updates.dim_size(d) != indices.dim_size(d)) return shape_err();
  }
  for (int d = 0; d < updates.dims() - batch_dim; ++d) {
    if (updates.dim_size(d + batch_dim) !=
        params_shape.dim_size(d + slice_dim)) {
      return shape_err();
    }
  }
  return Status::OK();
}

template <typename Index>
Status PrepareAndValidateInputs(const TensorShape& params_shape,
                                const Tensor& indices, const Tensor& updates,
                                int64* slice_dim, Index* num_updates,
                                Index* slice_size) {
  const TensorShape& indices_shape(indices.shape());
  const TensorShape& updates_shape(updates.shape());

  if (!TensorShapeUtils::IsVectorOrHigher(params_shape)) {
    return errors::InvalidArgument("Output must be at least 1-D, ",
                                   "got shape: ", params_shape.DebugString());
  }

  if (!ValidEmptyOutputShape(params_shape.num_elements(),
                             indices_shape.num_elements(),
                             updates_shape.num_elements())) {
    return errors::InvalidArgument(
        "Indices and updates specified for empty output.  indices shape: ",
        indices.shape().DebugString());
  }

  if (updates.dim_size(0) != indices.dim_size(0)) {
    return errors::InvalidArgument(
        "The outermost dimension of updates and indices ",
        "must match. Got indices.shape ", indices_shape.DebugString(),
        ", updates.shape ", updates_shape.DebugString());
  }
  TF_RETURN_IF_ERROR(ValidateUpdateShape(params_shape, indices, updates));

  // Check that we have enough index space
  const int64 N_big = indices.NumElements();
  if (N_big > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument("indices has too many elements for ",
                                   DataTypeString(DataTypeToEnum<Index>::v()),
                                   " indexing: ", N_big, " > ",
                                   std::numeric_limits<Index>::max());
  }
  if (params_shape.dim_size(0) > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument("params_shape[0] too large for ",
                                   DataTypeString(DataTypeToEnum<Index>::v()),
                                   " indexing: ", params_shape.dim_size(0),
                                   " > ", std::numeric_limits<Index>::max());
  }

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

  if (slice_size_big > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument(
        "slice size is too large for indexing: ", slice_size_big, " > ",
        std::numeric_limits<Index>::max());
  }

  *slice_size = static_cast<Index>(slice_size_big);

  const int64 safe_slice_dim = (*slice_dim < 1) ? 1 : *slice_dim;
  *num_updates = indices_shape.num_elements() / safe_slice_dim;

  return Status::OK();
}

template <typename Device, typename Index>
class IndexFlattener {
 public:
  inline typename TTypes<Index, 2>::ConstTensor operator()(
      OpKernelContext*, const Tensor& indices) {
    return indices.flat_inner_dims<Index>();
  }
};

#ifdef TENSORFLOW_USE_SYCL
template <typename Index>
class IndexFlattener<SYCLDevice, Index> {
 public:
  IndexFlattener() { indices_host_ = nullptr; }
  ~IndexFlattener() { delete[] indices_host_; }

  inline typename TTypes<Index, 2>::ConstTensor operator()(
      OpKernelContext* c, const Tensor& indices) {
    size_t num_indices = indices.NumElements();
    indices_host_ = new Index[num_indices];
    auto device = c->eigen_sycl_device();
    auto size = sizeof(Index) * num_indices;
    auto src_ptr = GetBase(&indices);
    device.memcpyDeviceToHost(indices_host_, static_cast<const Index*>(src_ptr),
                              size);
    return typename TTypes<Index, 2>::ConstTensor(
        indices_host_, indices.shape().AsEigenDSizes<2>());
  }

 private:
  Index* indices_host_;
};
#endif

template <typename Device, typename T, typename Index,
          scatter_nd_op::UpdateOp Op>
Status DoScatterNd(OpKernelContext* c, const Tensor& indices,
                   const Tensor& updates, const TensorShape& shape, Tensor* out,
                   bool allocate) {
  int64 slice_dim;
  Index num_updates;
  Index slice_size;
  TF_RETURN_IF_ERROR(PrepareAndValidateInputs<Index>(
      shape, indices, updates, &slice_dim, &num_updates, &slice_size));

  IndexFlattener<Device, Index> index_flattener;
  auto indices_flat = index_flattener(c, indices);
  auto updates_flat = updates.shaped<T, 2>({num_updates, slice_size});

  if (allocate) {
    TF_RETURN_IF_ERROR(c->allocate_temp(DataTypeToEnum<T>::value, shape, out));
  } else {
    CHECK_NOTNULL(out);
  }

  if (shape.num_elements() == 0) {
    return Status::OK();
  }

  if (allocate) {
    // Brand new tensor, zero it out.
    functor::SetZeroFunctor<Device, T> fill;
    fill(c->eigen_device<Device>(), out->flat<T>());
  }
  auto output_matrix =
      out->shaped<T, 2>({shape.num_elements() / slice_size, slice_size});

  Index bad_i = -1;

  if (shape.num_elements() > 0) {
    switch (slice_dim) {
#define PARAMS_CASE(IXDIM)                                                  \
  case IXDIM: {                                                             \
    typename Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix;    \
    for (int i = 0; i < IXDIM; ++i) {                                       \
      output_shape_prefix[i] = shape.dim_size(i);                           \
    }                                                                       \
    functor::ScatterNdFunctor<Device, T, Index, Op, IXDIM> functor;         \
    bad_i =                                                                 \
        functor(c->eigen_device<Device>(), slice_size, output_shape_prefix, \
                output_matrix, indices_flat, updates_flat, output_matrix);  \
  } break
      // TODO(simister): Re-enable this once binary size is under control.
      //      PARAMS_CASE(0);
      PARAMS_CASE(1);
      PARAMS_CASE(2);
      PARAMS_CASE(3);
      PARAMS_CASE(4);
      PARAMS_CASE(5);
      PARAMS_CASE(6);
      PARAMS_CASE(7);
#undef PARAMS_CASE
      default:
        return errors::InvalidArgument(
            "Only indices.shape[-1] values between 1 and 5 "
            "are currently supported.  Requested rank: ",
            slice_dim);
    }
  }
  if (bad_i >= 0) {
    auto slice_shape = indices.shape();
    slice_shape.RemoveLastDims(1);
    return errors::InvalidArgument(
        "indices", SliceDebugString(slice_shape, bad_i), " = [",
        str_util::Join(
            gtl::ArraySlice<Index>(&indices_flat(bad_i, 0), slice_dim), ", "),
        "] does not index into shape ", shape.DebugString());
  }
  return Status::OK();
}
}  // namespace functor

#ifdef GOOGLE_CUDA
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
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 5); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 6); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 7);

#define DECLARE_GPU_SPECS_INDEX(T, Index)                                \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::ASSIGN); \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::ADD);    \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::SUB)

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

TF_CALL_int32(DECLARE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_complex64(DECLARE_GPU_SPECS);
TF_CALL_complex128(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
#undef DECLARE_GPU_SPECS_INDEX_OP

}  // namespace functor

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
