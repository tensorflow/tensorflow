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

// See core/ops/sparse_ops.cc for documentation.
//
// NOTE: the operations in this file only are suitable for execution
// on CPUs.

#define EIGEN_USE_THREADS

#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/kernels/sparse_to_dense_op_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

namespace {

Status CheckSparseToDenseShapes(const Tensor& indices,
                                const Tensor& output_shape,
                                const Tensor& sparse_values,
                                const Tensor& default_value) {
  // sparse_indices
  if (indices.dims() > 2) {
    return errors::InvalidArgument(
        "sparse_indices should be a scalar, vector, or matrix, "
        "got shape ",
        indices.shape().DebugString());
  }
  const int64_t num_elems = indices.dims() > 0 ? indices.dim_size(0) : 1;
  const int64_t num_dims = indices.dims() > 1 ? indices.dim_size(1) : 1;

  // output_shape
  if (!TensorShapeUtils::IsVector(output_shape.shape())) {
    return errors::InvalidArgument("output_shape must be rank 1, got shape ",
                                   output_shape.shape().DebugString());
  }

  if (output_shape.NumElements() != num_dims) {
    return errors::InvalidArgument(
        "output_shape has incorrect number of elements: ",
        output_shape.NumElements(), " should be: ", num_dims);
  }

  // sparse_values
  const int64_t num_values = sparse_values.NumElements();
  if (sparse_values.dims() != 0 &&
      (sparse_values.dims() != 1 || num_values != num_elems)) {
    return errors::InvalidArgument("sparse_values has incorrect shape ",
                                   sparse_values.shape().DebugString(),
                                   ", should be [] or [", num_elems, "]");
  }

  // default_value
  if (!TensorShapeUtils::IsScalar(default_value.shape())) {
    return errors::InvalidArgument("default_value should be a scalar.");
  }
  return Status::OK();
}

}  // end namespace

// Operator to convert sparse representations to dense.
template <typename T, typename Index>
class SparseToDense : public OpKernel {
 public:
  explicit SparseToDense(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("validate_indices", &validate_indices_));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& indices = c->input(0);
    const Tensor& output_shape = c->input(1);
    const Tensor& sparse_values = c->input(2);
    const Tensor& default_value = c->input(3);
    OP_REQUIRES_OK(c, CheckSparseToDenseShapes(indices, output_shape,
                                               sparse_values, default_value));

    const int64_t num_elems = indices.dims() > 0 ? indices.dim_size(0) : 1;
    const int64_t num_dims = indices.dims() > 1 ? indices.dim_size(1) : 1;

    auto output_shape_vec = output_shape.flat<Index>();
    TensorShape output_tensor_shape;
    OP_REQUIRES_OK(c, TensorShapeUtils::MakeShape(output_shape_vec.data(),
                                                  output_shape_vec.size(),
                                                  &output_tensor_shape));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_tensor_shape, &output));

    const Tensor* indices_shaped;
    std::unique_ptr<Tensor> indices_shaped_holder;
    if (indices.dtype() == DT_INT64 && indices.dims() == 2) {
      indices_shaped = &indices;
    } else {
      TensorShape ix_shape({num_elems, num_dims});
      indices_shaped_holder = MakeUnique<Tensor>(DT_INT64, ix_shape);
      indices_shaped = indices_shaped_holder.get();
      if (indices.dtype() == DT_INT64) {
        CHECK(indices_shaped_holder->CopyFrom(indices, ix_shape));
      } else {
        indices_shaped_holder->matrix<int64_t>() =
            indices.shaped<Index, 2>(ix_shape.dim_sizes())
                .template cast<int64_t>();
      }
    }

    // If we received a scalar, we'll need to create a new
    // tensor with copies of the values as a vec.
    const Tensor* sparse_values_b;
    std::unique_ptr<Tensor> sparse_values_b_holder;

    if (TensorShapeUtils::IsScalar(sparse_values.shape())) {
      sparse_values_b_holder = MakeUnique<Tensor>(DataTypeToEnum<T>::value,
                                                  TensorShape({num_elems}));
      sparse_values_b = sparse_values_b_holder.get();
      sparse_values_b_holder->vec<T>().setConstant(sparse_values.scalar<T>()());
    } else {
      sparse_values_b = &sparse_values;
    }

    // Assume SparseTensor is lexicographically sorted.
    gtl::InlinedVector<int64_t, 8> order(output->shape().dims());
    std::iota(order.begin(), order.end(), 0);
    sparse::SparseTensor st;
    OP_REQUIRES_OK(
        c, sparse::SparseTensor::Create(*indices_shaped, *sparse_values_b,
                                        output->shape(), order, &st));

    if (validate_indices_) {
      OP_REQUIRES_OK(c, st.IndicesValid());
    }

    output->flat<T>().setConstant(default_value.scalar<T>()());
    OP_REQUIRES(c, st.template ToDense<T>(output, false /* initialize */),
                errors::InvalidArgument(
                    "Indices are not valid (out of bounds).  Shape: ",
                    output->shape().DebugString()));
  }

 private:
  bool validate_indices_;
};

#define REGISTER_KERNELS(type, index_type)                             \
  REGISTER_KERNEL_BUILDER(Name("SparseToDense")                        \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          SparseToDense<type, index_type>);

#define REGISTER_KERNELS_ALL(type) \
  REGISTER_KERNELS(type, int32);   \
  REGISTER_KERNELS(type, int64_t);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL);
REGISTER_KERNELS_ALL(bool);
REGISTER_KERNELS_ALL(tstring);

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T, typename Index>
class SparseToDenseGPU : public AsyncOpKernel {
 public:
  explicit SparseToDenseGPU(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("validate_indices", &validate_indices_));
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) final {
    auto* stream = c->op_device_context()->stream();
    OP_REQUIRES_ASYNC(c, stream, errors::Internal("No GPU stream available."),
                      done);

    const Tensor& indices = c->input(0);
    const Tensor& output_shape = c->input(1);
    const Tensor& sparse_values = c->input(2);
    const Tensor& default_value = c->input(3);
    OP_REQUIRES_OK_ASYNC(c,
                         CheckSparseToDenseShapes(indices, output_shape,
                                                  sparse_values, default_value),
                         done);

    auto output_shape_vec = output_shape.flat<Index>();
    TensorShape output_tensor_shape;
    OP_REQUIRES_OK_ASYNC(c,
                         TensorShapeUtils::MakeShape(output_shape_vec.data(),
                                                     output_shape_vec.size(),
                                                     &output_tensor_shape),
                         done);
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(0, output_tensor_shape, &output),
                         done);

    Tensor output_shape_tensor;
    OP_REQUIRES_OK_ASYNC(
        c,
        c->allocate_temp(DataTypeToEnum<Index>::value,
                         {output_shape_vec.size()}, &output_shape_tensor),
        done);
    auto output_shape_data =
        AsDeviceMemory(output_shape_tensor.template flat<Index>().data(),
                       output_shape_tensor.template flat<Index>().size());
    OP_REQUIRES_ASYNC(
        c,
        stream
            ->ThenMemcpy(&output_shape_data, output_shape_vec.data(),
                         output_shape_tensor.NumElements() * sizeof(Index))
            .ok(),
        errors::InvalidArgument(
            "failed to copy output_shape vector from host to "
            "device in SparseToDenseOp"),
        done);

    functor::LaunchSparseToDense<T, Index>()(
        c, done, this, validate_indices_, indices, sparse_values,
        output_shape_tensor, default_value.scalar<T>()(), output);
  }

 private:
  bool validate_indices_;
};

// TODO(b/184077412): SparseToDense causes an illegal access error.

#define REGISTER_GPU_KERNELS(type, index_type)                         \
  REGISTER_KERNEL_BUILDER(Name("SparseToDense")                        \
                              .Device(DEVICE_GPU)                      \
                              .HostMemory("default_value")             \
                              .HostMemory("output_shape")              \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          SparseToDenseGPU<type, index_type>);

#define REGISTER_GPU_KERNELS_ALL(type) \
  REGISTER_GPU_KERNELS(type, int32);   \
  REGISTER_GPU_KERNELS(type, int64_t);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS_ALL);
TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_KERNELS_ALL)
REGISTER_GPU_KERNELS_ALL(bool)

#undef REGISTER_GPU_KERNELS_ALL
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
