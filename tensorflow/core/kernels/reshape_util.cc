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
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/reshape_util.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <>
struct ReshapeSparseTensorFunctor<CPUDevice> {
  absl::Status operator()(
      OpKernelContext *context, const TensorShape &input_shape,
      const TensorShape &output_shape,
      typename TTypes<int64_t>::ConstMatrix input_indices,
      typename TTypes<int64_t>::Matrix output_indices) const {
    (void)context;  // Unused (only used in GPU implementation)
    const int64_t input_rank = input_shape.dims();
    const int64_t output_rank = output_shape.dims();
    const int64_t nnz = input_indices.dimension(0);
    absl::InlinedVector<int64_t, 8> input_strides(input_rank);
    if (input_rank > 0) {
      input_strides[input_rank - 1] = 1;
      for (int d = input_rank - 2; d >= 0; --d) {
        input_strides[d] = input_strides[d + 1] * input_shape.dim_size(d + 1);
      }
    }

    absl::InlinedVector<int64_t, 8> output_strides(output_rank);
    if (output_rank > 0) {
      output_strides[output_rank - 1] = 1;
      for (int d = output_rank - 2; d >= 0; --d) {
        output_strides[d] =
            output_strides[d + 1] * output_shape.dim_size(d + 1);
      }
    }

    const auto input_dims = input_shape.dim_sizes();
    for (int64_t i = 0; i < nnz; ++i) {
      int64_t id = 0;
      for (int j = 0; j < input_rank; ++j) {
        const int64_t index = input_indices(i, j);
        if (!FastBoundsCheck(index, input_dims[j])) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Input index [", i, ", ", j, "] = ", index,
              " is out of bounds for input shape ", input_shape.DebugString()));
        }
        id += index * input_strides[j];
      }
      for (int j = 0; j < output_rank; ++j) {
        output_indices(i, j) = id / output_strides[j];
        id %= output_strides[j];
      }
    }
    return absl::OkStatus();
  }
};

}  // namespace functor

namespace {

// Validates the inputs, computes the output shape and allocates the outputs.
// If the output indices still need to be computed, sets `*result_indices` to
// the allocated output indices; otherwise leaves it as nullptr.
absl::Status PrepareReshapeSparseTensor(
    OpKernelContext* context, const Tensor& input_indices_in,
    const Tensor& input_shape_in, const Tensor& target_shape_in,
    int output_indices_idx, int output_shape_idx, TensorShape* input_shape,
    TensorShape* output_shape, Tensor** result_indices) {
  *result_indices = nullptr;
  if (!TensorShapeUtils::IsMatrix(input_indices_in.shape())) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input indices should be a matrix but received shape ",
                     input_indices_in.shape().DebugString()));
  }
  if (!TensorShapeUtils::IsVector(input_shape_in.shape())) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input shape should be a vector but received shape ",
                     input_shape_in.shape().DebugString()));
  }
  if (!TensorShapeUtils::IsVector(target_shape_in.shape())) {
    return absl::InvalidArgumentError(
        absl::StrCat("Target shape should be a vector but received shape ",
                     target_shape_in.shape().DebugString()));
  }

  const int64_t output_rank = target_shape_in.NumElements();
  TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape(
      input_shape_in.vec<int64_t>(), input_shape));
  const int64_t dense_size = input_shape->num_elements();
  const int64_t nnz = input_indices_in.shape().dim_size(0);

  // Compute the output shape. Determine product of specified dimensions, and
  // find the index of the unspecified one.
  int64_t product = 1;
  int unknown_index = -1;
  auto target_shape = target_shape_in.vec<int64_t>();
  for (int d = 0; d < output_rank; ++d) {
    const int64_t size = target_shape(d);
    if (size == -1) {
      if (unknown_index != -1) {
        return absl::InvalidArgumentError(
            absl::StrCat("only one output dimension may be -1, not both ",
                         unknown_index, " and ", d));
      }
      unknown_index = d;
      TF_RETURN_IF_ERROR(output_shape->AddDimWithStatus(1));
    } else {
      if (size < 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("size ", d, " must be non-negative, not ", size));
      }
      // AddDimWithStatus checks for overflow, so the multiplication below
      // cannot overflow: `product` always equals output_shape->num_elements().
      TF_RETURN_IF_ERROR(output_shape->AddDimWithStatus(size));
      product *= size;
    }
  }
  if (unknown_index != -1) {
    if (product <= 0) {
      return absl::InvalidArgumentError(
          "reshape cannot infer the missing input size for an empty tensor "
          "unless all specified input sizes are non-zero");
    }
    const int64_t missing = dense_size / product;
    if (product * missing != dense_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input to reshape is a SparseTensor with ", dense_size,
          " dense values, but the requested shape requires a multiple of ",
          product, ". input_shape=", input_shape->DebugString(),
          " output_shape=", output_shape->DebugString()));
    }
    output_shape->set_dim(unknown_index, missing);
  }

  if (output_shape->num_elements() != dense_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input to reshape is a tensor with ", dense_size,
                     " dense values, but the requested shape has ",
                     output_shape->num_elements(),
                     ". input_shape=", input_shape->DebugString(),
                     " output_shape=", output_shape->DebugString()));
  }

  // Optimize for reshaping to the same shape.
  if (*input_shape == *output_shape) {
    context->set_output(output_indices_idx, input_indices_in);
    context->set_output(output_shape_idx, input_shape_in);
    return absl::OkStatus();
  }

  Tensor *result_shape = nullptr;
  TF_RETURN_IF_ERROR(context->allocate_output(
      output_shape_idx, TensorShape({output_rank}), &result_shape));
  auto output_shape_vec = result_shape->vec<int64_t>();
  for (int j = 0; j < output_shape->dims(); ++j) {
    output_shape_vec(j) = output_shape->dim_size(j);
  }

  Tensor* output_indices = nullptr;
  TF_RETURN_IF_ERROR(context->allocate_output(
      output_indices_idx, TensorShape({nnz, output_rank}), &output_indices));
  if (nnz > 0) {
    if (dense_size <= 0 || product <= 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input tensor has ", nnz, " non zero elements but input shape (",
          input_shape->DebugString(), ") or output shape (",
          output_shape->DebugString(), ") is empty"));
    }
    *result_indices = output_indices;
  }
  return absl::OkStatus();
}

}  // namespace

template <typename Device>
void ReshapeSparseTensor(OpKernelContext* context,
                         const Tensor& input_indices_in,
                         const Tensor& input_shape_in,
                         const Tensor& target_shape_in, int output_indices_idx,
                         int output_shape_idx) {
  TensorShape input_shape;
  TensorShape output_shape;
  Tensor* result_indices = nullptr;
  OP_REQUIRES_OK(context,
                 PrepareReshapeSparseTensor(
                     context, input_indices_in, input_shape_in, target_shape_in,
                     output_indices_idx, output_shape_idx, &input_shape,
                     &output_shape, &result_indices));
  if (result_indices == nullptr) return;
  OP_REQUIRES_OK(context, functor::ReshapeSparseTensorFunctor<Device>()(
                              context, input_shape, output_shape,
                              input_indices_in.matrix<int64_t>(),
                              result_indices->matrix<int64_t>()));
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
void ReshapeSparseTensorGpuAsync(OpKernelContext* context,
                                 const Tensor& input_indices_in,
                                 const Tensor& input_shape_in,
                                 const Tensor& target_shape_in,
                                 int output_indices_idx, int output_shape_idx,
                                 std::function<void()> done) {
  TensorShape input_shape;
  TensorShape output_shape;
  Tensor* result_indices = nullptr;
  OP_REQUIRES_OK_ASYNC(
      context,
      PrepareReshapeSparseTensor(context, input_indices_in, input_shape_in,
                                 target_shape_in, output_indices_idx,
                                 output_shape_idx, &input_shape, &output_shape,
                                 &result_indices),
      done);
  if (result_indices == nullptr) {
    done();
    return;
  }

  // The kernel writes the first out-of-bounds row of the input indices to a
  // device scalar, which is then copied to pinned host memory and checked once
  // the stream reaches that point, without blocking the host.
  Tensor first_invalid_row_dev;
  OP_REQUIRES_OK_ASYNC(
      context,
      context->allocate_temp(DT_INT64, TensorShape({}), &first_invalid_row_dev),
      done);
  AllocatorAttributes host_alloc_attr;
  host_alloc_attr.set_on_host(true);
  host_alloc_attr.set_gpu_compatible(true);
  Tensor first_invalid_row_host;
  OP_REQUIRES_OK_ASYNC(
      context,
      context->allocate_temp(DT_INT64, TensorShape({}), &first_invalid_row_host,
                             host_alloc_attr),
      done);

  OP_REQUIRES_OK_ASYNC(
      context,
      functor::ReshapeSparseTensorFunctor<GPUDevice>()(
          context, input_shape, output_shape,
          input_indices_in.matrix<int64_t>(), result_indices->matrix<int64_t>(),
          first_invalid_row_dev.scalar<int64_t>()),
      done);

  se::Stream* stream = context->op_device_context()->stream();
  OP_REQUIRES_ASYNC(context, stream != nullptr,
                    absl::InternalError("No GPU stream available."), done);
  stream_executor::DeviceAddressBase first_invalid_row_dev_mem(
      first_invalid_row_dev.scalar<int64_t>().data(), sizeof(int64_t));
  OP_REQUIRES_OK_ASYNC(
      context,
      stream->Memcpy(first_invalid_row_host.scalar<int64_t>().data(),
                     first_invalid_row_dev_mem, sizeof(int64_t)),
      done);

  const int64_t nnz = input_indices_in.dim_size(0);
  // Capture the temporary tensors by value to keep them alive until the
  // callback runs.
  auto check_indices = [context, first_invalid_row_dev, first_invalid_row_host,
                        input_shape, nnz, done]() {
    // Declared before scoped_activation so that the context is released before
    // done() runs (done() may inline another op kernel).
    auto cleanup = gtl::MakeCleanup([&done]() { done(); });
    se::Stream* stream = context->op_device_context()->stream();
    std::unique_ptr<se::ActivateContext> scoped_activation =
        stream->parent()->Activate();
    const int64_t first_invalid_row =
        first_invalid_row_host.scalar<int64_t>()();
    if (first_invalid_row != nnz) {
      context->SetStatus(absl::InvalidArgumentError(absl::StrCat(
          "Input index row ", first_invalid_row,
          " is out of bounds for input shape ", input_shape.DebugString())));
    }
  };
  context->device()
      ->tensorflow_accelerator_device_info()
      ->event_mgr->ThenExecute(stream, std::move(check_indices));
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EXPLICITLY_INSTANTIATE_FUNCTION(Device)                    \
  template void ReshapeSparseTensor<Device>(                       \
      OpKernelContext * context, const Tensor &input_indices_in,   \
      const Tensor &input_shape_in, const Tensor &target_shape_in, \
      int output_indices_idx, int output_shape_idx)
EXPLICITLY_INSTANTIATE_FUNCTION(CPUDevice);
#undef EXPLICITLY_INSTANTIATE_FUNCTION

}  // namespace tensorflow
