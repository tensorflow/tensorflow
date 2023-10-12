/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_CONCAT_SPLIT_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_CONCAT_SPLIT_UTIL_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/split_lib.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace concat_split_util {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Concatenates 'inputs' into a single tensor along the zeroth dimension.
// Requires that all elements of 'inputs' have element type T. Writes to
// 'output' using 'context' for the allocation to ensure proper device
// placement.
template <typename T>
Status Concat(OpKernelContext* context, const gtl::ArraySlice<Tensor> inputs,
              Tensor* output) {
  const int input_dims = inputs[0].dims();
  const TensorShape& input_shape = inputs[0].shape();

  // Note that we reduce the concat of k-dimensional tensors into a two
  // dimensional concat. Assuming the dimensions of any input tensor are
  // {y0, y1,...,ym-1}, we flatten it to {1, y}, where y = Prod_i(yi).
  std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>> inputs_flat;
  inputs_flat.reserve(inputs.size());
  int64_t output_dim0 = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Tensor& input = inputs[i];
    if (input.dims() != input_dims) {
      return errors::InvalidArgument(
          "Ranks of all input tensors should match: shape[0] = ",
          input_shape.DebugString(), " vs. shape[", i,
          "] = ", input.shape().DebugString());
    }
    for (int j = 1; j < input_dims; ++j) {
      if (input.dim_size(j) != input_shape.dim_size(j)) {
        return errors::InvalidArgument(
            "Dimensions of inputs should match: shape[0] = ",
            input_shape.DebugString(), " vs. shape[", i,
            "] = ", input.shape().DebugString());
      }
    }
    if (input.NumElements() > 0) {
      inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
          input.shaped<T, 2>({1, input.NumElements()})));
    }
    output_dim0 += input.dim_size(0);
  }

  TensorShape output_shape(input_shape);
  output_shape.set_dim(0, output_dim0);
  AllocatorAttributes attr;
  attr.set_on_host(true);
  TF_RETURN_IF_ERROR(context->allocate_temp(DataTypeToEnum<T>::value,
                                            output_shape, output, attr));
  if (output->NumElements() > 0) {
    auto output_flat = output->shaped<T, 2>({1, output->NumElements()});
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
    if (std::is_same<Device, GPUDevice>::value) {
      ConcatGPU<T>(context, inputs_flat, output, &output_flat);
      return OkStatus();
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    ConcatCPU<T>(context->device(), inputs_flat, &output_flat);
  }

  return OkStatus();
}

// Same as 'Concat' above, but handles Tensor dtype deduction automatically.
inline Status Concat(OpKernelContext* context,
                     const gtl::ArraySlice<Tensor> inputs, Tensor* output) {
  const DataType type = inputs[0].dtype();
  Status concat_status;
  switch (type) {
#define CASE(type)                                         \
  case DataTypeToEnum<type>::value:                        \
    concat_status = Concat<type>(context, inputs, output); \
    break;
    TF_CALL_ALL_TYPES(CASE);
#undef CASE
    default:
      concat_status = errors::InvalidArgument("Unsupported data type: ", type);
      break;
  }
  return concat_status;
}

// The Split*() functions split 'input' with element type T into 'sizes.size()'
// tensors along the zeroth dimension, with the ith split having zeroth-
// dimension size 'sizes[i]'. They allocate the output tensors using 'context',
// for proper device placement.

// Handles special cases that are cheap. Sets 'done==true' iff it found an
// applicable special case and wrote to the outputs. Otherwise acts as a no-op.
template <typename T>
Status SplitEasyCases(OpKernelContext* context, const Tensor& input,
                      const gtl::ArraySlice<int64_t> sizes,
                      std::vector<Tensor>* outputs, bool* done) {
  *done = false;

  int64_t total_size = 0;
  for (const int64_t size : sizes) {
    total_size += size;
  }
  if (total_size > input.shape().dim_size(0)) {
    return errors::InvalidArgument(
        "Sum of split sizes must not exceed dim0-size of input tensor");
  }

  // Special case 0: trivial 1-way split.
  if (sizes.size() == 1 && sizes.at(0) == input.shape().dim_size(0)) {
    outputs->push_back(input);
    *done = true;
    return OkStatus();
  }

  // Special case 1: input is aligned.
  if (IsInnerDimsSizeAligned<T>(input.shape())) {
    int64_t position = 0;
    for (const int64_t size : sizes) {
      outputs->emplace_back(input.Slice(position, position + size));
      position += size;
    }
    *done = true;
    return OkStatus();
  }

  return OkStatus();
}

// Handles the general case, on CPU.
template <typename T>
Status SplitCPU(OpKernelContext* context, const Tensor& input,
                const gtl::ArraySlice<int64_t> sizes,
                std::vector<Tensor>* outputs) {
  int64_t suffix_dim_size = 1;
  for (int i = 1; i < input.shape().dims(); ++i) {
    suffix_dim_size *= input.shape().dim_size(i);
  }
  auto input_reshaped =
      input.shaped<T, 2>({input.shape().dim_size(0), suffix_dim_size});

  int64_t position = 0;
  for (const int64_t size : sizes) {
    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, size);
    Tensor output;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    TF_RETURN_IF_ERROR(
        context->allocate_temp(input.dtype(), output_shape, &output, attr));
    auto output_shaped = output.shaped<T, 2>({size, suffix_dim_size});

    Eigen::DSizes<Eigen::DenseIndex, 2> slice_indices{
        static_cast<Eigen::DenseIndex>(position), 0};
    Eigen::DSizes<Eigen::DenseIndex, 2> slice_sizes{
        static_cast<Eigen::DenseIndex>(size),
        static_cast<Eigen::DenseIndex>(suffix_dim_size)};
    functor::Split<CPUDevice, T, 2>()(context->eigen_device<CPUDevice>(),
                                      output_shaped, input_reshaped,
                                      slice_indices, slice_sizes);

    outputs->emplace_back(output);

    position += size;
  }

  return OkStatus();
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

// Handles the general case, on GPU.
template <typename T>
Status SplitGPU(OpKernelContext* context, const Tensor& input,
                const gtl::ArraySlice<int64_t>& sizes,
                std::vector<Tensor>* outputs) {
  // TODO(olston, apassos): Implement this.
  LOG(FATAL) << "Not yet implemented";  // Crash ok
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// The outer function that dispatches to the various Split*() functions above.
template <typename T>
Status Split(OpKernelContext* context, const Tensor& input,
             const gtl::ArraySlice<int64_t> sizes,
             std::vector<Tensor>* outputs) {
  bool easy_cases_done;
  TF_RETURN_IF_ERROR(
      SplitEasyCases<T>(context, input, sizes, outputs, &easy_cases_done));
  if (easy_cases_done) {
    return OkStatus();
  }

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
// TODO(olston, apassos): Handle non-CPU cases.
// return SplitGPU<T>(context, input, sizes, outputs);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return SplitCPU<T>(context, input, sizes, outputs);
}

// Same as 'Split' above, but handles Tensor dtype automatically.
inline Status Split(OpKernelContext* context, const Tensor& input,
                    const gtl::ArraySlice<int64_t> sizes,
                    std::vector<Tensor>* outputs) {
  const DataType type = input.dtype();
  Status split_status;
  switch (type) {
#define CASE(type)                                              \
  case DataTypeToEnum<type>::value:                             \
    split_status = Split<type>(context, input, sizes, outputs); \
    break;
    TF_CALL_ALL_TYPES(CASE);
#undef CASE
    default:
      split_status = errors::InvalidArgument("Unsupported data type: ", type);
      break;
  }
  return split_status;
}

}  // namespace concat_split_util
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_CONCAT_SPLIT_UTIL_H_
