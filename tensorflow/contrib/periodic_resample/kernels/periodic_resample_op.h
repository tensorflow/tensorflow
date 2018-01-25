// =============================================================================
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef TENSORFLOW_KERNELS_PERIODICRESAMPLE_OP_H_
#define TENSORFLOW_KERNELS_PERIODICRESAMPLE_OP_H_

#include <cmath>
#include <type_traits>
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"

namespace {

template <class IndexVecT, class IndexT>
IndexT compute_input_index(
    IndexVecT* target_dimensions, const IndexT& output_index,
    const IndexVecT& original_dimensions, const int& adjustable_dimension,
    const std::vector<tensorflow::int64>& dimension_ceiling,
    const std::vector<tensorflow::int64>& cumulative_dimensions, IndexT* result,
    std::vector<IndexT>* output_indices, const int& rank) {
  *result = 0;
  output_indices->clear();

  // un-rasterize the output index
  auto last_reduced_i = output_index;
  for (auto r = rank - 1; r >= 0; --r) {
    (*output_indices)[r] = last_reduced_i % (*target_dimensions)[r];
    last_reduced_i =
        (last_reduced_i - (*output_indices)[r]) / (*target_dimensions)[r];
  }

  // rasterize the input index
  IndexT last_index_factor = 1;
  for (auto r = rank - 1; r >= 0; --r) {
    IndexT index = 0;
    if (r != adjustable_dimension)
      index = (*output_indices)[r] / dimension_ceiling[r];
    else {
      for (int qi = 0; qi < rank; ++qi) {
        if (qi == adjustable_dimension) continue;
        index += cumulative_dimensions[qi] *
                 ((*output_indices)[qi] % dimension_ceiling[qi]);
      }
      index *= (*target_dimensions)[adjustable_dimension];
      index += (*output_indices)[r];
    }
    *result += last_index_factor * index;
    last_index_factor *= original_dimensions[r];
  }

  return *result;
}

template <class InputDataT,
          class IndexVecT>  // both types are needed here b/c IndexVecT and
                            // InputDataT are not related
                            void
                            fill_periodic_tensor(
                                tensorflow::OpKernelContext* context,
                                const IndexVecT& desired_shape,
                                const tensorflow::Tensor& input_tensor) {
  // input is a strided array (last index is fastest, C-ordered)
  auto input = input_tensor.flat<InputDataT>();
  const int rank = input_tensor.dims();
  // original and target dimensions
  std::vector<tensorflow::int64> original_dimensions(rank),
      target_dimensions(rank);
  tensorflow::int64 total_size(input_tensor.NumElements()), new_sliced_size(1);
  // factors by which original_dimensions increases/decreases w.r.t.
  // target_dimensions
  std::vector<tensorflow::int64> dimension_ceiling(rank),
      cumulative_dimensions(rank);
  // index of adjustable dimension
  int adjustable_dimension;
  tensorflow::TensorShape output_shape;

  // requires that the rank of the input tensor and length of the desired shape
  // are equal
  OP_REQUIRES(context, rank == desired_shape.size(),
              tensorflow::errors::InvalidArgument(
                  "periodic_resample expects the rank of the input tensor, ",
                  rank, ", to be the same as the length of the desired shape, ",
                  desired_shape.size(), "."));

  bool found = false;
  const auto& input_tensor_shape = input_tensor.shape();

  for (int i = 0; i < rank; ++i) {
    // if (desired_shape(i) < 1) {
    if (desired_shape[i] < 1) {
      // only one index can be adjustable
      OP_REQUIRES(context, !found,
                  tensorflow::errors::InvalidArgument(
                      "periodic_resample expects only "
                      "one index to be marked as adjustable."));
      adjustable_dimension = i;
      found = true;
    } else {
      OP_REQUIRES(
          context, desired_shape[i] >= input_tensor_shape.dim_size(i),
          tensorflow::errors::InvalidArgument(
              "periodic_resample expects the size of non-adjustable "
              "dimensions be at least as large as size of input tensor."
              " Dimension ", i, " input tensor has size ",
              input_tensor_shape.dim_size(i), ", desired shape has size ",
              desired_shape[i], "."));

      // target_dimensions[i] = desired_shape(i);
      target_dimensions[i] = desired_shape[i];
      new_sliced_size *= target_dimensions[i];
    }
  }
  // at least one index needs to be adjustable
  OP_REQUIRES(context, found,
              tensorflow::errors::InvalidArgument(
                  "periodic_resample expects at least "
                  "one index to be marked as adjustable."));

  int count = 0;
  for (const auto dim_info : input_tensor.shape()) {
    original_dimensions[count] = dim_info.size;
    ++count;
  }

  target_dimensions[adjustable_dimension] = total_size / new_sliced_size;

  count = 0;
  for (int i = 0; i < input_tensor.shape().dims(); ++i) {
    dimension_ceiling[count] = tensorflow::int64(std::ceil(
        float(target_dimensions[count]) / float(original_dimensions[count])));
    if (count == 0)
      cumulative_dimensions[count] = 1;
    else
      cumulative_dimensions[count] =
          cumulative_dimensions[count - 1] * dimension_ceiling[count - 1];
    ++count;
  }

  // ensure that the new dimension is greater than zero
  OP_REQUIRES(context, target_dimensions[adjustable_dimension] > 0,
              tensorflow::errors::InvalidArgument(
                  "periodic_resample found that the "
                  "adjustable dimension, ",
                  adjustable_dimension, ", isn't greater than zero, ",
                  target_dimensions[adjustable_dimension], "."));
  for (int i = 0; i < rank; ++i) {
    output_shape.AddDim(target_dimensions[i]);
  }
  const auto new_size =
      new_sliced_size * target_dimensions[adjustable_dimension];

  // Create an output tensor and attach it to the current context
  tensorflow::Tensor* output_tensor = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, output_shape, &output_tensor));
  auto output = output_tensor->flat<InputDataT>();

  // memory is allocated for these variables outside the inner loop for
  // efficiency (although, I could create a separate class scope for
  // this purpose instead)
  tensorflow::int64 result = 0;
  std::vector<tensorflow::int64> output_indices(target_dimensions.size());

  // Fill output tensor with periodically resampled input tensor values
  for (tensorflow::int64 output_index = 0; output_index < new_size;
       ++output_index) {
    output(output_index) = input(compute_input_index(
        &target_dimensions, output_index, original_dimensions,
        adjustable_dimension, dimension_ceiling, cumulative_dimensions, &result,
        &output_indices, rank));
  }
}

void create_output_tensor(
    tensorflow::OpKernelContext* context,
    const tensorflow::Tensor& input_tensor,
    const tensorflow::DataType& input_tensor_type,
    const tensorflow::PartialTensorShape& desired_shape_tensor) {
  auto desired_shape = desired_shape_tensor.dim_sizes();

  // obligatory type switch
  switch (input_tensor_type) {
    case tensorflow::DataTypeToEnum<float>::value:
      fill_periodic_tensor<float>(context, desired_shape, input_tensor);
      break;
    case tensorflow::DataTypeToEnum<double>::value:
      fill_periodic_tensor<double>(context, desired_shape, input_tensor);
      break;
    case tensorflow::DataTypeToEnum<tensorflow::int32>::value:
      fill_periodic_tensor<tensorflow::int32>(context, desired_shape,
                                              input_tensor);
      break;
    case tensorflow::DataTypeToEnum<tensorflow::int64>::value:
      fill_periodic_tensor<tensorflow::int64>(context, desired_shape,
                                              input_tensor);
      break;
    default:;
  }
}

}  // namespace

class PeriodicResampleOp : public tensorflow::OpKernel {
 public:
  explicit PeriodicResampleOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    // Get the desired shape
    OP_REQUIRES_OK(context, context->GetAttr("shape", &desired_shape));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    // Grab the input tensor
    const tensorflow::Tensor& input_tensor = context->input(0);
    const tensorflow::DataType input_tensor_type = context->input_dtype(0);

    create_output_tensor(context, input_tensor, input_tensor_type,
                         desired_shape);
  }

 private:
  tensorflow::PartialTensorShape desired_shape;
};

#endif  // TENSORFLOW_KERNELS_PERIODICRESAMPLE_OP_H_
