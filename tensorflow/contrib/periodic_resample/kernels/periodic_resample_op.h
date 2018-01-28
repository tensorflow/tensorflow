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
#include "tensorflow/core/util/work_sharder.h"

namespace {

// Computes input tensor index for given output index during forward
// propagation through periodic_resample operation.
class InputIndexer {
 public:
  InputIndexer(const std::vector<tensorflow::int64>& output_dimensions,
               const tensorflow::TensorShape& input_shape,
               int adjustable_dimension)
      : output_dimensions_(output_dimensions),
        adjustable_dimension_(adjustable_dimension),
        rank_(input_shape.dims()),
        linear_output_index_(0),
        linear_input_index_(0),
        adjustable_dimension_carriage_sum_(0) {
    auto input_dimensions = TensorShapeToVector(input_shape);
    // factors by which input_dimensions increases/decreases w.r.t.
    // output_dimensions
    dimension_ceiling_ =
        ComputeDimensionCeiling(output_dimensions, input_dimensions);
    cumulative_dimensions_ = ComputeCumulativeDimensions();

    output_indices_.resize(output_dimensions_.size());
    input_indices_.resize(output_dimensions_.size());

    // Compute index_factors
    index_factors_.resize(rank_);
    tensorflow::int64 last_index_factor = 1;
    for (auto r = rank_ - 1; r >= 0; --r) {
      index_factors_[r] = last_index_factor;
      last_index_factor *= input_dimensions[r];
    }
  }

  tensorflow::int64 linear_input_index() const { return linear_input_index_; }

  void MoveToOutputIndex(tensorflow::int64 output_index);
  void IncrementOutputIndex();

 private:
  void RecomputeInputAdjustableDimensionIndex() {
    tensorflow::int64 index = adjustable_dimension_carriage_sum_;
    index *= output_dimensions_[adjustable_dimension_];
    index += output_indices_[adjustable_dimension_];
    input_indices_[adjustable_dimension_] = index;
  }

  std::vector<tensorflow::int64> TensorShapeToVector(
      const tensorflow::TensorShape& tensor_shape);

  std::vector<tensorflow::int64> ComputeDimensionCeiling(
      const std::vector<tensorflow::int64>& output_dimensions,
      const std::vector<tensorflow::int64>& input_dimensions);

  std::vector<tensorflow::int64> ComputeCumulativeDimensions();

  const std::vector<tensorflow::int64> output_dimensions_;
  std::vector<tensorflow::int64> dimension_ceiling_;
  std::vector<tensorflow::int64> index_factors_;
  std::vector<tensorflow::int64> cumulative_dimensions_;
  std::vector<tensorflow::int64> output_indices_;
  std::vector<tensorflow::int64> input_indices_;

  const int adjustable_dimension_;
  const int rank_;
  tensorflow::int64 linear_output_index_;
  tensorflow::int64 linear_input_index_;
  tensorflow::int64 adjustable_dimension_carriage_sum_;
};

void InputIndexer::MoveToOutputIndex(tensorflow::int64 output_index) {
  linear_output_index_ = output_index;
  linear_input_index_ = 0;

  // un-rasterize the output index
  auto last_reduced_i = output_index;
  for (auto r = rank_ - 1; r >= 0; --r) {
    output_indices_[r] = last_reduced_i % output_dimensions_[r];
    last_reduced_i =
        (last_reduced_i - output_indices_[r]) / output_dimensions_[r];
  }

  tensorflow::int64 carriage_sum = 0;
  for (int qi = 0; qi < rank_; ++qi) {
    if (qi == adjustable_dimension_) continue;
    carriage_sum += cumulative_dimensions_[qi] *
                    (output_indices_[qi] % dimension_ceiling_[qi]);
  }
  adjustable_dimension_carriage_sum_ = carriage_sum;

  // rasterize the input index
  for (auto r = rank_ - 1; r >= 0; --r) {
    if (r != adjustable_dimension_)
      input_indices_[r] = output_indices_[r] / dimension_ceiling_[r];
    else
      RecomputeInputAdjustableDimensionIndex();
  }
  for (auto r = rank_ - 1; r >= 0; --r) {
    linear_input_index_ += index_factors_[r] * input_indices_[r];
  }
}

void InputIndexer::IncrementOutputIndex() {
  linear_output_index_++;
  for (auto r = rank_ - 1; r >= 0; --r) {
    auto old_carriage_sum_increment =
        cumulative_dimensions_[r] *
        (output_indices_[r] % dimension_ceiling_[r]);
    output_indices_[r] = (output_indices_[r] + 1) % output_dimensions_[r];
    if (r != adjustable_dimension_) {
      auto new_input_index = output_indices_[r] / dimension_ceiling_[r];
      linear_input_index_ +=
          (new_input_index - input_indices_[r]) * index_factors_[r];

      input_indices_[r] = new_input_index;

      auto new_carriage_sum_increment =
          cumulative_dimensions_[r] *
          (output_indices_[r] % dimension_ceiling_[r]);

      adjustable_dimension_carriage_sum_ = adjustable_dimension_carriage_sum_ -
                                           old_carriage_sum_increment +
                                           new_carriage_sum_increment;
    }

    if (output_indices_[r] != 0) {
      // No more carries to higher indices.
      break;
    }
  }
  auto old_adjustable_dimension_input_index =
      input_indices_[adjustable_dimension_];
  RecomputeInputAdjustableDimensionIndex();
  linear_input_index_ += (input_indices_[adjustable_dimension_] -
                           old_adjustable_dimension_input_index) *
                          index_factors_[adjustable_dimension_];
}

std::vector<tensorflow::int64> InputIndexer::TensorShapeToVector(
    const tensorflow::TensorShape& tensor_shape) {
  std::vector<tensorflow::int64> result(tensor_shape.dims());
  int count = 0;
  for (const auto dim_info : tensor_shape) {
    result[count] = dim_info.size;
    ++count;
  }
  return result;
}

std::vector<tensorflow::int64> InputIndexer::ComputeDimensionCeiling(
    const std::vector<tensorflow::int64>& output_dimensions,
    const std::vector<tensorflow::int64>& input_dimensions) {
  std::vector<tensorflow::int64> dimension_ceiling(input_dimensions.size());
  for (size_t i = 0; i < input_dimensions.size(); ++i) {
    dimension_ceiling[i] = tensorflow::int64(
        std::ceil(float(output_dimensions[i]) / float(input_dimensions[i])));
  }
  return dimension_ceiling;
}

std::vector<tensorflow::int64> InputIndexer::ComputeCumulativeDimensions() {
  std::vector<tensorflow::int64> cumulative_dimensions(rank_);
  int count = 0;
  for (int i = 0; i < rank_; ++i) {
    if (count == 0)
      cumulative_dimensions[count] = 1;
    else
      cumulative_dimensions[count] =
          cumulative_dimensions[count - 1] * dimension_ceiling_[count - 1];
    ++count;
  }
  return cumulative_dimensions;
}

template <typename IndexVecT>
void process_desired_shape(tensorflow::OpKernelContext* context,
                           const tensorflow::TensorShape& input_tensor_shape,
                           const IndexVecT& desired_shape,
                           int* adjustable_dimension,
                           std::vector<tensorflow::int64>* target_dimensions,
                           tensorflow::int64* output_size) {
  tensorflow::int64 new_sliced_size = 1;
  bool found = false;
  const int rank = input_tensor_shape.dims();
  for (int i = 0; i < rank; ++i) {
    if (desired_shape[i] < 1) {
      // only one index can be adjustable
      OP_REQUIRES(context, !found,
                  tensorflow::errors::InvalidArgument(
                      "periodic_resample expects only "
                      "one index to be marked as adjustable."));
      *adjustable_dimension = i;
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

      (*target_dimensions)[i] = desired_shape[i];
      new_sliced_size *= (*target_dimensions)[i];
    }
  }
  // at least one index needs to be adjustable
  OP_REQUIRES(context, found,
              tensorflow::errors::InvalidArgument(
                  "periodic_resample expects at least "
                  "one index to be marked as adjustable."));
  (*target_dimensions)[*adjustable_dimension] =
      input_tensor_shape.num_elements() / new_sliced_size;

  *output_size = new_sliced_size * (*target_dimensions)[*adjustable_dimension];
}

// Heuristic number based on measurements on development machine.
const tensorflow::int64 costPerFillIndex = 75;

template <class InputDataT,
          class IndexVecT>  // both types are needed here b/c IndexVecT and
// InputDataT are not related
void
fill_periodic_tensor(tensorflow::OpKernelContext* context,
                     const IndexVecT& desired_shape,
                     const tensorflow::Tensor& input_tensor) {
  // input is a strided array (last index is fastest, C-ordered)
  auto input = input_tensor.flat<InputDataT>();
  const int rank = input_tensor.dims();
  std::vector<tensorflow::int64> target_dimensions(rank);
  // index of adjustable dimension
  int adjustable_dimension = 0;
  tensorflow::int64 new_size = 0;
  tensorflow::TensorShape output_shape;

  // requires that the rank of the input tensor and length of the desired shape
  // are equal
  OP_REQUIRES(context, rank == desired_shape.size(),
              tensorflow::errors::InvalidArgument(
                  "periodic_resample expects the rank of the input tensor, ",
                  rank, ", to be the same as the length of the desired shape, ",
                  desired_shape.size(), "."));

  process_desired_shape(context, input_tensor.shape(), desired_shape,
                        &adjustable_dimension, &target_dimensions, &new_size);

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

  // Create an output tensor and attach it to the current context
  tensorflow::Tensor* output_tensor = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, output_shape, &output_tensor));
  auto output = output_tensor->flat<InputDataT>();

  // Fill output tensor with periodically resampled input tensor values
  InputIndexer input_indexer(target_dimensions, input_tensor.shape(),
                             adjustable_dimension);

  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
  auto fill_output_tensor = [&input_indexer, &output, &input](
      tensorflow::int64 start, tensorflow::int64 limit) {
    InputIndexer local_indexer(input_indexer);
    local_indexer.MoveToOutputIndex(start);
    for (tensorflow::int64 output_index = start; output_index < limit;
         ++output_index) {
      output(output_index) = input(local_indexer.linear_input_index());
      local_indexer.IncrementOutputIndex();
    }
  };
  ::tensorflow::Shard(worker_threads.num_threads, worker_threads.workers,
                      new_size, costPerFillIndex, fill_output_tensor);
}

template <class InputDataT>
void fill_grad_tensor(tensorflow::OpKernelContext* context,
                      const tensorflow::TensorShape& original_shape,
                      const tensorflow::PartialTensorShape& desired_shape,
                      const tensorflow::Tensor& grad_tensor) {
  const int rank = grad_tensor.dims();
  // requires that the rank of the input tensor and length of the desired shape
  // are equal
  OP_REQUIRES(context, rank == desired_shape.dims(),
              tensorflow::errors::InvalidArgument(
                  "periodic_resample gradient expects the rank of the ",
                  "gradient tensor, ", rank, ", to be the same as the length",
                  " of the desired shape, ", desired_shape.dims(), "."));

  std::vector<tensorflow::int64> target_dimensions(rank);
  tensorflow::int64 new_size = 0;
  // index of adjustable dimension
  int adjustable_dimension = 0;
  process_desired_shape(context, original_shape, desired_shape.dim_sizes(),
                        &adjustable_dimension, &target_dimensions, &new_size);

  // ensure that the new dimension is greater than zero
  OP_REQUIRES(context, target_dimensions[adjustable_dimension] > 0,
              tensorflow::errors::InvalidArgument(
                  "periodic_resample gradient found that the "
                  "adjustable dimension, ",
                  adjustable_dimension, ", isn't greater than zero, ",
                  target_dimensions[adjustable_dimension], "."));

  // Create an output tensor and attach it to the current context
  tensorflow::Tensor* output_tensor = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, original_shape, &output_tensor));
  auto output = output_tensor->flat<InputDataT>();

  // Fill output tensor with periodically resampled input tensor values
  // input is a strided array (last index is fastest, C-ordered)
  auto input_grad_data = grad_tensor.flat<InputDataT>();
  InputIndexer input_indexer(target_dimensions, original_shape,
                             adjustable_dimension);

  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
  auto fill_output_tensor = [&input_indexer, &output, &input_grad_data](
      tensorflow::int64 start, tensorflow::int64 limit) {
    InputIndexer local_indexer(input_indexer);
    local_indexer.MoveToOutputIndex(start);
    for (tensorflow::int64 input_grad_index = start; input_grad_index < limit;
         ++input_grad_index) {
      output(local_indexer.linear_input_index()) =
          input_grad_data(input_grad_index);
      local_indexer.IncrementOutputIndex();
    }
  };
  ::tensorflow::Shard(worker_threads.num_threads, worker_threads.workers,
                      new_size, costPerFillIndex, fill_output_tensor);
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
    default:
      ;
  }
}

void create_grad_tensor(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& grad_tensor,
                        const tensorflow::DataType& grad_tensor_type,
                        const tensorflow::TensorShape& original_shape,
                        const tensorflow::PartialTensorShape& desired_shape) {
  // obligatory type switch
  switch (grad_tensor_type) {
    case tensorflow::DataTypeToEnum<float>::value:
      fill_grad_tensor<float>(context, original_shape, desired_shape,
                              grad_tensor);
      break;
    case tensorflow::DataTypeToEnum<double>::value:
      fill_grad_tensor<double>(context, original_shape, desired_shape,
                               grad_tensor);
      break;
    case tensorflow::DataTypeToEnum<tensorflow::int32>::value:
      fill_grad_tensor<tensorflow::int32>(context, original_shape,
                                          desired_shape, grad_tensor);
      break;
    case tensorflow::DataTypeToEnum<tensorflow::int64>::value:
      fill_grad_tensor<tensorflow::int64>(context, original_shape,
                                          desired_shape, grad_tensor);
      break;
    default:
      ;
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

class PeriodicResampleOpGrad : public tensorflow::OpKernel {
 public:
  explicit PeriodicResampleOpGrad(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("original_shape", &original_shape));
    OP_REQUIRES_OK(context, context->GetAttr("desired_shape", &desired_shape));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& grad_tensor = context->input(0);
    const tensorflow::DataType grad_tensor_type = context->input_dtype(0);
    create_grad_tensor(context, grad_tensor, grad_tensor_type, original_shape,
                       desired_shape);
  }

 private:
  tensorflow::TensorShape original_shape;
  tensorflow::PartialTensorShape desired_shape;
};

#endif  // TENSORFLOW_KERNELS_PERIODICRESAMPLE_OP_H_
