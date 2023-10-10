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

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_CONDITIONAL_ACCUMULATOR_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_CONDITIONAL_ACCUMULATOR_H_

#include "tensorflow/core/kernels/typed_conditional_accumulator_base.h"

namespace tensorflow {

/**
 * An aggregation object for adding sparse gradients, represented as a tuple of
 * indices, values, and a (possibly empty) shape.
 *
 * The two main methods of this class are TryApplyGrad and TryTakeGrad.
 *
 * TryApplyGrad tries add a gradient to the accumulator. The attempt is
 * successful if local_step >= global_step, i.e., if the gradient is not stale,
 * having been computed using up-to-date information. Otherwise, the gradient is
 * silently dropped.
 *
 * TryTakeGrad logs an attempt to read the average gradient. The attempt is
 * blocked until the number of gradients accumulated (via TryApplyGrad) is equal
 * or exceeds the number requested by TryTakeGrad.
 * Once this condition is satisfied, the following actions are taken:
 * (1) the value of the average gradient is returned
 * (2) the count of accumulated gradients is reset to 0
 * (3) the internal global_step value (current_global_step_) is incremented by 1
 *
 * SparseConditionalAccumulator is the datatype-dependent templated sub-class of
 * ConditionalAccumulatorBase. It implements the virtual arithmetic methods that
 * are used by for aggregating, averaging, allocating, returning indexed slices.
 */
template <typename Device, typename T>
class SparseConditionalAccumulator
    : public TypedConditionalAccumulatorBase<
          std::tuple<const Tensor*, const Tensor*, const Tensor*>> {
 public:
  SparseConditionalAccumulator(const DataType& dtype,
                               const PartialTensorShape& shape,
                               const string& name, const string& reduction_type)
      : TypedConditionalAccumulatorBase<
            std::tuple<const Tensor*, const Tensor*, const Tensor*>>(
            dtype, shape, name, reduction_type),
        accum_val_(std::make_unique<Tensor>()) {}

 protected:
  std::unique_ptr<std::vector<int64_t>> accum_idx_vec_;
  std::unique_ptr<std::vector<int>> count_element_;

  std::unique_ptr<Tensor> accum_val_;

  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>,
                           Eigen::Unaligned>
      SliceT;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,
                           Eigen::Unaligned>
      SliceConstT;

  Status ValidateShape(
      std::tuple<const Tensor*, const Tensor*, const Tensor*>* tensor,
      bool has_known_shape) TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    const Tensor* tensor_idx = std::get<0>(*tensor);
    const Tensor* tensor_val = std::get<1>(*tensor);
    const Tensor* tensor_shape = std::get<2>(*tensor);
    int64_t grad_val_dims = tensor_val->dims();
    int64_t grad_dims = grad_val_dims;

    // Compare with provided shape
    if (has_known_shape) {
      if (shape_.dims() > tensor_shape->NumElements()) {
        return errors::InvalidArgument(
            "Shape mismatch: expected shape rank at least ", shape_.dims(),
            ", got ", tensor_shape->NumElements());
      }
      const auto tensor_shape_flat = tensor_shape->flat<int64_t>();
      for (int64_t i = 0; i < shape_.dims(); i++) {
        if (shape_.dim_size(i) != -1 &&
            shape_.dim_size(i) != tensor_shape_flat(i)) {
          return errors::InvalidArgument("Shape mismatch: expected shape dim ",
                                         i, " to be ", shape_.dim_size(i),
                                         ", got ", tensor_shape_flat(i));
        }
      }
    }
    // Check that indices are within limits
    if (shape_.dims() > 0 && shape_.dim_size(0) != -1 &&
        tensor_idx->dims() > 0) {
      for (int64_t i = 0; i < tensor_idx->dim_size(0); i++) {
        if (tensor_idx->vec<int64_t>()(i) >= shape_.dim_size(0)) {
          return errors::InvalidArgument(
              "Shape mismatch: index of slice ", i, " exceeded limits of shape",
              "; index is ", tensor_idx->vec<int64_t>()(i), " exceeded ",
              shape_.dim_size(0));
        }
      }
    }

    // Check values compatibility with accumulated gradient if available
    if (counter_ > 0) {
      int64_t accum_val_dims = accum_val_->dims();
      if (accum_val_dims != grad_val_dims) {
        return errors::InvalidArgument("Shape mismatch: expected values rank ",
                                       accum_val_dims, ", got ", grad_val_dims);
      }
      for (int64_t i = 1; i < accum_val_dims; i++) {
        if (accum_val_->dim_size(i) != tensor_val->dim_size(i)) {
          return errors::InvalidArgument("Shape mismatch: expected values dim ",
                                         i, " to be ", accum_val_->dim_size(i),
                                         ", got ", tensor_val->dim_size(i));
        }
      }
    } else {
      // If there are no accumulated gradients, check against shape_
      if (shape_.dims() > grad_dims) {
        return errors::InvalidArgument(
            "Shape mismatch: expected values rank at least ", shape_.dims(),
            ", got ", grad_dims);
      }
      // Check that values have correct dimensions
      for (int64_t i = 1; i < shape_.dims(); i++) {
        if (shape_.dim_size(i) != -1 &&
            shape_.dim_size(i) != tensor_val->dim_size(i)) {
          return errors::InvalidArgument("Shape mismatch: expected values dim ",
                                         i, " to be ", shape_.dim_size(i),
                                         ", got ", tensor_val->dim_size(i));
        }
      }
    }

    return OkStatus();
  }

  void AllocateAndAssignToAccumGradFunction(
      OpKernelContext* ctx,
      std::tuple<const Tensor*, const Tensor*, const Tensor*>* grad) override {
    const Tensor* grad_idx = std::get<0>(*grad);
    const Tensor* grad_val = std::get<1>(*grad);

    const int64_t nnz = grad_idx->dim_size(0);

    // Assign indices
    accum_idx_vec_ = std::make_unique<std::vector<int64_t>>();
    accum_idx_vec_->reserve(nnz);
    for (int i = 0; i < nnz; i++) {
      accum_idx_vec_->push_back(grad_idx->vec<int64_t>()(i));
    }

    // Assign values to accum_val_tensor
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(dtype_, grad_val->shape(), accum_val_.get()));
    accum_val_->flat<T>().device(ctx->template eigen_device<Device>()) =
        grad_val->flat<T>();

    // Assign count_element_
    count_element_ = std::make_unique<std::vector<int>>(nnz, 1);

    // Do not need shape; Assume that the op has checked that the shapes match,
    // so grad's shape == shape_
  }

  void AddToAccumGradFunction(
      OpKernelContext* ctx,
      std::tuple<const Tensor*, const Tensor*, const Tensor*>* grad) override {
    // Modeled after third_party/tensorflow/core/kernels/sparse_add_op

    const Tensor* grad_idx = std::get<0>(*grad);
    const Tensor* grad_val = std::get<1>(*grad);

    const int64_t accum_nnz = accum_idx_vec_->size();
    const int64_t grad_nnz = grad_idx->dim_size(0);

    // Source enumerates the origin of a non-zero element: whether it is from
    // the new gradient, the accumulated gradient, or the sum of both.
    enum Source { from_accum, from_grad, from_accum_and_grad };

    // (1) do a pass over inputs, and append values and indices to vectors
    std::vector<std::tuple<Source, int64, int64>> entries_to_copy;
    entries_to_copy.reserve(accum_nnz + grad_nnz);

    // Pass over all non-zero elements of both the gradient and the accumulated
    // value, to identify where each non-zero element of the sum comes from.
    // The input and output indexed slices are assumed to be ordered along
    // increasing dimension number.
    int64_t i = 0, j = 0;
    int64_t sum_nnz = 0;
    while (i < accum_nnz && j < grad_nnz) {
      sum_nnz++;
      switch (cmp(accum_idx_vec_.get(), grad_idx, i, j)) {
        case -1:
          entries_to_copy.emplace_back(from_accum, i, -1);
          ++i;
          break;
        case 0:
          entries_to_copy.emplace_back(from_accum_and_grad, i, j);
          ++i;
          ++j;
          break;
        case 1:
          entries_to_copy.emplace_back(from_grad, -1, j);
          ++j;
          break;
      }
    }

    // Handle leftovers
    while (i < accum_nnz) {
      sum_nnz++;
      entries_to_copy.emplace_back(from_accum, i, -1);
      ++i;
    }
    while (j < grad_nnz) {
      sum_nnz++;
      entries_to_copy.emplace_back(from_grad, -1, j);
      ++j;
    }

    // (2) Copy or sum the non-zero elements into sum_indices and sum_tensor
    std::vector<int64_t>* sum_indices_vec = new std::vector<int64_t>();
    sum_indices_vec->reserve(sum_nnz);

    std::vector<int>* sum_counts = new std::vector<int>();
    sum_counts->reserve(sum_nnz);

    Tensor* sum_tensor = new Tensor();

    TensorShape sum_shape = grad_val->shape();
    sum_shape.set_dim(0, sum_nnz);

    OP_REQUIRES_OK(ctx, ctx->allocate_temp(dtype_, sum_shape, sum_tensor));
    auto sum_flat = sum_tensor->flat_outer_dims<T>();
    auto accum_flat = accum_val_->flat_outer_dims<T>();
    auto grad_flat = grad_val->flat_outer_dims<T>();

    const int64_t num_col = grad_flat.dimension(1);

    Eigen::DSizes<Eigen::DenseIndex, 1> slice_shape(num_col);

    for (i = 0; i < sum_nnz; ++i) {
      const Source src = std::get<0>(entries_to_copy[i]);
      const int64_t idx_a = std::get<1>(entries_to_copy[i]);
      const int64_t idx_b = std::get<2>(entries_to_copy[i]);
      T* sum_slice_ptr = &sum_flat(i, 0);
      SliceT sum_slice(sum_slice_ptr, slice_shape);
      if (src == from_accum) {
        // Element comes from accumulator; directly copy data structures over
        sum_indices_vec->push_back(accum_idx_vec_->at(idx_a));
        T* accum_slice_ptr = &accum_flat(idx_a, 0);
        SliceT accum_slice(accum_slice_ptr, slice_shape);
        sum_slice = accum_slice;
        sum_counts->push_back(count_element_->at(idx_a));
      } else if (src == from_accum_and_grad) {
        // Element is a sum of accumulated value and new gradient;
        // compute sum here
        sum_indices_vec->push_back(accum_idx_vec_->at(idx_a));
        const T* grad_slice_ptr = &grad_flat(idx_b, 0);
        SliceConstT grad_slice(grad_slice_ptr, slice_shape);
        T* accum_slice_ptr = &accum_flat(idx_a, 0);
        SliceT accum_slice(accum_slice_ptr, slice_shape);
        sum_slice = grad_slice + accum_slice;
        sum_counts->push_back(count_element_->at(idx_a) + 1);
      } else if (src == from_grad) {
        // Element comes from new gradient; make a copy of indices and values
        sum_indices_vec->push_back(grad_idx->vec<int64_t>()(idx_b));
        const T* grad_slice_ptr = &grad_flat(idx_b, 0);
        SliceConstT grad_slice(grad_slice_ptr, slice_shape);
        sum_slice = grad_slice;
        sum_counts->push_back(1);
      }
    }

    // (3) Keep output, i.e., switch pointers to point to new data structures
    // representing the sum
    // Indices
    accum_idx_vec_.reset(sum_indices_vec);
    // Values
    accum_val_.reset(sum_tensor);
    // Counts
    count_element_.reset(sum_counts);

    // No need to copy shape, since shape remains the same after sum.
  }

  void DivideAccumGradByCounter(OpKernelContext* ctx) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    const int64_t nnz = count_element_->size();
    auto accum_flat = accum_val_->flat_outer_dims<T>();
    std::vector<T> count_typet;
    std::transform(count_element_->begin(), count_element_->end(),
                   std::back_inserter(count_typet),
                   TypeConverter<T, int>::ConvertUToT);

    // Option 1: divide all by counter
    /*
    std::transform(
        &accum_flat(0,0), &accum_flat(nnz,0), &accum_flat(0,0),
        std::bind2nd(std::divides<T>(),
                     TypeConverter<T, int>::ConvertUToT(this->counter_)));
    */

    // Option 2: average element-wise
    Eigen::DSizes<Eigen::DenseIndex, 1> slice_shape(accum_flat.dimension(1));
    for (int64_t i = 0; i < nnz; i++) {
      T* accum_slice_ptr = &accum_flat(i, 0);
      SliceT accum_slice(accum_slice_ptr, slice_shape);
      accum_slice.device(ctx->template eigen_device<Device>()) =
          accum_slice / count_typet[i];
    }
  }

  bool SetOutput(OpKernelContext* ctx) override {
    bool is_successful = true;
    if (is_successful) is_successful = ReturnIdxTensor(ctx);
    if (is_successful) is_successful = ReturnValTensor(ctx);
    if (is_successful) is_successful = ReturnShapeTensor(ctx);
    return is_successful;
  }

  bool GetAndValidateTensorInputForApplyGrad(
      OpKernelContext* ctx,
      std::tuple<const Tensor*, const Tensor*, const Tensor*>** tensor) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    // TODO(xinghao, jmchen): The roundabout way of getting attr from
    // OpKernelContext (instead of OpKernelConstruction) is a hack, and should
    // be fixed if it affects efficiency.
    bool has_known_shape = false;
    OP_REQUIRES_OK_BOOLEAN(
        ctx, GetNodeAttr(ctx->op_kernel().def(), "has_known_shape",
                         &has_known_shape));

    // Get input gradient tensors
    const Tensor* grad_idx_tensor;
    OP_REQUIRES_OK_BOOLEAN(ctx,
                           ctx->input("gradient_indices", &grad_idx_tensor));
    const Tensor* grad_val_tensor;
    OP_REQUIRES_OK_BOOLEAN(ctx,
                           ctx->input("gradient_values", &grad_val_tensor));
    const Tensor* grad_shape_tensor = nullptr;
    if (has_known_shape) {
      OP_REQUIRES_OK_BOOLEAN(ctx,
                             ctx->input("gradient_shape", &grad_shape_tensor));
    }

    // Checks
    OP_REQUIRES_BOOLEAN(
        ctx, TensorShapeUtils::IsVector(grad_idx_tensor->shape()),
        errors::InvalidArgument(
            "Input indices should be vector but received shape: ",
            grad_idx_tensor->shape().DebugString()));
    const int64_t nnz = grad_idx_tensor->dim_size(0);
    OP_REQUIRES_BOOLEAN(
        ctx, grad_val_tensor->dims() > 0,
        errors::InvalidArgument("Values cannot be 0-dimensional."));
    OP_REQUIRES_BOOLEAN(ctx, grad_val_tensor->dim_size(0) == nnz,
                        errors::InvalidArgument("Expected ", nnz,
                                                " non-empty input values, got ",
                                                grad_val_tensor->dim_size(0)));

    *tensor = new std::tuple<const Tensor*, const Tensor*, const Tensor*>(
        grad_idx_tensor, grad_val_tensor, grad_shape_tensor);

    OP_REQUIRES_OK_BOOLEAN(ctx, this->ValidateShape(*tensor, has_known_shape));

    return true;
  }

  void CleanUpGradTensor(std::tuple<const Tensor*, const Tensor*,
                                    const Tensor*>* tensor) override {
    if (tensor != nullptr) delete tensor;
  }

 private:
  inline int cmp(std::vector<int64_t>* a_idx, const Tensor* b_idx,
                 const int64_t a_row, const int64_t b_row) {
    const int64_t a = a_idx->at(a_row);
    const int64_t b = b_idx->vec<int64_t>()(b_row);
    if (a < b) {
      return -1;
    } else if (a > b) {
      return 1;
    }
    return 0;
  }

  inline bool ReturnIdxTensor(OpKernelContext* ctx) {
    Tensor* idx_tensor;
    const int64_t nnz = accum_idx_vec_->size();
    OP_REQUIRES_OK_BOOLEAN(ctx, ctx->allocate_output(0, {nnz}, &idx_tensor));
    // If allocate_output fails, OP_REQUIRES_OK_BOOLEAN will short-circuit
    // the remaining code and just return false
    auto idx_tensor_vec = idx_tensor->vec<int64_t>();
    for (int i = 0; i < nnz; ++i) {
      idx_tensor_vec(i) = accum_idx_vec_->at(i);
    }
    return true;
  }

  inline bool ReturnValTensor(OpKernelContext* ctx) {
    ctx->set_output(1, *accum_val_);
    return true;
  }

  inline bool ReturnShapeTensor(OpKernelContext* ctx) {
    int64_t accum_val_dims = accum_val_->dims();
    Tensor* shape_tensor;
    OP_REQUIRES_OK_BOOLEAN(
        ctx, ctx->allocate_output(2, {accum_val_dims}, &shape_tensor));
    // If allocate_output fails, OP_REQUIRES_OK_BOOLEAN will short-circuit
    // the remaining code and just return false

    // First dim of shape is defined by shape_, others by accum_val_->shape
    shape_tensor->flat<int64_t>()(0) =
        (shape_.dims() > 0) ? shape_.dim_size(0) : -1;
    for (int64_t i = 1; i < accum_val_dims; i++) {
      shape_tensor->flat<int64_t>()(i) = accum_val_->dim_size(i);
    }
    return true;
  }

  SparseConditionalAccumulator(const SparseConditionalAccumulator&) = delete;
  void operator=(const SparseConditionalAccumulator&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_CONDITIONAL_ACCUMULATOR_H_
