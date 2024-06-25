/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_KERNELS_SHARDING_UTILS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_SHARDING_UTILS_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "Eigen/Core"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace sharding_internal {
absl::Status ValidateShapesForSlice(absl::string_view input_name,
                                    const Tensor* input,
                                    const std::vector<int32_t>& num_splits,
                                    const std::vector<int32_t>& paddings);
template <int Rank>
Eigen::DSizes<Eigen::DenseIndex, Rank> TF_ATTRIBUTE_NOINLINE
ShapeAsEigenDSizes(const TensorShape& shape);
template <int Rank>
Eigen::DSizes<Eigen::DenseIndex, Rank> ShapeAsEigenDSizes(
    const TensorShape& shape) {
  return shape.AsEigenDSizes<Rank>();
}

}  // namespace sharding_internal

// Converts flatten index to start indices (subscript scaled with slice shape)
// for determining where to start a slice in the input tensor.
template <int Rank>
Eigen::DSizes<Eigen::DenseIndex, Rank> GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, Rank>& slice_shape, int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 1> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 1>& slice_shape, int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 2> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 2>& slice_shape, int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 3> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_shape, int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 4> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 4>& slice_shape, int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 5> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 5>& slice_shape, int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 6> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 6>& slice_shape, int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 7> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 7>& slice_shape, int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 8> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 8>& slice_shape, int index);

template <int Rank>
Eigen::DSizes<Eigen::DenseIndex, Rank> GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, Rank>& slice_shape,
    const int index) {
  return Eigen::DSizes<Eigen::DenseIndex, Rank>();
}

// Shared base class to save code space
template <typename Device, typename T>
class XlaNDSplitter {
 public:
  static absl::StatusOr<XlaNDSplitter<Device, T>> Create(
      const std::vector<int32_t>& num_splits, int num_slices,
      const std::vector<int32_t>& paddings, bool has_paddings) {
    if (num_splits.size() != paddings.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("num_splits size ", num_splits.size(),
                       " mismatch with paddings size ", paddings.size(), "."));
    }

    int splits_cnt = 1;
    for (auto split : num_splits) {
      splits_cnt *= split;
    }

    if (num_slices != splits_cnt) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expect num_slices ", splits_cnt, " but got ", num_slices));
    }

    return XlaNDSplitter<Device, T>(num_splits, num_slices, paddings,
                                    has_paddings);
  }

  // Split the given input.
  //
  // The splitted outputs are stored into tensors allocated by
  // `allocate_output_fn`. In the simple case of pass through (no split and no
  // padding), the output is stored through the fast path by
  // `assign_or_copy_value_fn`.
  absl::Status Split(
      const Tensor* input, absl::string_view input_name,
      const std::function<Status(const Tensor&)>& assign_or_copy_value_fn,
      const std::function<Status(int index, const TensorShape& shape,
                                 Tensor** tensor)>& allocate_output_fn,
      const Device& device) {
    if (num_splits_.size() != paddings_.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("num_splits size ", num_splits_.size(),
                       " mismatch with paddings size ", paddings_.size(), "."));
    }

    const int rank = input->shape().dims();
    const auto& input_shape = input->shape().dim_sizes();

    TF_RETURN_IF_ERROR(sharding_internal::ValidateShapesForSlice(
        input_name, input, num_splits_, paddings_));

    TensorShape output_slice_shape;
    for (int i = 0; i < rank; ++i) {
      output_slice_shape.AddDim((input_shape[i] + paddings_[i]) /
                                ((num_slices_ == 1) ? 1 : num_splits_[i]));
    }
    if (num_slices_ == 1 && !has_paddings_) {
      // Handle simple case first
      TF_RETURN_IF_ERROR(assign_or_copy_value_fn(*input));
    } else {
      std::vector<Tensor*> output_slices(num_slices_);
      for (int i = 0; i < num_slices_; i++) {
        TF_RETURN_IF_ERROR(allocate_output_fn(
            /*index=*/i, output_slice_shape, &output_slices[i]));
      }

      if (rank == 1) {
        SliceAndMaybePad<1>(device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 2) {
        SliceAndMaybePad<2>(device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 3) {
        SliceAndMaybePad<3>(device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 4) {
        SliceAndMaybePad<4>(device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 5) {
        SliceAndMaybePad<5>(device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 6) {
        SliceAndMaybePad<6>(device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 7) {
        SliceAndMaybePad<7>(device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 8) {
        SliceAndMaybePad<8>(device, input, input_shape, output_slice_shape,
                            output_slices);
      }
    }
    return absl::OkStatus();
  }

 private:
  template <int Rank>
  class SliceAndMaybePadState {
   public:
    int num_complete_pad_dims_;
    int num_partial_pad_dims_;
    TensorShape non_padded_slice_shape_;
    Eigen::array<Eigen::IndexPair<int64_t>, Rank> slice_paddings_;
    Eigen::DSizes<Eigen::DenseIndex, Rank> slice_indices_;
    Eigen::DSizes<Eigen::DenseIndex, Rank> output_slice_shape_dsizes_;
    Eigen::DSizes<Eigen::DenseIndex, Rank> non_padded_slice_shape_dsizes_;

    TF_ATTRIBUTE_NOINLINE SliceAndMaybePadState(
        absl::Span<const int32_t> num_splits,
        const absl::Span<const int64_t> input_shape,
        const TensorShape& output_slice_shape, int slice_index) {
      output_slice_shape_dsizes_ =
          sharding_internal::ShapeAsEigenDSizes<Rank>(output_slice_shape);
      num_complete_pad_dims_ = 0;
      num_partial_pad_dims_ = 0;
      slice_indices_ = GetSliceIndices<Rank>(
          num_splits, output_slice_shape_dsizes_, slice_index);

      // Calculate paddings necessary for slice instead of padding input and
      // slicing subsequently to reduce temporary memory allocation.
      for (int dim = 0; dim < Rank; ++dim) {
        const int64_t dim_size = input_shape[dim];
        const int64_t out_dim = output_slice_shape_dsizes_[dim];
        int64_t non_padded_dim = 0;
        if (slice_indices_[dim] >= dim_size) {
          // Complete padding.
          slice_indices_[dim] = dim_size;
          non_padded_dim = 0;
          slice_paddings_[dim] = {0, out_dim};
          num_complete_pad_dims_++;
        } else if (slice_indices_[dim] + out_dim > dim_size) {
          // Partial padding.
          non_padded_dim = dim_size - slice_indices_[dim];
          slice_paddings_[dim] = {0, out_dim - non_padded_dim};
          num_partial_pad_dims_++;
        } else {
          non_padded_dim = out_dim;
        }
        non_padded_slice_shape_.AddDim(non_padded_dim);
      }
      non_padded_slice_shape_dsizes_ =
          sharding_internal::ShapeAsEigenDSizes<Rank>(non_padded_slice_shape_);
    }
  };

  std::vector<int32_t> num_splits_;
  int num_slices_;
  std::vector<int32_t> paddings_;
  bool has_paddings_;

  explicit XlaNDSplitter(const std::vector<int32_t>& num_splits, int num_slices,
                         const std::vector<int32_t>& paddings,
                         bool has_paddings)
      : num_splits_(num_splits),
        num_slices_(num_slices),
        paddings_(paddings),
        has_paddings_(has_paddings) {}

  void TF_ATTRIBUTE_NOINLINE SetToConstant(Tensor* output_slice,
                                           const Device& device) {
    auto output_flat = output_slice->flat<T>();
    output_flat.device(device) = output_flat.constant(T());
  }

  template <int Rank>
  void TF_ATTRIBUTE_NOINLINE AssignFromInput(
      Tensor* output_slice, const Device& device, const Tensor* input,
      const Eigen::DSizes<Eigen::DenseIndex, Rank>& slice_indices,
      const Eigen::DSizes<Eigen::DenseIndex, Rank>& output_slice_shape_dsizes) {
    output_slice->tensor<T, Rank>().device(device) =
        input->tensor<T, Rank>().slice(slice_indices,
                                       output_slice_shape_dsizes);
  }

  template <int Rank>
  void TF_ATTRIBUTE_NOINLINE
  SliceAndMaybePad(const Device& device, const Tensor* input,
                   const absl::Span<const int64_t> input_shape,
                   const TensorShape& output_slice_shape,
                   const std::vector<Tensor*>& output_slices) {
    const auto& input_tensor = input->tensor<T, Rank>();
    // Slice shape with optional padding.
    for (int i = 0; i < num_slices_; ++i) {
      Tensor* output_slice = output_slices[i];
      SliceAndMaybePadState<Rank> r(num_splits_, input_shape,
                                    output_slice_shape, i);
      if (r.num_complete_pad_dims_ == Rank ||
          (r.num_complete_pad_dims_ > 0 || r.num_partial_pad_dims_ > 0)) {
        // Need to init padding
        SetToConstant(output_slice, device);
      }
      if (r.num_complete_pad_dims_ == Rank) {
        // Done
      } else if (r.num_complete_pad_dims_ > 0 || r.num_partial_pad_dims_ > 0) {
        output_slice->tensor<T, Rank>()
            .slice(Eigen::DSizes<Eigen::DenseIndex, Rank>(),
                   r.non_padded_slice_shape_dsizes_)
            .device(device) = input_tensor.slice(
            r.slice_indices_, r.non_padded_slice_shape_dsizes_);
      } else {
        AssignFromInput<Rank>(output_slice, device, input, r.slice_indices_,
                              r.output_slice_shape_dsizes_);
      }
    }
  }
};

// Shared base class to save code space
template <typename Device, typename T>
class XlaNDConcatenator {
 public:
  static absl::StatusOr<XlaNDConcatenator<Device, T>> Create(
      const std::vector<int32_t>& num_concats, int num_slices,
      const std::vector<int32_t>& paddings, bool has_paddings) {
    if (num_concats.size() != paddings.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("num_concats size ", num_concats.size(),
                       " mismatch with paddings size ", paddings.size(), "."));
    }

    int concats_cnt = 1;
    for (auto concat : num_concats) {
      concats_cnt *= concat;
    }

    if (num_slices != concats_cnt) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expect num_slices ", concats_cnt, " but got ", num_slices));
    }

    return XlaNDConcatenator<Device, T>(num_concats, num_slices, paddings,
                                        has_paddings);
  }
  absl::Status ComputeInternal(
      absl::Span<const Tensor> inputs,
      const std::function<Status(const Tensor&)>& assign_or_copy_value_fn,
      const std::function<absl::StatusOr<Tensor*>()>& get_output_fn,
      const Device& device) {
    const int rank = inputs[0].shape().dims();

    if (rank < 1 || rank > 8) {
      return absl::InvalidArgumentError(absl::StrCat(
          "'inputs' tensors must have rank in range (0, 8], but got ", rank,
          "."));
    }

    if (num_slices_ == 1 && !has_paddings_) {
      // Simple case
      return assign_or_copy_value_fn(inputs[0]);
    }

    TF_ASSIGN_OR_RETURN(Tensor * output, get_output_fn());

    if (rank == 1) {
      MaybeUnpadAndAssign<1>(device, inputs, output);
    } else if (rank == 2) {
      MaybeUnpadAndAssign<2>(device, inputs, output);
    } else if (rank == 3) {
      MaybeUnpadAndAssign<3>(device, inputs, output);
    } else if (rank == 4) {
      MaybeUnpadAndAssign<4>(device, inputs, output);
    } else if (rank == 5) {
      MaybeUnpadAndAssign<5>(device, inputs, output);
    } else if (rank == 6) {
      MaybeUnpadAndAssign<6>(device, inputs, output);
    } else if (rank == 7) {
      MaybeUnpadAndAssign<7>(device, inputs, output);
    } else if (rank == 8) {
      MaybeUnpadAndAssign<8>(device, inputs, output);
    }
    return absl::OkStatus();
  }

 private:
  template <int Rank>
  class MaybeUnpadAndAssignState {
   public:
    int num_complete_pad_dims_;
    int num_partial_pad_dims_;
    TensorShape non_padded_slice_shape_;
    Eigen::DSizes<Eigen::DenseIndex, Rank> slice_shape_dsizes_;
    Eigen::array<Eigen::IndexPair<int64_t>, Rank> slice_paddings_;
    Eigen::DSizes<Eigen::DenseIndex, Rank> slice_indices_;
    Eigen::DSizes<Eigen::DenseIndex, Rank> output_slice_shape_dsizes_;
    Eigen::DSizes<Eigen::DenseIndex, Rank> non_padded_slice_shape_dsizes_;

    TF_ATTRIBUTE_NOINLINE MaybeUnpadAndAssignState(
        absl::Span<const int32_t> num_concats, const Tensor& input0,
        Tensor* output, int slice_index) {
      slice_shape_dsizes_ = input0.shape().AsEigenDSizes<Rank>();
      slice_indices_ =
          GetSliceIndices<Rank>(num_concats, slice_shape_dsizes_, slice_index);
      num_complete_pad_dims_ = 0;
      num_partial_pad_dims_ = 0;
      // Calculate paddings necessary to strip from slice.
      for (int dim = 0; dim < Rank; ++dim) {
        const int64_t dim_size = output->shape().dim_size(dim);
        int64_t non_padded_dim = 0;
        if (slice_indices_[dim] >= dim_size) {
          // Complete padding.
          slice_indices_[dim] = dim_size;
          non_padded_dim = 0;
          num_complete_pad_dims_++;
        } else if (slice_indices_[dim] + slice_shape_dsizes_[dim] > dim_size) {
          // Partial padding.
          non_padded_dim = dim_size - slice_indices_[dim];
          num_partial_pad_dims_++;
        } else {
          non_padded_dim = slice_shape_dsizes_[dim];
        }
        non_padded_slice_shape_.AddDim(non_padded_dim);
      }
      non_padded_slice_shape_dsizes_ =
          non_padded_slice_shape_.AsEigenDSizes<Rank>();
    }
  };

  std::vector<int32_t> num_concats_;
  int num_slices_;
  std::vector<int32_t> paddings_;
  bool has_paddings_;

  explicit TF_ATTRIBUTE_NOINLINE XlaNDConcatenator(
      const std::vector<int32_t>& num_concats, int num_slices,
      const std::vector<int32_t>& paddings, bool has_paddings)
      : num_concats_(num_concats),
        num_slices_(num_slices),
        paddings_(paddings),
        has_paddings_(has_paddings) {}

  template <int Rank>
  void TF_ATTRIBUTE_NOINLINE MaybeUnpadAndAssign(
      const Device& device, absl::Span<const Tensor> inputs, Tensor* output) {
    for (int i = 0; i < num_slices_; ++i) {
      MaybeUnpadAndAssignState<Rank> r(num_concats_, inputs[0], output, i);
      if (r.num_complete_pad_dims_ == Rank) {
        continue;
      } else if (r.num_complete_pad_dims_ > 0 || r.num_partial_pad_dims_ > 0) {
        output->tensor<T, Rank>()
            .slice(r.slice_indices_, r.non_padded_slice_shape_dsizes_)
            .device(device) = inputs[i].tensor<T, Rank>().slice(
            Eigen::DSizes<Eigen::DenseIndex, Rank>(),
            r.non_padded_slice_shape_dsizes_);
      } else {
        output->tensor<T, Rank>()
            .slice(r.slice_indices_, r.slice_shape_dsizes_)
            .device(device) = inputs[i].tensor<T, Rank>();
      }
    }
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_SHARDING_UTILS_H_
