/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <functional>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

constexpr absl::string_view kNumSplitsAttrName = "num_splits";
constexpr absl::string_view kNumConcatsAttrName = "num_concats";

Status GetAndValidateAttributesHelper(bool split, OpKernelConstruction* ctx,
                                      std::vector<int32>& num_partitions,
                                      int& num_slices,
                                      std::vector<int32>& paddings,
                                      bool& has_paddings) {
  absl::string_view num_partitions_attr_name =
      split ? kNumSplitsAttrName : kNumConcatsAttrName;
  TF_RETURN_IF_ERROR(ctx->GetAttr(num_partitions_attr_name, &num_partitions));

  int num_dims_to_split = 0;
  for (int i = 0, e = num_partitions.size(); i < e; ++i) {
    const auto& split = num_partitions[i];
    if (split <= 0) {
      return errors::InvalidArgument("'", num_partitions_attr_name,
                                     "' at index ", i,
                                     " must be positive, but got ", split, ".");
    }
    if (split > 1) {
      ++num_dims_to_split;
    }
    num_slices *= split;
  }

  int n;
  TF_RETURN_IF_ERROR(ctx->GetAttr("N", &n));
  if (n != num_slices) {
    return errors::InvalidArgument(
        "'N' must match number of slices ", num_slices, " from '",
        num_partitions_attr_name, "', but got ", n, ".");
  }

  TF_RETURN_IF_ERROR(ctx->GetAttr("paddings", &paddings));
  const int expected_rank = num_partitions.size();
  if (!paddings.empty()) {
    if (paddings.size() != expected_rank) {
      return errors::InvalidArgument(
          "'paddings' length must match '", num_partitions_attr_name,
          "' length ", expected_rank, ", but got ", paddings.size(), ".");
    }

    for (int dim = 0; dim < expected_rank; ++dim) {
      if (paddings[dim] < 0) {
        return errors::InvalidArgument(
            "'padding' must be all non-negative, but got ", paddings[dim],
            " at index ", dim, ".");
      }
      if (paddings[dim] > 0) {
        has_paddings = true;
      }
    }
  } else {
    paddings.assign(expected_rank, 0);
  }

  return OkStatus();
}

void GetAndValidateAttributes(bool split, OpKernelConstruction* ctx,
                              std::vector<int32>& num_partitions,
                              int& num_slices, std::vector<int32>& paddings,
                              bool& has_paddings) {
  OP_REQUIRES_OK(
      ctx, GetAndValidateAttributesHelper(split, ctx, num_partitions,
                                          num_slices, paddings, has_paddings));
}

absl::string_view kHandle = "handle";
absl::string_view kTensor = "tensor";

template <bool Handle>
Status CreateResourceInvalidDTypeError(const ResourceHandle& handle,
                                       DataType actual_dtype,
                                       DataType expected_dtype) {
  absl::string_view resource_component = Handle ? kHandle : kTensor;
  return errors::InvalidArgument(
      "'T' must match 'resource' variable ", resource_component, " ('",
      handle.name(), "') container ('", handle.container(), "') dtype ",
      DataTypeString(actual_dtype), ", but got ",
      DataTypeString(expected_dtype), ".");
}

// Converts flatten index to start indices (subscript scaled with slice shape)
// for determining where to start a slice in the input tensor.
template <int Rank>
Eigen::DSizes<Eigen::DenseIndex, Rank> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, Rank>& slice_shape, const int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 1> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 1>& slice_shape, const int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 2> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 2>& slice_shape, const int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 3> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_shape, const int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 4> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 4>& slice_shape, const int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 5> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 5>& slice_shape, const int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 6> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 6>& slice_shape, const int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 7> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 7>& slice_shape, const int index);
template <>
Eigen::DSizes<Eigen::DenseIndex, 8> TF_ATTRIBUTE_NOINLINE GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 8>& slice_shape, const int index);

template <int Rank>
Eigen::DSizes<Eigen::DenseIndex, Rank> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, Rank>& slice_shape,
    const int index) {
  return Eigen::DSizes<Eigen::DenseIndex, Rank>();
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 1> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 1>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 1> subscript;
  subscript[0] = index * slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 2> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 2>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 2> subscript;
  subscript[1] = (index % num_partitions[1]) * slice_shape[1];
  subscript[0] = (index / num_partitions[1]) * slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 3> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 3> subscript;
  subscript[2] = (index % num_partitions[2]) * slice_shape[2];
  subscript[1] =
      ((index / num_partitions[2]) % num_partitions[1]) * slice_shape[1];
  subscript[0] =
      (index / (num_partitions[2] * num_partitions[1])) * slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 4> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 4>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 4> subscript;
  subscript[3] = (index % num_partitions[3]) * slice_shape[3];
  subscript[2] =
      ((index / num_partitions[3]) % num_partitions[2]) * slice_shape[2];
  subscript[1] =
      ((index / (num_partitions[3] * num_partitions[2])) % num_partitions[1]) *
      slice_shape[1];
  subscript[0] =
      (index / (num_partitions[3] * num_partitions[2] * num_partitions[1])) *
      slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 5> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 5>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 5> subscript;
  subscript[4] = (index % num_partitions[4]) * slice_shape[4];
  subscript[3] =
      ((index / num_partitions[4]) % num_partitions[3]) * slice_shape[3];
  subscript[2] =
      ((index / (num_partitions[4] * num_partitions[3])) % num_partitions[2]) *
      slice_shape[2];
  subscript[1] =
      ((index / (num_partitions[4] * num_partitions[3] * num_partitions[2])) %
       num_partitions[1]) *
      slice_shape[1];
  subscript[0] = (index / (num_partitions[4] * num_partitions[3] *
                           num_partitions[2] * num_partitions[1])) *
                 slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 6> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 6>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 6> subscript;
  subscript[5] = (index % num_partitions[5]) * slice_shape[5];
  subscript[4] =
      ((index / num_partitions[5]) % num_partitions[4]) * slice_shape[4];
  subscript[3] =
      ((index / (num_partitions[5] * num_partitions[4])) % num_partitions[3]) *
      slice_shape[3];
  subscript[2] =
      ((index / (num_partitions[5] * num_partitions[4] * num_partitions[3])) %
       num_partitions[2]) *
      slice_shape[2];
  subscript[1] = ((index / (num_partitions[5] * num_partitions[4] *
                            num_partitions[3] * num_partitions[2])) %
                  num_partitions[1]) *
                 slice_shape[1];
  subscript[0] =
      (index / (num_partitions[5] * num_partitions[4] * num_partitions[3] *
                num_partitions[2] * num_partitions[1])) *
      slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 7> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 7>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 7> subscript;
  subscript[6] = (index % num_partitions[6]) * slice_shape[6];
  subscript[5] =
      ((index / num_partitions[6]) % num_partitions[5]) * slice_shape[5];
  subscript[4] =
      ((index / (num_partitions[6] * num_partitions[5])) % num_partitions[4]) *
      slice_shape[4];
  subscript[3] =
      ((index / (num_partitions[6] * num_partitions[5] * num_partitions[4])) %
       num_partitions[3]) *
      slice_shape[3];
  subscript[2] = ((index / (num_partitions[6] * num_partitions[5] *
                            num_partitions[4] * num_partitions[3])) %
                  num_partitions[2]) *
                 slice_shape[2];
  subscript[1] =
      ((index / (num_partitions[6] * num_partitions[5] * num_partitions[4] *
                 num_partitions[3] * num_partitions[2])) %
       num_partitions[1]) *
      slice_shape[1];
  subscript[0] =
      (index / (num_partitions[6] * num_partitions[5] * num_partitions[4] *
                num_partitions[3] * num_partitions[2] * num_partitions[1])) *
      slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 8> GetSliceIndices(
    absl::Span<const int32> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 8>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 8> subscript;
  subscript[7] = (index % num_partitions[7]) * slice_shape[7];
  subscript[6] =
      ((index / num_partitions[7]) % num_partitions[6]) * slice_shape[6];
  subscript[5] =
      ((index / (num_partitions[7] * num_partitions[6])) % num_partitions[5]) *
      slice_shape[5];
  subscript[4] =
      ((index / (num_partitions[7] * num_partitions[6] * num_partitions[5])) %
       num_partitions[4]) *
      slice_shape[4];
  subscript[3] = ((index / (num_partitions[7] * num_partitions[6] *
                            num_partitions[5] * num_partitions[4])) %
                  num_partitions[3]) *
                 slice_shape[3];
  subscript[2] =
      ((index / (num_partitions[7] * num_partitions[6] * num_partitions[5] *
                 num_partitions[4] * num_partitions[3])) %
       num_partitions[2]) *
      slice_shape[2];
  subscript[1] =
      ((index / (num_partitions[7] * num_partitions[6] * num_partitions[5] *
                 num_partitions[4] * num_partitions[3] * num_partitions[2])) %
       num_partitions[1]) *
      slice_shape[1];
  subscript[0] =
      (index / (num_partitions[7] * num_partitions[6] * num_partitions[5] *
                num_partitions[4] * num_partitions[3] * num_partitions[2] *
                num_partitions[1])) *
      slice_shape[0];
  return subscript;
}

constexpr absl::string_view kTensorName = "'input' tensor";
constexpr absl::string_view kResourceName = "'resource' variable tensor";

template <int Rank>
Eigen::DSizes<Eigen::DenseIndex, Rank> TF_ATTRIBUTE_NOINLINE
ShapeAsEigenDSizes(const TensorShape& shape);
template <int Rank>
Eigen::DSizes<Eigen::DenseIndex, Rank> ShapeAsEigenDSizes(
    const TensorShape& shape) {
  return shape.AsEigenDSizes<Rank>();
}

bool TF_ATTRIBUTE_NOINLINE ValidateShapesForSlice(
    OpKernelContext* ctx, bool resource, const Tensor* input,
    const std::vector<int32>& num_splits, const std::vector<int32>& paddings);

bool ValidateShapesForSlice(OpKernelContext* ctx, bool resource,
                            const Tensor* input,
                            const std::vector<int32>& num_splits,
                            const std::vector<int32>& paddings) {
  const auto& ishape = input->shape();

  Status s;

  absl::string_view input_name = resource ? kResourceName : kTensorName;
  const int rank = ishape.dims();
  const auto& input_shape = ishape.dim_sizes();
  if (rank <= 0 || rank > 8) {
    s = errors::InvalidArgument(
        input_name, " must have rank in range (0, 8], but got ", rank, ".");
  } else if (rank != num_splits.size()) {
    s = errors::InvalidArgument(
        input_name, " rank must be the same as 'num_splits' length ",
        num_splits.size(), ", but got rank ", rank, ".");
  } else {
    for (int dim = 0; dim < rank; ++dim) {
      const auto input_shape_dim = input_shape[dim];
      const auto paddings_dim = paddings[dim];
      const auto num_splits_dim = num_splits[dim];
      if ((input_shape_dim + paddings_dim) % num_splits_dim != 0) {
        s = errors::InvalidArgument(
            input_name, " shape dimension ", dim, " (", input_shape_dim,
            ") with padding ", paddings_dim,
            " must be evenly divisible by 'num_splits' ", num_splits_dim, ".");
        break;
      }
    }
  }
  if (!s.ok()) {
    ctx->CtxFailure(__FILE__, __LINE__, s);
    return false;
  }
  return true;
}

// Shared base class to save code space
class XlaSplitNDShared : public OpKernel {
 public:
  explicit TF_ATTRIBUTE_NOINLINE XlaSplitNDShared(OpKernelConstruction* ctx)
      : OpKernel(ctx), num_slices_(1), has_paddings_(false) {
    GetAndValidateAttributes(/*split=*/true, ctx, num_splits_, num_slices_,
                             paddings_, has_paddings_);
  }

 protected:
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
        absl::Span<const int32> num_splits,
        const absl::Span<const int64_t> input_shape,
        const TensorShape& output_slice_shape, int slice_index) {
      output_slice_shape_dsizes_ = ShapeAsEigenDSizes<Rank>(output_slice_shape);
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
          ShapeAsEigenDSizes<Rank>(non_padded_slice_shape_);
    }
  };

  static void TF_ATTRIBUTE_NOINLINE GetDtypeHelper(OpKernelConstruction* ctx,
                                                   const char* attr_name,
                                                   DataType* dtype_ptr) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(attr_name, dtype_ptr));
  }

  std::vector<int32> num_splits_;
  int num_slices_;
  std::vector<int32> paddings_;
  bool has_paddings_;
};

template <typename Device, typename T>
class XlaSplitNDBaseOp : public XlaSplitNDShared {
 public:
  explicit XlaSplitNDBaseOp(OpKernelConstruction* ctx)
      : XlaSplitNDShared(ctx) {}

 protected:
  void ComputeInternal(
      bool resource, OpKernelContext* ctx,
      const std::function<Status(const Tensor&)>& assign_or_copy_value_fn,
      const Tensor* input) {
    const int rank = input->shape().dims();
    const auto& input_shape = input->shape().dim_sizes();

    if (!ValidateShapesForSlice(ctx, resource, input, num_splits_, paddings_)) {
      return;
    }

    TensorShape output_slice_shape;
    for (int i = 0; i < rank; ++i) {
      output_slice_shape.AddDim((input_shape[i] + paddings_[i]) /
                                ((num_slices_ == 1) ? 1 : num_splits_[i]));
    }
    if (num_slices_ == 1 && !has_paddings_) {
      // Handle simple case first
      OP_REQUIRES_OK(ctx, assign_or_copy_value_fn(*input));
    } else {
      const Device& device = ctx->eigen_device<Device>();
      std::vector<Tensor*> output_slices(num_slices_);
      for (int i = 0; i < num_slices_; i++) {
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_output(
                           /*index=*/i, output_slice_shape, &output_slices[i]));
      }

      if (rank == 1) {
        SliceAndMaybePad<1>(ctx, device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 2) {
        SliceAndMaybePad<2>(ctx, device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 3) {
        SliceAndMaybePad<3>(ctx, device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 4) {
        SliceAndMaybePad<4>(ctx, device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 5) {
        SliceAndMaybePad<5>(ctx, device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 6) {
        SliceAndMaybePad<6>(ctx, device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 7) {
        SliceAndMaybePad<7>(ctx, device, input, input_shape, output_slice_shape,
                            output_slices);
      } else if (rank == 8) {
        SliceAndMaybePad<8>(ctx, device, input, input_shape, output_slice_shape,
                            output_slices);
      }
      return;
    }
  }

 private:
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
  void TF_ATTRIBUTE_NOINLINE SliceAndMaybePad(
      OpKernelContext* ctx, const Device& device, const Tensor* input,
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

template <typename Device, typename T>
class XlaSplitNDOp : public XlaSplitNDBaseOp<Device, T> {
 public:
  explicit TF_ATTRIBUTE_NOINLINE XlaSplitNDOp(OpKernelConstruction* ctx)
      : XlaSplitNDBaseOp<Device, T>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    auto assign_or_copy_value_fn = [&ctx](const Tensor& input) -> Status {
      ctx->set_output(/*index=*/0, input);
      return OkStatus();
    };

    this->ComputeInternal(/*resource=*/false, ctx, assign_or_copy_value_fn,
                          &input);
  }
};

template <typename Device, typename T>
class ReadVariableXlaSplitNDOp : public XlaSplitNDBaseOp<Device, T> {
 public:
  explicit TF_ATTRIBUTE_NOINLINE ReadVariableXlaSplitNDOp(
      OpKernelConstruction* ctx)
      : XlaSplitNDBaseOp<Device, T>(ctx) {
    XlaSplitNDShared::GetDtypeHelper(ctx, "T", &dtype_);
  }

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> variable;
    const ResourceHandle& handle = HandleFromInput(ctx, 0);
    const Status status = LookupResource(ctx, handle, &variable);
    OP_REQUIRES(
        ctx, status.ok(),
        errors::InvalidArgument("'resource' variable handle ('", handle.name(),
                                "') container ('", handle.container(),
                                "') cannot be found."));

    tf_shared_lock ml(*variable->mu());
    const Tensor* input = variable->tensor();
    OP_REQUIRES(
        ctx, input->dtype() == dtype_,
        CreateResourceInvalidDTypeError<false>(handle, input->dtype(), dtype_));

    auto assign_or_copy_value_fn = [&ctx,
                                    &variable](const Tensor& input) -> Status {
      if (variable->copy_on_read_mode.load()) {
        Tensor* output;
        TF_RETURN_IF_ERROR(
            ctx->allocate_output(/*index=*/0, input.shape(), &output));
        output->flat<T>().device(ctx->eigen_device<Device>()) = input.flat<T>();
      } else {
        ctx->set_output(/*index=*/0, input);
      }
      return OkStatus();
    };

    this->ComputeInternal(/*resource=*/true, ctx, assign_or_copy_value_fn,
                          input);
  }

 private:
  DataType dtype_;
};

#define REGISTER_XLA_SPLIT_ND(type)                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("XlaSplitND").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      XlaSplitNDOp<Eigen::ThreadPoolDevice, type>)

TF_CALL_POD_TYPES(REGISTER_XLA_SPLIT_ND);
TF_CALL_QUANTIZED_TYPES(REGISTER_XLA_SPLIT_ND);
#undef REGISTER_XLA_SPLIT_ND

#define REGISTER_READ_VARIABLE_XLA_SPLIT_ND(type) \
  REGISTER_KERNEL_BUILDER(                        \
      Name("ReadVariableXlaSplitND")              \
          .Device(DEVICE_CPU)                     \
          .TypeConstraint<type>("T"),             \
      ReadVariableXlaSplitNDOp<Eigen::ThreadPoolDevice, type>)

TF_CALL_POD_TYPES(REGISTER_READ_VARIABLE_XLA_SPLIT_ND);
TF_CALL_QUANTIZED_TYPES(REGISTER_READ_VARIABLE_XLA_SPLIT_ND);
#undef REGISTER_READ_VARIABLE_XLA_SPLIT_ND

// Shared base class to save code space
class XlaConcatNDShared : public OpKernel {
 public:
  explicit TF_ATTRIBUTE_NOINLINE XlaConcatNDShared(OpKernelConstruction* ctx)
      : OpKernel(ctx), num_slices_(1), has_paddings_(false) {
    GetAndValidateAttributes(/*split=*/false, ctx, num_concats_, num_slices_,
                             paddings_, has_paddings_);
  }

 protected:
  Status GetInputsAndOutputShape(OpKernelContext* ctx, OpInputList& inputs,
                                 TensorShape& output_shape) {
    TF_RETURN_IF_ERROR(ctx->input_list("inputs", &inputs));
    DCHECK_EQ(inputs.size(), num_slices_);

    const TensorShape& slice_shape = inputs[0].shape();
    if (slice_shape.dims() != num_concats_.size()) {
      return errors::InvalidArgument(
          "'inputs' rank must be the same as 'num_concats' length ",
          num_concats_.size(), ", but got rank ", slice_shape.dims(), ".");
    }
    for (int i = 1; i < num_slices_; ++i) {
      const TensorShape& slice_shape_i = inputs[i].shape();
      if (slice_shape != slice_shape_i) {
        return errors::InvalidArgument(
            "'inputs' must all have the same expected shape ", slice_shape,
            ", but got ", slice_shape_i, " at index ", i, ".");
      }
    }

    for (int i = 0, e = num_concats_.size(); i < e; ++i) {
      const int max_dim_size = slice_shape.dim_size(i) * num_concats_[i];
      if (paddings_[i] > max_dim_size) {
        return errors::InvalidArgument(
            "'paddings' must not exceed expected output shape dimension ",
            max_dim_size, " at index ", i, ", but got ", paddings_[i], ".");
      }
      TF_RETURN_IF_ERROR(
          output_shape.AddDimWithStatus(max_dim_size - paddings_[i]));
    }

    return OkStatus();
  }
  void ApplyAssignOrCopyShared(
      OpKernelContext* ctx,
      const std::function<Status(const Tensor&)>& assign_or_copy_value_fn,
      const Tensor& input) {
    OP_REQUIRES_OK(ctx, assign_or_copy_value_fn(input));
  }

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
        absl::Span<const int32> num_concats, const Tensor& input0,
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

  std::vector<int32> num_concats_;
  int num_slices_;
  std::vector<int32> paddings_;
  bool has_paddings_;
};

template <typename Device, typename T>
class XlaConcatNDBaseOp : public XlaConcatNDShared {
 public:
  explicit TF_ATTRIBUTE_NOINLINE XlaConcatNDBaseOp(OpKernelConstruction* ctx)
      : XlaConcatNDShared(ctx) {}

 protected:
  void ComputeInternal(
      bool resource, OpKernelContext* ctx, const OpInputList& inputs,
      const std::function<Status(const Tensor&)>& assign_or_copy_value_fn,
      const std::function<StatusOr<Tensor*>()>& get_output_fn) {
    const int rank = inputs[0].shape().dims();

    OP_REQUIRES(ctx, rank > 0 && rank <= 8,
                errors::InvalidArgument(
                    "'inputs' tensors must have rank in range (0, 8], but got ",
                    rank, "."));

    if (num_slices_ == 1 && !has_paddings_) {
      // Simple case
      ApplyAssignOrCopyShared(ctx, assign_or_copy_value_fn, inputs[0]);
      return;
    }

    const Device& device = ctx->eigen_device<Device>();
    auto status_or_output = get_output_fn();
    OP_REQUIRES_OK(ctx, status_or_output.status());
    Tensor* output = std::move(status_or_output).value();

    if (rank == 1) {
      MaybeUnpadAndAssign<1>(ctx, device, inputs, output);
    } else if (rank == 2) {
      MaybeUnpadAndAssign<2>(ctx, device, inputs, output);
    } else if (rank == 3) {
      MaybeUnpadAndAssign<3>(ctx, device, inputs, output);
    } else if (rank == 4) {
      MaybeUnpadAndAssign<4>(ctx, device, inputs, output);
    } else if (rank == 5) {
      MaybeUnpadAndAssign<5>(ctx, device, inputs, output);
    } else if (rank == 6) {
      MaybeUnpadAndAssign<6>(ctx, device, inputs, output);
    } else if (rank == 7) {
      MaybeUnpadAndAssign<7>(ctx, device, inputs, output);
    } else if (rank == 8) {
      MaybeUnpadAndAssign<8>(ctx, device, inputs, output);
    }
  }

 private:
  template <int Rank>
  void TF_ATTRIBUTE_NOINLINE MaybeUnpadAndAssign(OpKernelContext* ctx,
                                                 const Device& device,
                                                 const OpInputList& inputs,
                                                 Tensor* output) {
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

template <typename Device, typename T>
class XlaConcatNDOp : public XlaConcatNDBaseOp<Device, T> {
 public:
  explicit XlaConcatNDOp(OpKernelConstruction* ctx)
      : XlaConcatNDBaseOp<Device, T>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OpInputList inputs;
    TensorShape output_shape;
    OP_REQUIRES_OK(ctx,
                   this->GetInputsAndOutputShape(ctx, inputs, output_shape));

    auto assign_or_copy_value_fn = [&ctx](const Tensor& input) -> Status {
      ctx->set_output(/*index=*/0, input);
      return OkStatus();
    };

    auto get_output_fn = [&ctx, &output_shape]() -> StatusOr<Tensor*> {
      Tensor* output = nullptr;
      TF_RETURN_IF_ERROR(
          ctx->allocate_output(/*index=*/0, output_shape, &output));
      return output;
    };
    this->ComputeInternal(/*resource=*/false, ctx, inputs,
                          assign_or_copy_value_fn, get_output_fn);
  }
};

template <typename Device, typename T>
class AssignVariableXlaConcatNDOp : public XlaConcatNDBaseOp<Device, T> {
 public:
  explicit TF_ATTRIBUTE_NOINLINE AssignVariableXlaConcatNDOp(
      OpKernelConstruction* ctx)
      : XlaConcatNDBaseOp<Device, T>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    OpInputList inputs;
    TensorShape output_shape;
    OP_REQUIRES_OK(ctx,
                   this->GetInputsAndOutputShape(ctx, inputs, output_shape));

    core::RefCountPtr<Var> variable;
    const ResourceHandle& handle = HandleFromInput(ctx, 0);
    if (handle.dtypes_and_shapes().size() == 1) {
      const DtypeAndPartialTensorShape dtype_and_shape =
          handle.dtypes_and_shapes().front();
      OP_REQUIRES(ctx, dtype_and_shape.dtype == dtype_,
                  CreateResourceInvalidDTypeError<true>(
                      handle, dtype_and_shape.dtype, dtype_));
      OP_REQUIRES(ctx, dtype_and_shape.shape.IsCompatibleWith(output_shape),
                  errors::InvalidArgument(
                      "'resource' variable handle ('", handle.name(),
                      "') container ('", handle.container(),
                      "') shape must be compatible with expected shape ",
                      output_shape, ", but got ", dtype_and_shape.shape, "."));
    }
    OP_REQUIRES_OK(ctx, LookupOrCreateResource<Var>(ctx, handle, &variable,
                                                    [this](Var** ptr) {
                                                      *ptr = new Var(dtype_);
                                                      return OkStatus();
                                                    }));
    mutex_lock ml(*variable->mu());

    OP_REQUIRES(ctx, variable->tensor()->dtype() == dtype_,
                CreateResourceInvalidDTypeError<false>(
                    handle, variable->tensor()->dtype(), dtype_));

    auto assign_or_copy_value_fn = [this, &ctx, &output_shape,
                                    &variable](const Tensor& input) -> Status {
      if (variable->copy_on_read_mode.load()) {
        TF_RETURN_IF_ERROR(
            ctx->allocate_temp(dtype_, output_shape, variable->tensor()));
        variable->tensor()->flat<T>().device(ctx->eigen_device<Device>()) =
            input.flat<T>();
      } else {
        *variable->tensor() = input;
      }
      return OkStatus();
    };

    auto get_output_fn = [this, &ctx, &output_shape,
                          &variable]() -> StatusOr<Tensor*> {
      if (variable->copy_on_read_mode.load() ||
          !variable->tensor()->RefCountIsOne() ||
          !variable->tensor()->shape().IsSameSize(output_shape)) {
        TF_RETURN_IF_ERROR(
            ctx->allocate_temp(dtype_, output_shape, variable->tensor()));
      }
      return variable->tensor();
    };

    this->ComputeInternal(/*resource=*/true, ctx, inputs,
                          assign_or_copy_value_fn, get_output_fn);
    variable->is_initialized = true;
  }

  DataType dtype_;
};

#define REGISTER_XLA_CONCAT_ND(type)                                    \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("XlaConcatND").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      XlaConcatNDOp<Eigen::ThreadPoolDevice, type>)

TF_CALL_POD_TYPES(REGISTER_XLA_CONCAT_ND);
TF_CALL_QUANTIZED_TYPES(REGISTER_XLA_CONCAT_ND);
#undef REGISTER_XLA_CONCAT_ND

#define REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND(type) \
  REGISTER_KERNEL_BUILDER(                           \
      Name("AssignVariableXlaConcatND")              \
          .Device(DEVICE_CPU)                        \
          .TypeConstraint<type>("T"),                \
      AssignVariableXlaConcatNDOp<Eigen::ThreadPoolDevice, type>)

TF_CALL_POD_TYPES(REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND);
TF_CALL_QUANTIZED_TYPES(REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND);
#undef REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND

}  // anonymous namespace
}  // namespace tensorflow
