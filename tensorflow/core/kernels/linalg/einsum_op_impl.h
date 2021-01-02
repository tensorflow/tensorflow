/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_IMPL_H_

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/linalg/einsum_op.h"
#include "tensorflow/core/kernels/matmul_op_impl.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/einsum_op_util.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/reduction_ops_common_gpu.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

using ShapeVec = gtl::InlinedVector<int64, 8>;
using Labels = gtl::InlinedVector<int, 8>;
using OperandLabels = gtl::InlinedVector<Labels, 2>;
using LabelCounts = gtl::InlinedVector<int, 8>;
using OperandLabelCounts = gtl::InlinedVector<LabelCounts, 2>;
using LabelToDimSizes = gtl::InlinedVector<int64, 8>;

// Dummy axis label used to denote an ellipsis in an input or output subscript.
constexpr int kEllipsisLabel = -1;

struct EinsumHelper {
  // Each dimension is categorized into exactly one of five types based on
  // whether its corresponding label is present in the input and/or the output
  // subscripts.
  enum DimensionType {
    // Batch dimensions are those present in two inputs as well as the output.
    // They are part of the batch dimensions during Tensor contraction.
    // Such dimensions may be broadcasting dimensions (those mapping to
    // ellipsis)
    // or explicit batch dimensions corresponding to named axis labels.
    kBroadcasting = 0,
    kBatch = 1,
    // Free dimensions are present in exactly one of the inputs, and also the
    // output. These are non-contracted axes in the Tensor contraction.
    kFree = 2,
    // Contract dimensions are present in two inputs, but not the output. These
    // dimensions are contracted in Tensor contraction.
    kContract = 3,
    // Reduce dimensions are present in exactly one input; and not in the output
    // and are summed over prior to Tensor contraction.
    kReduce = 4,
  };

  // Returns the DimensionType given whether the corresponding label is present
  // in exactly one input subscript (is_unique) and whether it is absent from
  // the output subscripts (is_removed). Does not handle broadcasting
  // dimensions.
  static DimensionType GetDimensionType(bool is_removed, bool is_unique) {
    if (!is_removed && !is_unique)
      return kBatch;
    else if (!is_removed && is_unique)
      return kFree;
    else if (is_removed && !is_unique)
      return kContract;
    else  // is_removed && is_unique
      return kReduce;
  }

  // Maps the character labels to consecutive integers.
  static void MapToLabels(const string& subscript, Labels* labels,
                          absl::flat_hash_map<char, int>* label_mapping) {
    for (int i = 0; i < subscript.size(); ++i) {
      const char label_char = subscript[i];
      if (label_char == '.') {
        labels->push_back(kEllipsisLabel);
        i += 2;  // Skip next 2 characters as well.
        continue;
      }
      if (!label_mapping->contains(label_char)) {
        const int next_label = label_mapping->size();
        (*label_mapping)[label_char] = next_label;
      }
      const int mapped_label = (*label_mapping)[label_char];
      labels->push_back(mapped_label);
    }
  }

  // Parses and validates the equation and the input shapes. Single character
  // labels are integerized and we populate input and output label subscripts
  // and corresponding counts. Also create the mapping from (named) labels to
  // their DimensionType.
  static Status ParseEquation(const string& equation,
                              OperandLabels* input_labels,
                              Labels* output_labels,
                              std::vector<DimensionType>* label_types,
                              OperandLabelCounts* input_label_counts,
                              LabelCounts* output_label_counts,
                              gtl::InlinedVector<bool, 2>* input_has_ellipsis,
                              bool* output_has_ellipsis) {
    gtl::InlinedVector<string, 2> input_str;
    string output_str;
    TF_RETURN_IF_ERROR(ParseEinsumEquation(equation, &input_str, &output_str));

    // Temporary map from single character labels to (consecutive) integer
    // labels.
    absl::flat_hash_map<char, int> label_mapping;
    int num_inputs = input_str.size();
    input_labels->resize(num_inputs);

    // Map from single characters to integer labels.
    for (int i = 0; i < num_inputs; ++i) {
      MapToLabels(input_str[i], &input_labels->at(i), &label_mapping);
    }
    MapToLabels(output_str, output_labels, &label_mapping);

    // Compute counts for input and output labels.
    int num_labels = label_mapping.size();
    input_label_counts->resize(num_inputs);
    input_has_ellipsis->resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      input_label_counts->at(i).resize(num_labels);
      for (const int label : input_labels->at(i)) {
        if (label != kEllipsisLabel)
          input_label_counts->at(i)[label] += 1;
        else
          input_has_ellipsis->at(i) = true;
      }
    }
    output_label_counts->resize(num_labels);
    for (const int label : *output_labels) {
      if (label != kEllipsisLabel)
        output_label_counts->at(label) += 1;
      else
        *output_has_ellipsis = true;
    }

    // Map each label to a unique DimensionType.
    label_types->resize(num_labels);
    for (int label = 0; label < num_labels; ++label) {
      if (label == kEllipsisLabel) continue;
      bool removed = (*output_label_counts)[label] == 0;
      bool unique = num_inputs == 1 || (*input_label_counts)[0][label] == 0 ||
                    (*input_label_counts)[1][label] == 0;
      (*label_types)[label] = GetDimensionType(removed, unique);
    }
    return Status::OK();
  }

  // Insert new (unnamed) broadcasting labels at the location of ellipsis.
  static void InsertBroadcastLabels(int num_bcast_dims, int num_named_labels,
                                    int ellipsis_axis, Labels* labels,
                                    LabelCounts* label_counts) {
    labels->erase(labels->begin() + ellipsis_axis);
    labels->insert(labels->begin() + ellipsis_axis, num_bcast_dims, 0);
    std::iota(labels->begin() + ellipsis_axis,
              labels->begin() + ellipsis_axis + num_bcast_dims,
              num_named_labels);
    // Increment label counts. Since these are new labels, the count is set
    // to 1.
    label_counts->resize(num_named_labels + num_bcast_dims, 1);
  }

  // Record and validate the label to dimension mapping. Must be a named
  // (non-broadcasting) label as broadcasting labels don't have a fixed
  // dimension.
  static Status RecordLabelToDimension(const int label, const int axis,
                                       const Tensor& input,
                                       LabelToDimSizes* label_to_dim_sizes) {
    const int64 input_dim = input.dim_size(axis);
    // We know that label_to_dim_sizes has the size to accommodate named labels.
    if (label_to_dim_sizes->at(label) != 0 &&
        label_to_dim_sizes->at(label) != input_dim) {
      return errors::InvalidArgument(
          "Expected dimension ", label_to_dim_sizes->at(label), " at axis ",
          axis, " of the input shaped ", input.shape().DebugString(),
          " but got dimension ", input_dim);
    }
    (*label_to_dim_sizes)[label] = input_dim;
    return Status::OK();
  }

  // Validate input dimensions and populate unnamed labels and their label
  // counts.
  static Status ProcessDimensions(
      const OpInputList& inputs,
      const gtl::InlinedVector<bool, 2>& input_has_ellipsis,
      const bool output_has_ellipsis, OperandLabels* input_labels,
      Labels* output_labels, std::vector<DimensionType>* label_types,
      OperandLabelCounts* input_label_counts, LabelCounts* output_label_counts,
      LabelToDimSizes* label_to_dim_sizes) {
    if (inputs.size() != input_labels->size()) {
      return errors::InvalidArgument("Expected ", input_labels->size(),
                                     " inputs but got: ", inputs.size());
    }
    const int num_inputs = inputs.size();

    // We infer the number of broadcasting dimensions by taking the maximum rank
    // among the broadcasting subshapes of the input.
    int max_bcast_dims = 0;
    const int num_named_labels = label_types->size();
    label_to_dim_sizes->resize(num_named_labels);
    for (int i = 0; i < num_inputs; ++i) {
      Labels* labels = &(*input_labels)[i];

      if (!input_has_ellipsis[i]) {
        if (inputs[i].dims() != labels->size()) {
          return errors::InvalidArgument("Expected input ", i, " to have rank ",
                                         labels->size(),
                                         " but got: ", inputs[i].dims());
        }
        for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
          const int label = (*labels)[label_idx];
          TF_RETURN_IF_ERROR(RecordLabelToDimension(label, label_idx, inputs[i],
                                                    label_to_dim_sizes));
        }
        continue;
      }

      // Input has an ellipsis.
      if (inputs[i].dims() + 1 < labels->size()) {
        return errors::InvalidArgument(
            "Expected input ", i, " to have rank at least ", labels->size() - 1,
            " but got: ", inputs[i].dims());
      }
      int ellipsis_axis = -1;
      const int num_bcast_dims = inputs[i].dims() - labels->size() + 1;
      for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
        const int label = (*labels)[label_idx];
        if (label == kEllipsisLabel) {
          ellipsis_axis = label_idx;
          continue;
        }
        // Current label is not an ellipsis.
        const int axis =
            label_idx + (ellipsis_axis == -1 ? 0 : num_bcast_dims - 1);
        TF_RETURN_IF_ERROR(
            RecordLabelToDimension(label, axis, inputs[i], label_to_dim_sizes));
      }
      // Found an ellipsis. Replace 'kEllipsisLabel' with broadcasting
      // dimensions.
      if (ellipsis_axis != -1) {
        InsertBroadcastLabels(num_bcast_dims, num_named_labels, ellipsis_axis,
                              labels, &input_label_counts->at(i));
        max_bcast_dims = std::max(max_bcast_dims, num_bcast_dims);
      }
    }
    if (!absl::c_linear_search(input_has_ellipsis, true) &&
        !output_has_ellipsis) {
      return Status::OK();
    }
    // Insert broadcasting dimensions in the output labels.
    auto it =
        std::find(output_labels->begin(), output_labels->end(), kEllipsisLabel);
    if (it != output_labels->end()) {
      const int ellipsis_axis = it - output_labels->begin();
      InsertBroadcastLabels(max_bcast_dims, num_named_labels, ellipsis_axis,
                            output_labels, output_label_counts);
    } else if (max_bcast_dims > 0) {
      return errors::InvalidArgument(
          "Output contains ", max_bcast_dims,
          " broadcasting dimension(s) but no ellipsis "
          "(...) was found in the output subscripts.");
    }
    // Populate DimensionType for the new broadcasting labels.
    label_types->resize(num_named_labels + max_bcast_dims, kBroadcasting);
    return Status::OK();
  }

  // Permutes the labels according to the given permutation.
  static void PermuteLabels(const std::vector<int>& permutation,
                            Labels* labels) {
    Labels permuted_labels(labels->size());
    for (int i = 0; i < labels->size(); ++i) {
      permuted_labels[i] = (*labels)[permutation[i]];
    }
    labels->swap(permuted_labels);
  }

  // Returns a reshaped input Tensor. The underlying buffer is not copied.
  static Status CopyFrom(const Tensor& input, const TensorShape& shape,
                         Tensor* output) {
    if (output->CopyFrom(input, shape)) return Status::OK();
    return errors::Internal(
        "Encountered error while reshaping a Tensor of shape ",
        input.shape().DebugString(), " to shape ", shape.DebugString());
  }

  // Returns whether transposing would be a no-op; whether input has rank < 2 or
  // the permutation is the identity permutation.
  static bool ShouldTranspose(const TensorShape& input_shape,
                              const std::vector<int>& permutation) {
    if (input_shape.dims() < 2) return false;
    for (int i = 0; i < permutation.size(); ++i) {
      if (permutation[i] != i) return true;
    }
    return false;
  }

  // Transpose the input given a permutation. Returns a reference to the input
  // if transposing is not necessary.
  template <typename Device, typename T>
  static Status TransposeOperand(OpKernelContext* ctx, const Tensor& input,
                                 const std::vector<int>& permutation,
                                 Tensor* output) {
    if (!ShouldTranspose(input.shape(), permutation)) {
      return CopyFrom(input, input.shape(), output);
    }
    TensorShape transposed_shape;
    for (int i = 0; i < input.dims(); ++i) {
      transposed_shape.AddDim(input.dim_size(permutation[i]));
    }
    // For empty Tensors, just change the shape. E.g. we may need to transpose
    // from shape [1, 0, 5] to [5, 1, 0].
    if (input.NumElements() == 0) {
      return CopyFrom(input, transposed_shape, output);
    }
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, transposed_shape, output));
    const Device& device = ctx->eigen_device<Device>();
    TF_RETURN_IF_ERROR(DoTranspose(device, input, permutation, output));
    return Status::OK();
  }

  // If there are repeated labels in either the input or output, then this
  // strides the input (e.g. iii->i) or inflates it (e.g. i->iii), respectively.
  template <typename Device, typename T>
  static Status StrideOrInflate(OpKernelContext* ctx, const Tensor& input,
                                const Labels& labels,
                                const LabelCounts& label_counts,
                                const bool should_inflate, Tensor* output) {
    // Return early if there are no repeated indices.
    if (absl::c_all_of(label_counts, [](int c) { return c <= 1; })) {
      return CopyFrom(input, input.shape(), output);
    }
    // We reshape so that each repeated label is compressed to one dimension.
    // E.g. For iiij -> ij, The shape [3, 3, 3, 5] would be compressed to [27,
    // 5]. Striding appropriately (in this case with strides 14 (=1+3+9) and 1)
    // recovers the generalized diagonal of shape [3, 5].
    ShapeVec reshape;
    ShapeVec strides;
    // Strided and inflated shapes correspond to input and output shapes,
    // respectively, should_inflate is true (vice-versa if should_inflate is
    // false). E.g. they are [3, 5] and [3, 3, 3, 5] in the above example.
    ShapeVec strided_shape;
    ShapeVec inflated_shape;
    for (int label : labels) {
      const int count = label_counts[label];
      const int current_axis =
          should_inflate ? strided_shape.size() : inflated_shape.size();
      const int64 dim = input.dim_size(current_axis);
      strided_shape.push_back(dim);
      inflated_shape.insert(inflated_shape.end(), count, dim);
      const int64 reshape_dim = MathUtil::IPow(dim, count);
      reshape.push_back(reshape_dim);
      // While taking the d-diagonal in a rank k Tensor, we take d
      // equally-spaced elements including the first and last element. Then, (k
      // - 1) * stride = d^k - 1, or, stride = (d^k - 1)/(d - 1).
      const int64 stride =
          (dim > 1 && count > 1) ? (reshape_dim - 1) / (dim - 1) : 1;
      strides.push_back(stride);
    }

    TensorShape output_shape =
        TensorShape(should_inflate ? inflated_shape : strided_shape);
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    const Device& device = ctx->eigen_device<Device>();
    switch (reshape.size()) {
#define NDIMS_CASE(N)                                                 \
  case N: {                                                           \
    if (should_inflate) {                                             \
      auto output_map = output->shaped<T, N>(reshape);                \
      auto input_map = input.shaped<T, N>(strided_shape);             \
      functor::InflateFunctor<Device, T, N>()(                        \
          device, input_map, TensorShape(strides).AsEigenDSizes<N>(), \
          output_map);                                                \
    } else {                                                          \
      auto input_map = input.shaped<T, N>(reshape);                   \
      auto output_map = output->shaped<T, N>(strided_shape);          \
      functor::StrideFunctor<Device, T, N>()(                         \
          device, input_map, TensorShape(strides).AsEigenDSizes<N>(), \
          output_map);                                                \
    }                                                                 \
  } break;
      NDIMS_CASE(1);
      NDIMS_CASE(2);
      NDIMS_CASE(3);
      NDIMS_CASE(4);
      NDIMS_CASE(5);
      NDIMS_CASE(6);
      default:
        return errors::Unimplemented(
            "Unsupported rank: ", reshape.size(),
            " while handling repeated indices. Up to rank 6 is supported.");
#undef NDIMS_CASE
    }
    return Status::OK();
  }

  // Returns true if the input dimensions are already sorted in the order
  // [batch, contract, free, reduce]. Used to implement an optimization to avoid
  // an extra transpose and instead uses (adj_x and adj_y) in BatchMatMul.
  static bool ShouldSwapFreeAndContract(
      const Labels& labels, const std::vector<DimensionType>& label_types) {
    // Check that ordering is according to dimension type, with the role of
    // free and contract dimensions swapped.
    gtl::InlinedVector<int, 5> remap = {0, 1, 3, 2, 4};
    for (int i = 0; i + 1 < labels.size(); ++i) {
      const int dimtype_a = remap[label_types[labels[i]]];
      const int dimtype_b = remap[label_types[labels[i + 1]]];
      if (dimtype_a > dimtype_b ||
          (dimtype_a == dimtype_b && labels[i] > labels[i + 1])) {
        return false;
      }
    }
    return true;
  }

  template <typename Device, typename T>
  static Status ReduceOperand(OpKernelContext* ctx, const Tensor& input,
                              const std::vector<DimensionType>& label_types,
                              const LabelCounts& label_counts, Labels* labels,
                              Labels* free_labels, bool* swap_free_and_contract,
                              Tensor* output) {
    // Find the permutation to transpose the input dimensions in the order of
    // DimensionType; i.e. batch, free, contract and reduce dimensions. This
    // makes it more convenient to invoke Reduce/Contract operations.
    std::vector<int> permutation(input.dims());
    absl::c_iota(permutation, 0);
    Tensor input_transposed;
    // Check if we can avoid the transpose. We need to flip the adj_x (or adj_y)
    // flag during BatchMatMul. This is an extra optimization not necessary for
    // correctness.
    if (ShouldSwapFreeAndContract(*labels, label_types)) {
      *swap_free_and_contract = true;
    } else {
      absl::c_sort(permutation, [&](int i, int j) {
        int label_i = (*labels)[i];
        int label_j = (*labels)[j];
        return std::tie(label_types[label_i], label_i) <
               std::tie(label_types[label_j], label_j);
      });
    }
    // Transpose the input so that DimensionTypes are in order.
    TF_RETURN_IF_ERROR(TransposeOperand<Device, T>(ctx, input, permutation,
                                                   &input_transposed));
    PermuteLabels(permutation, labels);

    // Take the generalized diagonal for dimensions with repeated axis labels.
    Tensor input_deduped;
    labels->erase(std::unique(labels->begin(), labels->end()), labels->end());
    TF_RETURN_IF_ERROR(
        StrideOrInflate<Device, T>(ctx, input_transposed, *labels, label_counts,
                                   false /* should_inflate */, &input_deduped));

    // Reshape denotes the rank-5 shape [broadcast, batch, free, contract,
    // reduce] where we've compacted the dimensions of each DimensionType.
    gtl::InlinedVector<int64, 5> reshape(5, 1);
    // The output shape is [batch shape] + [free size, contract size]
    // That is, the batch shape is preserved (for broadcasting while
    // contracting) while the free dims and contract dims are compressed to one
    // dimension each.
    TensorShape output_shape;
    for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
      const int label = labels->at(label_idx);
      int64 dim = input_deduped.dim_size(label_idx);
      if (label_types[label] == kBroadcasting || label_types[label] == kBatch) {
        output_shape.AddDim(dim);
      } else if (label_types[label] == kFree) {
        free_labels->push_back(label);
      }
      reshape[label_types[label]] *= dim;
    }
    if (*swap_free_and_contract) std::swap(reshape[kFree], reshape[kContract]);
    output_shape.AddDim(reshape[kFree]);
    output_shape.AddDim(reshape[kContract]);

    if (reshape[kReduce] == 1) {  // No need to actually reduce.
      return CopyFrom(input_deduped, output_shape, output);
    }
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    using Reducer = Eigen::internal::SumReducer<T>;
    using Index = typename TTypes<T>::Tensor::Index;
    // Reduce along the last axis (i.e axis 1) of the rank-2 Tensor.
    const int64 output_size = reshape[kBroadcasting] * reshape[kBatch] *
                              reshape[kFree] * reshape[kContract];
    functor::ReduceFunctor<Device, Reducer>::Reduce(
        ctx, output->shaped<T, 1>({output_size}),
        const_cast<const Tensor&>(input_deduped)
            .shaped<T, 2>({output_size, reshape[kReduce]}),
        Eigen::array<Index, 1>({1}), Reducer());
    return Status::OK();
  }

  // Reshapes a Tensor of shape [b0,b1...bk,N,M] to [prod(b0,b1...bk),N,M].
  static Status ReshapeToRank3(const Tensor& input, int batch_size,
                               Tensor* output) {
    const int rank = input.dims();
    TensorShape output_shape = {batch_size, input.dim_size(rank - 2),
                                input.dim_size(rank - 1)};
    return CopyFrom(input, output_shape, output);
  }

  // Contracts the inputs along the last axis (or the second last if the
  // corresponding value of swap_free_and_contract is true). The batch
  // dimensions are broadcast to the output shape.
  // TODO(anudhyan): BatchMatMul might devolve into a component-wise
  // multiplication when the matrix shape is [1,1]; in this case BatchMatMul
  // functor would be very inefficient. The functor should detect if this is the
  // case and perform componentwise multiplication functor instead.
  template <typename Device, typename T>
  static Status ContractOperands(OpKernelContext* ctx,
                                 absl::Span<const Tensor> inputs,
                                 absl::Span<const bool> swap_free_and_contract,
                                 Tensor* output) {
    if (inputs.size() == 1)
      return CopyFrom(inputs[0], inputs[0].shape(), output);
    MatMulBCast bcast(inputs[0].shape().dim_sizes(),
                      inputs[1].shape().dim_sizes());
    if (!bcast.IsValid()) {
      return errors::InvalidArgument(
          "Invalid broadcasting dimensions: ", inputs[0].shape().DebugString(),
          " vs. ", inputs[1].shape().DebugString());
    }
    Tensor lhs;
    TF_RETURN_IF_ERROR(ReshapeToRank3(inputs[0], bcast.x_batch_size(), &lhs));
    Tensor rhs;
    TF_RETURN_IF_ERROR(ReshapeToRank3(inputs[1], bcast.y_batch_size(), &rhs));
    TensorShape output_shape = bcast.output_batch_shape();
    for (int i = 0; i < inputs.size(); ++i) {
      const int64 free_axis =
          inputs[i].dims() - (swap_free_and_contract[i] ? 1 : 2);
      output_shape.AddDim(inputs[i].dim_size(free_axis));
    }
    bool trans_x = swap_free_and_contract[0];
    bool trans_y = !swap_free_and_contract[1];
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      functor::SetZeroFunctor<Device, T> set_zero;
      set_zero(ctx->eigen_device<Device>(), output->flat<T>());
      return Status::OK();
    }
    Tensor output_reshaped;
    TF_RETURN_IF_ERROR(
        ReshapeToRank3(*output, bcast.output_batch_size(), &output_reshaped));
    LaunchBatchMatMul<Device, T>::Launch(ctx, lhs, rhs, /*adj_x=*/false,
                                         /*adj_y=*/false, trans_x, trans_y,
                                         bcast, &output_reshaped);
    return Status::OK();
  }
};

template <typename Device, typename T>
class EinsumOp : public OpKernel {
 public:
  explicit EinsumOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("equation", &equation_));
    OP_REQUIRES_OK(
        c, EinsumHelper::ParseEquation(
               equation_, &input_labels_, &output_labels_, &label_types_,
               &input_label_counts_, &output_label_counts_,
               &input_has_ellipsis_, &output_has_ellipsis_));
  }

  void Compute(OpKernelContext* ctx) override {
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));

    OperandLabels input_labels(input_labels_);
    Labels output_labels(output_labels_);
    std::vector<EinsumHelper::DimensionType> label_types(label_types_);
    OperandLabelCounts input_label_counts(input_label_counts_);
    LabelCounts output_label_counts(output_label_counts_);
    LabelToDimSizes label_to_dim_sizes;

    OP_REQUIRES_OK(ctx, EinsumHelper::ProcessDimensions(
                            inputs, input_has_ellipsis_, output_has_ellipsis_,
                            &input_labels, &output_labels, &label_types,
                            &input_label_counts, &output_label_counts,
                            &label_to_dim_sizes));

    // The reduction phase (a) sums across reduction dimensions, (b) takes
    // generalized diagonals, and (c) reshapes it into shape
    //   [(broadcasting) batch shape] + [F,C]
    // where F and C denote the total (compacted) size of free and contract
    // dimensions, respectively.
    const int num_inputs = inputs.size();
    OperandLabels free_labels(num_inputs);
    gtl::InlinedVector<Tensor, 2> inputs_reduced(num_inputs);
    gtl::InlinedVector<bool, 2> swap_free_and_contract(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      OP_REQUIRES_OK(ctx,
                     EinsumHelper::ReduceOperand<Device, T>(
                         ctx, inputs[i], label_types, input_label_counts[i],
                         &input_labels[i], &free_labels[i],
                         &swap_free_and_contract[i], &inputs_reduced[i]));
    }

    // After reduction, the inputs should be reshaped to Tensors suitable for
    // contraction. If num_inputs is 1, the reduced input is simply forwarded to
    // the output.
    Tensor contraction_output_reshaped;
    OP_REQUIRES_OK(ctx, EinsumHelper::ContractOperands<Device, T>(
                            ctx, inputs_reduced, swap_free_and_contract,
                            &contraction_output_reshaped));

    // Copy the batch labels from the contraction output. Recover the batch
    // shape, which may have been broadcasted.
    TensorShape result_shape = contraction_output_reshaped.shape();
    result_shape.RemoveLastDims(2);

    int num_labels = label_types.size();
    Labels result_labels;
    // All batch dimensions should be present in the contracted result. First
    // the broadcasting dimensions, then the named batch dimensions.
    for (int label = 0; label < num_labels; ++label) {
      if (label_types[label] == EinsumHelper::kBroadcasting)
        result_labels.push_back(label);
    }
    for (int label = 0; label < num_labels; ++label) {
      if (label_types[label] == EinsumHelper::kBatch)
        result_labels.push_back(label);
    }
    for (int i = 0; i < num_inputs; ++i) {
      for (int label : free_labels[i]) {
        result_labels.push_back(label);
        result_shape.AddDim(label_to_dim_sizes[label]);
      }
    }

    // Reshape the contraction (or reduction) result to its expanded shape:
    // [(broadcasted) batch shape] + [free shape 0] + [free shape 1].
    Tensor contraction_output;
    OP_REQUIRES_OK(
        ctx, EinsumHelper::CopyFrom(contraction_output_reshaped, result_shape,
                                    &contraction_output));

    // Inflate the output if necessary. (E.g. for the equation 'i->iii' which
    // may arise while computing gradient of a regular Einsum).
    // TODO(anudhyan): It's possible that Eigen's contract and inflate can be
    // chained here to avoid materializing an intermediate.
    Tensor output_inflated;
    OP_REQUIRES_OK(
        ctx, EinsumHelper::StrideOrInflate<Device, T>(
                 ctx, contraction_output, result_labels, output_label_counts,
                 true /* should_inflate */, &output_inflated));
    if (output_inflated.dims() > contraction_output.dims()) {
      // We inflated the output. Modify result labels accordingly.
      Labels inflated_labels;
      for (int label : result_labels) {
        inflated_labels.insert(inflated_labels.end(),
                               output_label_counts[label], label);
      }
      result_labels.swap(inflated_labels);
    }
    // Find the permutation to map the result labels to the output labels. Note
    // that both the result and the final output may have the repeated labels,
    // in which case the permutation preserves the left-to-right ordering.
    // E.g. if result labels are [0, 0, 1] and output is [0, l, 0] then the
    // permutation should be [0, 2, 1]. We also use the fact that repeated
    // labels in the result are adjacent to each other.
    std::vector<int> output_permutation(output_labels.size());
    std::vector<int> label_to_position(num_labels, -1);
    for (int i = 0; i < result_labels.size(); ++i) {
      // Remember the position of only the leftmost result label.
      if (label_to_position[result_labels[i]] == -1) {
        label_to_position[result_labels[i]] = i;
      }
    }
    for (int i = 0; i < output_labels.size(); ++i) {
      output_permutation[i] = label_to_position[output_labels[i]];
      // We have found the leftmost occurrence. The next one would be adjacent.
      label_to_position[output_labels[i]] += 1;
    }
    Tensor output;
    OP_REQUIRES_OK(ctx, EinsumHelper::TransposeOperand<Device, T>(
                            ctx, output_inflated, output_permutation, &output));
    ctx->set_output(0, output);
  }

  string TraceString(const OpKernelContext& ctx, bool verbose) const override {
    string op = profiler::TraceMeOp(name_view(), type_string_view());
    string equation = strings::StrCat("(", equation_, ")");
    if (verbose) {
      string shape = ShapeTraceString(ctx);
      if (!shape.empty()) {
        return profiler::TraceMeEncode(
            std::move(op), {{"equation", equation}, {"shape", shape}});
      }
    }
    return profiler::TraceMeEncode(std::move(op), {{"equation", equation}});
  }

 private:
  string equation_;
  OperandLabels input_labels_;
  Labels output_labels_;
  std::vector<EinsumHelper::DimensionType> label_types_;
  OperandLabelCounts input_label_counts_;
  LabelCounts output_label_counts_;
  gtl::InlinedVector<bool, 2> input_has_ellipsis_;
  bool output_has_ellipsis_ = false;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, N)                                      \
  template <>                                                       \
  void StrideFunctor<GPUDevice, T, N>::operator()(                  \
      const GPUDevice& d, typename TTypes<T, N>::ConstTensor input, \
      const Eigen::DSizes<Eigen::DenseIndex, N>& strides,           \
      typename TTypes<T, N>::Tensor output);                        \
  extern template struct StrideFunctor<GPUDevice, T, N>;            \
  template <>                                                       \
  void InflateFunctor<GPUDevice, T, N>::operator()(                 \
      const GPUDevice& d, typename TTypes<T, N>::ConstTensor input, \
      const Eigen::DSizes<Eigen::DenseIndex, N>& strides,           \
      typename TTypes<T, N>::Tensor output);                        \
  extern template struct InflateFunctor<GPUDevice, T, N>;

#define DECLARE_GPU_SPECS(T) \
  DECLARE_GPU_SPEC(T, 1);    \
  DECLARE_GPU_SPEC(T, 2);    \
  DECLARE_GPU_SPEC(T, 3);    \
  DECLARE_GPU_SPEC(T, 4);    \
  DECLARE_GPU_SPEC(T, 5);    \
  DECLARE_GPU_SPEC(T, 6);

DECLARE_GPU_SPECS(Eigen::half);
DECLARE_GPU_SPECS(double);
DECLARE_GPU_SPECS(float);
// TODO(rocm): Enable once complex types are supported.
#if GOOGLE_CUDA
DECLARE_GPU_SPECS(complex64);
DECLARE_GPU_SPECS(complex128);
#endif
#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPECS
}  // namespace functor
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_IMPL_H_
