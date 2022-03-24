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

#include "tensorflow/core/util/strided_slice_op.h"

#include <array>
#include <iterator>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {

/// Constants
constexpr int32_t kShrinkAxis = -1, kNewAxis = -2;

// Sparse slicing specification
// if one does foo[3:5, ..., -3], this will have 3 length tensors
struct StridedSliceSparseSpec {
  int64_t dims;
  int32 num_add_axis_after_ellipsis;
  const Tensor* begin_tensor;
  const Tensor* end_tensor;
  const Tensor& strides_tensor;
  const int32 begin_mask, end_mask;
  int32 ellipsis_mask;
  const int32 new_axis_mask, shrink_axis_mask;
};

// Dense slicing specification
// all ellipses and newaxis' are expanded out. So if
// foo[3:5, ..., -3] where foo is 10 dimensional,
// each inlinedVector will have 10 entries whereas the
// sparse had 3 length tensors.
struct StridedSliceDenseSpec {
  const int64_t dims;
  int32 begin_mask;
  int32 end_mask;
  bool begin_valid;
  bool end_valid;
  gtl::InlinedVector<int64_t, 4>& begin;
  gtl::InlinedVector<int64_t, 4>& end;
  gtl::InlinedVector<int64_t, 4>& strides;
  // This vector helps construct the final shape of the slice.
  // The final tensor is reduced in rank whenever a single index e.g. foo[3]
  // is called for. The final tensor increases in rank with tf.newaxis
  // entries. If an index in this array is positive, the size of the dimension
  // is obtained from canonical end-begin. Otherwise, if it is a kNewAxis,
  // it will be 1. A shrunk dimension is skipped.
  gtl::InlinedVector<int32, 4> final_shape_gather_indices;
  // This vector has the same size as final_shape_gather_indices, but it
  // remembers the sparse index that a dimension comes from, instead of dense
  // index. A -1 in this vector means there the index is not from the sparse
  // input.
  gtl::InlinedVector<int32, 4> final_shape_gather_indices_sparse;
  gtl::InlinedVector<int32, 4> input_shape_gather_indices_sparse;
  // The dense indexed shrink mask is which processing dimensions
  // should be shrunk. For example, if foo.shape = (10,10,10,10)
  // foo[3, ..., 5] has sparse_shrink_axis_mask of 0x5 and
  // dense_shrink_axis_mask of 0x9, yielding a final shape (10,10).
  int32 shrink_axis_mask;
};

}  // namespace

template <class T>
static Status TF_MUST_USE_RESULT BuildDenseSpec(
    const StridedSliceSparseSpec& sparse, StridedSliceDenseSpec* dense) {
  // Build expanded begin, end, strides, begin_mask, end_mask
  // to remove any ellipsis
  dense->begin.resize(dense->dims);
  dense->end.resize(dense->dims);
  dense->strides.resize(dense->dims);
  dense->input_shape_gather_indices_sparse.resize(dense->dims);
  // What indices to get the final shape from.
  dense->begin_mask = 0;
  dense->end_mask = 0;
  dense->shrink_axis_mask = 0;
  {
    int full_index = 0;

    const T* const strides_flat = sparse.strides_tensor.vec<T>().data();
    dense->begin_valid = sparse.begin_tensor != nullptr;
    dense->end_valid = sparse.end_tensor != nullptr;

    const T* const begin_flat = sparse.begin_tensor != nullptr
                                    ? sparse.begin_tensor->vec<T>().data()
                                    : nullptr;
    const T* const end_flat = sparse.end_tensor != nullptr
                                  ? sparse.end_tensor->vec<T>().data()
                                  : nullptr;

    for (int i = 0; i < sparse.dims; i++) {
      if ((1 << i) & sparse.ellipsis_mask) {
        // Expand the ellipsis into the appropriate indices
        // NOTE: this only works because we guaranteed one ellipsis
        int32_t next_index = std::min(dense->dims - (sparse.dims - i) + 1 +
                                          sparse.num_add_axis_after_ellipsis,
                                      dense->dims);
        for (; full_index < next_index; full_index++) {
          // new_axis' aren't real axis so you have to skip
          dense->begin[full_index] = dense->end[full_index] = 0;
          dense->strides[full_index] = 1;
          dense->begin_mask |= (1 << full_index);
          dense->end_mask |= (1 << full_index);
          dense->final_shape_gather_indices.push_back(full_index);
          dense->final_shape_gather_indices_sparse.push_back(-1);
          dense->input_shape_gather_indices_sparse[full_index] = i;
        }
      } else if ((1 << i) & sparse.new_axis_mask) {
        dense->final_shape_gather_indices.push_back(kNewAxis);
        dense->final_shape_gather_indices_sparse.push_back(-1);
      } else {
        if (full_index == dense->begin.size()) {
          return errors::InvalidArgument("Index out of range using input dim ",
                                         full_index, "; input has only ",
                                         dense->dims, " dims");
        }

        // Gather slicing spec into appropriate index
        if (begin_flat != nullptr) {
          dense->begin[full_index] = internal::SubtleMustCopy<T>(begin_flat[i]);
        }
        if (end_flat != nullptr) {
          dense->end[full_index] = internal::SubtleMustCopy<T>(end_flat[i]);
        }
        dense->strides[full_index] =
            internal::SubtleMustCopy<T>(strides_flat[i]);
        if (sparse.begin_mask & (1 << i)) {
          dense->begin_mask |= (1 << full_index);
        }
        if (sparse.end_mask & (1 << i)) {
          dense->end_mask |= (1 << full_index);
        }
        // If shrink, record where to get the dimensionality from (i.e.
        // new_axis creates a fake 1 size dimension. Also remember shrink
        // axis (now in dense form) so we can ignore dense->end below.
        if (sparse.shrink_axis_mask & (1 << i)) {
          dense->final_shape_gather_indices.push_back(kShrinkAxis);
          dense->final_shape_gather_indices_sparse.push_back(-1);
          dense->shrink_axis_mask |= (1 << full_index);
        } else {
          dense->final_shape_gather_indices.push_back(full_index);
          // Remember that where in the sparse shape the dense dim comes
          // from.
          dense->final_shape_gather_indices_sparse.push_back(i);
        }
        dense->input_shape_gather_indices_sparse[full_index] = i;
        full_index++;
      }
    }
  }
  return Status::OK();
}

Status ValidateStridedSliceOp(
    const Tensor* begin_tensor, const Tensor* end_tensor,
    const Tensor& strides_tensor, const PartialTensorShape& input_shape,
    int32_t begin_mask_spec, int32_t end_mask_spec, const int32_t ellipsis_mask,
    int32_t new_axis_mask, int32_t shrink_axis_mask,
    PartialTensorShape* processing_shape, PartialTensorShape* final_shape,
    bool* is_identity, bool* is_simple_slice, bool* slice_dim0,
    gtl::InlinedVector<int64_t, 4>* begin, gtl::InlinedVector<int64_t, 4>* end,
    gtl::InlinedVector<int64_t, 4>* strides,
    StridedSliceShapeSpec* shape_spec) {
  const bool begin_is_wrong =
      begin_tensor != nullptr &&
      !(TensorShapeUtils::IsVector(begin_tensor->shape()) &&
        begin_tensor->NumElements() == strides_tensor.NumElements() &&
        begin_tensor->NumElements() < 32 /* using 32 bit masks */);
  const bool end_is_wrong =
      end_tensor != nullptr &&
      !(TensorShapeUtils::IsVector(end_tensor->shape()) &&
        end_tensor->NumElements() == strides_tensor.NumElements());
  if (begin_is_wrong || end_is_wrong ||
      !TensorShapeUtils::IsVector(strides_tensor.shape())) {
    if (begin_tensor != nullptr && end_tensor != nullptr) {
      return errors::InvalidArgument(
          "Expected begin, end, and strides to be 1D equal size tensors, ",
          "but got shapes ", begin_tensor->shape().DebugString(), ", ",
          end_tensor->shape().DebugString(), ", and ",
          strides_tensor.shape().DebugString(), " instead.");
    } else {
      return errors::InvalidArgument(
          "Expected begin, end, and strides to be 1D equal size tensors, ",
          "but got shape ", strides_tensor.shape().DebugString(),
          " for strides.");
    }
  }
  // Use bit compares to ensure ellipsis_mask is 0 or a power of 2
  // i.e. there exists only no more than one ellipsis
  if (ellipsis_mask && ((ellipsis_mask & (ellipsis_mask - 1)) != 0)) {
    return errors::InvalidArgument(
        "Multiple ellipses in slice spec not allowed");
  }

  // Step 1: Account for ellipsis and new axis
  //
  // Check for ellipses and count how many non-newaxis' there are after
  // TODO(aselle): Convert this to do a fast log2 followed by iteration
  //               counting ones in next guys
  bool ellipsis_seen = false;

  StridedSliceSparseSpec sparse_spec = {strides_tensor.NumElements(),
                                        0,
                                        begin_tensor,
                                        end_tensor,
                                        strides_tensor,
                                        begin_mask_spec,
                                        end_mask_spec,
                                        ellipsis_mask,
                                        new_axis_mask,
                                        shrink_axis_mask};

  for (int32_t i = 0; i < sparse_spec.dims; i++) {
    if (ellipsis_seen && ((1 << i) & new_axis_mask) != 0) {
      sparse_spec.num_add_axis_after_ellipsis++;
    }
    if ((1 << i) & ellipsis_mask) {
      ellipsis_seen = true;
    }
  }
  // If no ellipsis insert one at the end
  if (!ellipsis_seen) {
    sparse_spec.ellipsis_mask |= (1 << sparse_spec.dims);
    sparse_spec.dims++;  // this effects loop iteration below
  }

  // Step 2: Make a sparse spec into a full index spec
  //
  // The sparse spec does not correspond to the number of dimensions
  // Make a dense spec that corresponds to the number of dimensions
  //
  // For example suppose foo[...,3:] on foo.shape=(2,2,3) then
  // we need to produce the missing begin_mask for the first two
  // dimensions i.e. from begin_mask_spec=0, end_mask_spec=2
  // we achieve begin_mask=6, end_mask=7
  StridedSliceDenseSpec dense_spec = {input_shape.dims(),
                                      0 /* begin_mask */,
                                      0 /* end_mask */,
                                      false /* begin_valid */,
                                      false /* end_valid */,
                                      *begin,
                                      *end,
                                      *strides};

  if (strides_tensor.dtype() == DT_INT32) {
    TF_RETURN_IF_ERROR(BuildDenseSpec<int32>(sparse_spec, &dense_spec));
  } else if (strides_tensor.dtype() == DT_INT64) {
    TF_RETURN_IF_ERROR(BuildDenseSpec<int64_t>(sparse_spec, &dense_spec));
  } else if (strides_tensor.dtype() == DT_INT16) {
    TF_RETURN_IF_ERROR(BuildDenseSpec<int16_t>(sparse_spec, &dense_spec));
  } else {
    LOG(FATAL) << "begin must be either int16, int32 or int64";
  }

  // Step 3: Make implicit ranges (non-zero begin_masks and end_masks) explicit
  //         and bounds check!
  *is_identity = true;
  *slice_dim0 = true;
  *is_simple_slice = true;
  processing_shape->Clear();
  for (int i = 0; i < input_shape.dims(); ++i) {
    int64_t& begin_i = (*begin)[i];
    int64_t& end_i = (*end)[i];
    int64_t& stride_i = (*strides)[i];
    int64_t dim_i = input_shape.dim_size(i);
    if (stride_i == 0) {
      return errors::InvalidArgument("strides[", i, "] must be non-zero");
    }
    bool shrink_i = (dense_spec.shrink_axis_mask & (1 << i));
    if (dim_i == -1) {
      processing_shape->AddDim(shrink_i ? 1 : -1);
      continue;
    }

    const std::array<int64_t, 2> masks = {
        {dense_spec.begin_mask & (1 << i), dense_spec.end_mask & (1 << i)}};
    const std::array<int64_t, 2> valid_range = {
        {stride_i > 0 ? 0 : -1, stride_i > 0 ? dim_i : dim_i - 1}};

    auto canonical = [stride_i, dim_i, masks, valid_range](int64_t x, int c) {
      if (masks[c]) {
        return stride_i > 0 ? valid_range[c] : valid_range[(c + 1) & 1];
      } else {
        int64_t x_fwd =
            x < 0 ? dim_i + x : x;  // make negative indices positive
        return x_fwd < valid_range[0]
                   ? valid_range[0]
                   : x_fwd > valid_range[1] ? valid_range[1] : x_fwd;
      }
    };
    if (shrink_i && stride_i <= 0) {
      return errors::InvalidArgument(
          "only stride 1 allowed on non-range indexing.");
    }
    (*is_simple_slice) &= stride_i == 1;

    const bool begin_and_end_masked =
        (dense_spec.begin_mask & (1 << i)) && (dense_spec.end_mask & (1 << i));
    if (dense_spec.begin_valid && dense_spec.end_valid) {
      if (shrink_i) {
        // If we are shrinking, the end index is now possibly incorrect. In
        // particular foo[-1] produces sparse_begin = -1, sparse_end = 0.
        // and canonical puts these to n-1 and 0, which implies a degenerate
        // interval. Fortunately, it is now safe to re-create end as begin+1.
        int64_t x_fwd = begin_i < 0 ? dim_i + begin_i : begin_i;
        begin_i = x_fwd;
        end_i = begin_i + 1;
        if (x_fwd < 0 || x_fwd >= dim_i) {
          return errors::InvalidArgument(
              "slice index ", begin_i, " of dimension ", i, " out of bounds.");
        }
      } else {
        begin_i = canonical(begin_i, 0);
        end_i = canonical(end_i, 1);
      }
      // Update optimization values
      bool take_all_in_dimension =
          stride_i == 1 && begin_i == 0 && end_i == dim_i;
      (*is_identity) &= take_all_in_dimension;
      (*slice_dim0) &= (i == 0 && stride_i == 1) || take_all_in_dimension;
    } else {
      (*is_identity) &= stride_i == 1 && begin_and_end_masked;
      (*slice_dim0) &= (i == 0 && stride_i == 1) || begin_and_end_masked;
    }
    // Compute the processing shape (the intermediate Eigen will produce)
    int64_t interval_length;
    bool known_interval = false;
    if (dense_spec.begin_valid && dense_spec.end_valid) {
      interval_length = end_i - begin_i;
      known_interval = true;
    } else if (shrink_i) {
      // The dimension is still known as 1 for the processing_shape, but will be
      // discarded for the final shape.
      interval_length = 1;
      known_interval = true;
    } else if (begin_and_end_masked) {
      // Even if we don't have values for begin or end, we do know that this
      // dimension covers the whole interval. If we have shape information for
      // this dimension, that tells us the interval length.
      if (dim_i >= 0) {
        if (stride_i < 0) {
          interval_length = -dim_i;
        } else {
          interval_length = dim_i;
        }
        known_interval = true;
      }
    }
    if (known_interval) {
      int64_t size_i;
      // Hold zero if the interval is degenerate, otherwise account for
      // remainder
      if (interval_length == 0 || ((interval_length < 0) != (stride_i < 0))) {
        size_i = 0;
      } else {
        size_i = interval_length / stride_i +
                 (interval_length % stride_i != 0 ? 1 : 0);
      }
      processing_shape->AddDim(size_i);
    } else {
      processing_shape->AddDim(-1);
    }
  }

  // Step 4: Compute the final shape
  //
  // new_axis will increase dimension by 1 (with a one-size dimension)
  // slices like foo[3,...] will reduce dimension by 1.
  // This cannot be done earlier, because it depends on Step 3.
  final_shape->Clear();
  if (shape_spec != nullptr) {
    shape_spec->output_to_sparse_mapping.clear();
    shape_spec->output_to_processing_mapping.clear();
    shape_spec->processing_to_sparse_mapping.assign(
        dense_spec.input_shape_gather_indices_sparse.begin(),
        dense_spec.input_shape_gather_indices_sparse.end());

    shape_spec->begin_dense_mask = dense_spec.begin_mask;
    shape_spec->end_dense_mask = dense_spec.end_mask;
    shape_spec->shrink_axis_dense_mask = dense_spec.shrink_axis_mask;
  }

  for (int64_t dense_dim = 0;
       dense_dim < dense_spec.final_shape_gather_indices.size(); ++dense_dim) {
    int64_t gather_index = dense_spec.final_shape_gather_indices[dense_dim];
    int64_t sparse_index =
        dense_spec.final_shape_gather_indices_sparse[dense_dim];
    if (gather_index >= 0) {
      final_shape->AddDim(processing_shape->dim_size(gather_index));
      if (shape_spec != nullptr) {
        shape_spec->output_to_sparse_mapping.push_back(sparse_index);
        shape_spec->output_to_processing_mapping.push_back(gather_index);
      }
    } else if (gather_index == kNewAxis) {
      final_shape->AddDim(1);
      if (shape_spec != nullptr) {
        shape_spec->output_to_sparse_mapping.push_back(-1);
        shape_spec->output_to_processing_mapping.push_back(-1);
      }
    }
  }

  return Status::OK();
}

Status ValidateStridedSliceOp(
    const Tensor* begin_tensor, const Tensor* end_tensor,
    const Tensor& strides_tensor, const PartialTensorShape& input_shape,
    int32_t begin_mask_spec, int32_t end_mask_spec, const int32_t ellipsis_mask,
    int32_t new_axis_mask, int32_t shrink_axis_mask,
    TensorShape* processing_shape, TensorShape* final_shape, bool* is_identity,
    bool* is_simple_slice, bool* slice_dim0,
    gtl::InlinedVector<int64_t, 4>* begin, gtl::InlinedVector<int64_t, 4>* end,
    gtl::InlinedVector<int64_t, 4>* strides,
    StridedSliceShapeSpec* shape_spec) {
  // Validate with PartialTensorShape output
  PartialTensorShape partial_processing_shape, partial_final_shape;
  TF_RETURN_IF_ERROR(ValidateStridedSliceOp(
      begin_tensor, end_tensor, strides_tensor, input_shape, begin_mask_spec,
      end_mask_spec, ellipsis_mask, new_axis_mask, shrink_axis_mask,
      &partial_processing_shape, &partial_final_shape, is_identity,
      is_simple_slice, slice_dim0, begin, end, strides, shape_spec));

  // Verify that the output shapes are fully known
  if (!partial_processing_shape.AsTensorShape(processing_shape) ||
      !partial_final_shape.AsTensorShape(final_shape)) {
    return errors::Internal("ValidateStridedSliceOp returned partial shapes ",
                            partial_processing_shape.DebugString(), " and ",
                            partial_final_shape.DebugString());
  }
  return Status::OK();
}

}  // namespace tensorflow
