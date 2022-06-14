/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_UTIL_STRIDED_SLICE_OP_H_
#define TENSORFLOW_CORE_UTIL_STRIDED_SLICE_OP_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

struct StridedSliceShapeSpec {
  // Begin mask canonlized in dense form.
  int32_t begin_dense_mask;
  // End mask canonlized in dense form.
  int32_t end_dense_mask;
  // Shrink axis mask canonlized in dense form.
  int32_t shrink_axis_dense_mask;
  // output_to_sparse_mapping[i] represents output[i]'s the corresponding dim
  // index in the begin_tensor. If
  // output_to_sparse_mapping[i] is -1, it means the dimension doesn't show up
  // in sparse_mapping.
  gtl::InlinedVector<int64_t, 4> output_to_sparse_mapping;
  // output_to_processing_mapping is similar to output_to_sparse_mapping, but
  // for processing shape.
  gtl::InlinedVector<int64_t, 4> output_to_processing_mapping;
  // processing_to_sparse_mapping[i] represents input_shape[i]'s corresponding
  // dim index in the begin_tensor.
  gtl::InlinedVector<int64_t, 4> processing_to_sparse_mapping;
};

// Runs validation on the strided slice op parameters.
//
// Is a separate translation unit from the kernel so that:
// 1. The op's shape function can use it.
// 2. The code size is reduced vs templating this on the kernel's type.
//
// Note that when input_shape is not fully specified, only <final_shape> and
// <processing_shape> are valid; <is_identity>, <is_simple_slice> and other
// output parameters will not be accurate.
//
// If <begin_tensor> or <end_tensor> are nullptr, <begin> and <end> will not be
// valid. In this case, <slice_dim0> and <is_identity> will be true only if a
// determination can be made based on the information given. A best effort is
// made to set <processing_shape> and <final_shape> based on <input_shape>, but
// some dimensions of <processing_shape> and/or <final_shape> may be unknown
// (-1). Any validation that can be done without complete information is
// performed.
//
Status ValidateStridedSliceOp(
    const Tensor* begin_tensor, const Tensor* end_tensor,
    const Tensor& strides_tensor, const PartialTensorShape& input_shape,
    int32_t begin_mask_spec, int32_t end_mask_spec, const int32_t ellipsis_mask,
    int32_t new_axis_mask, int32_t shrink_axis_mask,
    PartialTensorShape* processing_shape, PartialTensorShape* final_shape,
    bool* is_identity, bool* is_simple_slice, bool* slice_dim0,
    gtl::InlinedVector<int64_t, 4>* begin, gtl::InlinedVector<int64_t, 4>* end,
    gtl::InlinedVector<int64_t, 4>* strides,
    StridedSliceShapeSpec* shape_spec = nullptr);

// Same as above, but the outputs are TensorShape, not PartialTensorShape
Status ValidateStridedSliceOp(
    const Tensor* begin_tensor, const Tensor* end_tensor,
    const Tensor& strides_tensor, const PartialTensorShape& input_shape,
    int32_t begin_mask_spec, int32_t end_mask_spec, const int32_t ellipsis_mask,
    int32_t new_axis_mask, int32_t shrink_axis_mask,
    TensorShape* processing_shape, TensorShape* final_shape, bool* is_identity,
    bool* is_simple_slice, bool* slice_dim0,
    gtl::InlinedVector<int64_t, 4>* begin, gtl::InlinedVector<int64_t, 4>* end,
    gtl::InlinedVector<int64_t, 4>* strides,
    StridedSliceShapeSpec* shape_spec = nullptr);

// Simple class for determining if it is possible to broadcast a tensor to a
// strided slice.  Modelled after tensorflow::BCast, but with a few key
// differences:
// - the input_shape must be broadcastable to output_shape
//   (i.e. the slice shape does not grow).
// - does not allow reducing or flattening dimensions, since we cannot apply
//   these simplications to the destination slice.
// - allows for remapping dimensions, required in order to associate the input
//   with correct dimensions in the full (unsliced) destination tensor.
class StridedSliceAssignBCast {
 public:
  using Vec = gtl::InlinedVector<int64_t, 4>;

  StridedSliceAssignBCast(const Vec& input_shape, const Vec& output_shape);

  // Remaps broadcast, resize, and output dimensions via the provided map.
  // Negative values in the map correspond to dimensions being removed.
  // Unmapped dimensions are set to 1.
  //
  // This is to support remapping slice -> processing dimensions.  To relate
  // the sliced output dimensions back to processing dimensions (i.e. those
  // relative to the the original unsliced input), we need to remove any axes
  // that were added via the `new_axis_mask`, and add back any axes that were
  // removed via the `shrink_axis_mask`.  For example, an expression like
  //
  // >>> t = tf.zeros([3, 3])
  // >>> t[2, tf.newaxis, 0:2, tf.newaxis] = tf.ones([1, 3, 1])
  //       ^                                          ^  ^  ^
  //       |__ shrink axis                 new axis __|  |  |__ new axis
  //                                                     |_____ dim 1 of t
  //
  // would have `new_axis_mask = 0b1010` and `shrink_axis_mask = 0b0001`. The
  // slice has shape [1, 3, 1], but the original input tensor `t` has shape
  // [3, 3]. To remap the slice dimensions back to the input dimensions, the
  // mapping would use `num_dims = 2`, `dimension_map = {-1, 1, -1}`. This
  // removes the two new axes added for the slice, maps the middle slice
  // dimension to input dimension 1, and leaves input dimension 0 to have a
  // default size of 1 to add back the shrink axis.
  //
  // Returns false if the remapping fails.
  bool RemapDimensions(int64_t num_dims, const Vec& dimension_map);

  bool IsValid() const { return valid_; }

  bool IsBroadcastingRequired() const { return broadcasting_required_; }

  const Vec& reshape() const { return reshape_; }

  const Vec& bcast() const { return bcast_; }

  const Vec& result_shape() const { return result_shape_; }

 private:
  bool valid_ = true;
  bool broadcasting_required_ = false;
  Vec reshape_;
  Vec bcast_;
  Vec result_shape_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_STRIDED_SLICE_OP_H_
