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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_UTIL_STRIDED_SLICE_OP_H_
#define THIRD_PARTY_TENSORFLOW_CORE_UTIL_STRIDED_SLICE_OP_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

// This class and its subclasses allow ValidateStridedSliceOp to be called with
// different implementations of partial tensors.
class ShapeReadWriteInterface {
 public:
  virtual ~ShapeReadWriteInterface() {}
  virtual int dims() const = 0;
  // Returns -1 for unknown size.
  virtual int64 dim_size(int idx) const = 0;
  // Passes -1 for unknown dim size.
  virtual void add_dim(int64 size) = 0;
};

// Implementation of ShapeReadWriteInterface that modifies the given TensorShape
// <shape> in-place. Does not support adding unknown dims in add_dim.
class ShapeReadWriteFromTensorShape : public ShapeReadWriteInterface {
 public:
  ShapeReadWriteFromTensorShape(TensorShape* shape)
      : const_shape_(shape), shape_(shape) {}
  ShapeReadWriteFromTensorShape(const TensorShape* shape)
      : const_shape_(shape) {}
  ~ShapeReadWriteFromTensorShape() override {}
  int dims() const override;
  int64 dim_size(int idx) const override;
  void add_dim(int64 size) override;

 private:
  const TensorShape* const const_shape_;
  // same as const_shape_, or nullptr if the non-const ctr is used.
  TensorShape* const shape_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(ShapeReadWriteFromTensorShape);
};

// Implementation of ShapeReadWriteInterface that modifies the given
// TensorShapeProto in place.
class ShapeReadWriteFromTensorShapeProto : public ShapeReadWriteInterface {
 public:
  ShapeReadWriteFromTensorShapeProto(TensorShapeProto* shape)
      : const_shape_(shape), shape_(shape) {}
  ShapeReadWriteFromTensorShapeProto(const TensorShapeProto* shape)
      : const_shape_(shape) {}
  ~ShapeReadWriteFromTensorShapeProto() override {}
  int dims() const override;
  int64 dim_size(int idx) const override;
  void add_dim(int64 size) override;

 private:
  const TensorShapeProto* const const_shape_;
  // same as shape_, or nullptr if the non-const ctr is used.
  TensorShapeProto* const shape_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(ShapeReadWriteFromTensorShapeProto);
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
Status ValidateStridedSliceOp(
    const Tensor* begin_tensor, const Tensor* end_tensor,
    const Tensor& strides_tensor, const ShapeReadWriteInterface& input_shape,
    int32 begin_mask_spec, int32 end_mask_spec, const int32 ellipsis_mask,
    int32 new_axis_mask, int32 shrink_axis_mask,
    ShapeReadWriteInterface* processing_shape,
    ShapeReadWriteInterface* final_shape, bool* is_identity,
    bool* is_simple_slice, bool* slice_dim0,
    gtl::InlinedVector<int64, 4>* begin, gtl::InlinedVector<int64, 4>* end,
    gtl::InlinedVector<int64, 4>* strides);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_UTIL_STRIDED_SLICE_OP_H_
