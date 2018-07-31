/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_KERNEL_TILING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_KERNEL_TILING_H_

#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"

namespace xla {
namespace llvm_ir {

// About 0-2-1 transpose:
//
// If a shape can be viewed as three logical components 0-1-2 in the order of
// major to minor, a 0-2-1-transpose changes the order of such logical
// components to 0-2-1. We call the shape being transposed the input shape and
// the transposed shape the output shape. The logical view of the input and
// output shapes for the transpose are called the 0-1-2 shape or reduced input
// shape and the 0-2-1 shape or the reduced output shape respectively. The
// original input and output shapes are called the unreduced input and output
// shapes.

// If `b` is a 0-2-1 transpose of `a` in 0-1-2, return the dimensions for the
// reduced shape of `b` or the 0-2-1 shape.
tensorflow::gtl::optional<std::vector<int64> > FindTranspose021(const Shape& a,
                                                                const Shape& b);

// Return the unreduced output index corresponding to the given reduced output
// index.
IrArray::Index GetUnreducedOutputIndex(
    const IrArray::Index& reduced_output_index,
    const Shape& reduced_output_shape, const Shape& unreduced_output_shape,
    llvm::IRBuilder<>* b);

// A class to represent information for tiled parameters to support IR emission
// for 021 transpose.
class TiledParameterInfo {
 public:
  TiledParameterInfo(tensorflow::gtl::ArraySlice<llvm::Value*> param_buffers,
                     llvm::Value* y, llvm::Value* x)
      : param_buffers_(param_buffers), y_(y), x_(x) {}

  llvm::Value* x() const { return x_; }
  llvm::Value* y() const { return y_; }

  void set_x(llvm::Value* x) { x_ = x; }
  void set_y(llvm::Value* y) { y_ = y; }

  llvm::Value* GetBufferForParameter(int64 index) const {
    return param_buffers_[index];
  }

 private:
  // Param_buffers_[i] stores the tile buffer for the ith parameter or nullptr
  // if the parameter is not tiled.
  tensorflow::gtl::ArraySlice<llvm::Value*> param_buffers_;
  // The y coordinate within a tile.
  llvm::Value* y_;
  // The x coordinate within a tile.
  llvm::Value* x_;
};

}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_KERNEL_TILING_H_
