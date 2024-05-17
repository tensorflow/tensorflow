/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LLVM_IR_LOOP_EMITTER_H_
#define XLA_SERVICE_LLVM_IR_LOOP_EMITTER_H_

#include <functional>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/statusor.h"

namespace xla {
namespace llvm_ir {

// A function type for emitting code that generates an element in the target
// array. The function gets a multi-dimensional index as its only input. This
// index specifies the target element for which a value needs to be computed.
// The function has to emit code to compute this value and return the resulting
// llvm::Value*.
using ElementGenerator =
    std::function<absl::StatusOr<llvm::Value*>(const IrArray::Index& index)>;
using BodyEmitter = std::function<absl::Status(const IrArray::Index& index)>;

// Creates the body emitter from target arrays.
BodyEmitter MakeBodyEmitter(const ElementGenerator& target_element_generator,
                            absl::Span<IrArray const> target_arrays,
                            llvm::IRBuilder<>* b, bool is_tuple);

// Emits a loop for every element in the given shape.
class LoopEmitter {
 public:
  LoopEmitter(const BodyEmitter& body_emitter, const Shape& shape,
              llvm::IRBuilder<>* b);

  // Constructs a LoopEmitter from an body_emitter that generates
  // element of the given target array in the dynamic dimension.
  LoopEmitter(const BodyEmitter& body_emitter, const Shape& shape,
              std::vector<llvm::Value*> dynamic_dims, llvm::IRBuilder<>* b);

  // Constructs a LoopEmitter from an element generator that generates each
  // element of the given target array.
  LoopEmitter(const ElementGenerator& target_element_generator,
              const IrArray& target_array, llvm::IRBuilder<>* b);

  // Constructs a LoopEmitter that emits one element into each of N separate
  // arrays on each iteration of the loop.
  //
  // This is used for multi-output fusion.  target_element_generator must
  // produce an LLVM struct with N elements.
  LoopEmitter(const ElementGenerator& target_element_generator,
              absl::Span<const IrArray> target_arrays, llvm::IRBuilder<>* b);

  LoopEmitter(const LoopEmitter&) = delete;
  LoopEmitter& operator=(const LoopEmitter&) = delete;
  virtual ~LoopEmitter() = default;

  // Emits a loop nest (with a yet-to-be-filled loop body) that iterates through
  // every element in the given shape. Returns the multi-dimensional index that
  // specifies the element, will return multiple indices if the loop is
  // unrolled.
  virtual std::vector<IrArray::Index> EmitIndexAndSetExitBasicBlock(
      absl::string_view loop_name, llvm::Type* index_type,
      llvm::Value* base_index);

  // Emits a complete loop nest for every element in the given shape.
  absl::Status EmitLoop(absl::string_view loop_name = "",
                        llvm::Type* index_type = nullptr);

 protected:
  // An IR emitter that generates the loop body.
  BodyEmitter body_emitter_;

  // The shape that the emitted loop iterates through.
  Shape shape_;

  // Dynamic dimensions that  emitted loop iterates through. Generate the
  // loop based on the dynamic dimensions if this vector is not empty.
  std::vector<llvm::Value*> dynamic_dims_;

  // Points to the exit block of the emitted loop. If the given shape is
  // scalar, no loops are emitted and exit_bb_ is nullptr in that case.
  llvm::BasicBlock* exit_bb_;

  llvm::IRBuilder<>* b_;

 private:
  IrArray::Index EmitStaticIndex(ForLoopNest* loop_nest,
                                 llvm::Type* index_type);
  IrArray::Index EmitDynamicIndex(ForLoopNest* loop_nest,
                                  llvm::Type* index_type);
};

}  // namespace llvm_ir
}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_LOOP_EMITTER_H_
