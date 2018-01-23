/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_LOOP_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_LOOP_H_

#include <memory>
#include <string>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace llvm_ir {

// A class for constructing a for-loop in LLVM IR.
class ForLoop {
 public:
  // Emit a for-loop at the current insert point of the given IRBuilder.
  //
  // start_index and end_index are the loop bounds (end_index is not inclusive).
  // `step` is the increment of the loop index after each iteration.
  //
  // The current insert basic block of the builder is the preheader to the loop
  // (see below for definition of basic block names). All instructions (if any)
  // at or after the insert point in the insert basic block are moved to a newly
  // created exit basic block. Instructions before the insert point remain in
  // the insert BB:
  //
  //                   +--------------+         +----------------+
  //                   |  insert BB   |         |   insert BB    |
  //                   |     ...      |         | (preheader BB) |
  //                   | %foo = ...   |         |      ...       |
  //    insert point ->| %bar = ...   |  ===>   | %foo = ...     |
  //                   |     ...      |         +----------------+
  //                   +--------------+                 |
  //                                                    V
  //                                              [[ LOOP BBs ]]
  //                                                    |
  //                                                    V
  //                                             +--------------+
  //                                             |   exit BB    |
  //                                             | %bar = ...   |
  //                                             |     ...      |
  //                                             +--------------+
  //
  // `prefix` is used to disambiguate variable and basic block names emitted in
  // LLVM IR. If non-empty, it is prepended to the name of the induction
  // variable value and each basic block created for the loop.
  //
  // If `prevent_unrolling` is true then emit metadata that directs LLVM to not
  // unroll the generated loop.
  static std::unique_ptr<ForLoop> EmitForLoop(
      tensorflow::StringPiece prefix, llvm::Value* start_index,
      llvm::Value* end_index, llvm::Value* step, llvm::IRBuilder<>* ir_builder,
      bool prevent_unrolling = false, bool prevent_vectorization = false);

  // The names of the blocks follow LLVM's conventions. Control flow amongst the
  // blocks for the example C code looks like:
  //
  //   for (int i = 0; i < n; ++i) {
  //     do_stuff(i);
  //   }
  //
  //      +--------------+
  //      | preheader BB |
  //      |     i = 0    |
  //      +--------------+
  //              |
  //              V
  //      +-------------+
  //      |  header BB  |<-+
  //      | if i < n:   |  |
  //      |   goto body |  |
  //      | else:       |  |
  //      |   goto exit |  |
  //      +-------------+  |
  //            | |        |
  //   +--------+ |        |
  //   |          V        |
  //   |  +-------------+  |
  //   |  |   body BB   |  |
  //   |  | dostuff(i)  |--+
  //   |  | ++i         |
  //   |  +-------------+
  //   |
  //   |  +-------------+
  //   +->|   exit BB   |
  //      +-------------+
  //
  // Caller-emitted code to execute within the loop should be placed within the
  // "body" basic block.
  //
  // Return pointers to various blocks in the loop.
  llvm::BasicBlock* GetPreheaderBasicBlock() const { return preheader_bb_; }
  llvm::BasicBlock* GetHeaderBasicBlock() const { return header_bb_; }
  llvm::BasicBlock* GetBodyBasicBlock() const { return body_bb_; }
  llvm::BasicBlock* GetExitBasicBlock() const { return exit_bb_; }

  // Return the Value representing the induction variable in the body basic
  // block of the loop.
  llvm::Value* GetIndVarValue() const { return indvar_; }

 private:
  // Allow ForLoopNest to call this private constructor.
  friend class ForLoopNest;

  ForLoop(tensorflow::StringPiece prefix, tensorflow::StringPiece suffix,
          llvm::Value* start_index, llvm::Value* end_index, llvm::Value* step,
          bool prevent_unrolling, bool prevent_vectorization);

  // Emit the loop at the insert point of the builder.
  void Emit(llvm::IRBuilder<>* ir_builder);

  llvm::BasicBlock* CreateLoopBB(tensorflow::StringPiece name,
                                 llvm::IRBuilder<>* ir_builder);

  // Creates a name for an LLVM construct, appending prefix_ and suffix_, if
  // they are set.
  string GetQualifiedName(tensorflow::StringPiece name);

  // Return a list of metadata nodes that should be associated with the
  // llvm::Loop for this `ForLoop`.
  std::vector<llvm::Metadata*> GetLoopMetadata(llvm::IRBuilder<>* ir_builder);

  string prefix_;
  string suffix_;
  llvm::Value* start_index_;
  llvm::Value* end_index_;
  llvm::Value* step_;

  // To improve readability of the IR, we want the basic blocks to appear
  // consecutively in the following order: preheader, header, body, loop,
  // exit. The member insert_before_bb_ points to where the next basic block
  // should be created to ensure this ordering.
  llvm::BasicBlock* insert_before_bb_;

  llvm::BasicBlock* preheader_bb_;
  llvm::BasicBlock* header_bb_;
  llvm::BasicBlock* body_bb_;
  llvm::BasicBlock* exit_bb_;
  llvm::Value* indvar_;
  bool prevent_unrolling_;
  bool prevent_vectorization_;

  TF_DISALLOW_COPY_AND_ASSIGN(ForLoop);
};

// A simple class for constructing nested for-loops.
class ForLoopNest {
 public:
  explicit ForLoopNest(llvm::IRBuilder<>* ir_builder)
      : ForLoopNest(/*name=*/"", ir_builder) {}

  ForLoopNest(tensorflow::StringPiece name, llvm::IRBuilder<>* ir_builder)
      : name_(name.ToString()),
        outer_loop_preheader_bb_(nullptr),
        outer_loop_exit_bb_(nullptr),
        inner_loop_body_bb_(nullptr),
        ir_builder_(ir_builder) {}

  // Adds a loop to the nest. If no loop has been added yet then emit a loop at
  // the current insert point of the given builder. If one or more loops have
  // been added then emit loop inside the body of the last added loop.  If
  // prevent_unrolling is true, then metadata is emitting directing LLVM to not
  // unroll this loop.
  std::unique_ptr<ForLoop> AddLoop(tensorflow::StringPiece suffix,
                                   llvm::Value* start_index,
                                   llvm::Value* end_index, llvm::Value* stride,
                                   bool prevent_unrolling = false,
                                   bool prevent_vectorization = false);

  // Like the above, except that it defaults to a stride of one.
  std::unique_ptr<ForLoop> AddLoop(tensorflow::StringPiece suffix,
                                   llvm::Value* start_index,
                                   llvm::Value* end_index,
                                   bool prevent_unrolling = false,
                                   bool prevent_vectorization = false);

  // A convenient wrapper of the other flavor of AddLoop. The given start and
  // end index are constant.
  std::unique_ptr<ForLoop> AddLoop(int64 start_index, int64 end_index,
                                   int64 stride, tensorflow::StringPiece suffix,
                                   bool prevent_unrolling = false,
                                   bool prevent_vectorization = false);

  // Like the above, except that it defaults to a stride of one.
  std::unique_ptr<ForLoop> AddLoop(int64 start_index, int64 end_index,
                                   tensorflow::StringPiece suffix,
                                   bool prevent_unrolling = false,
                                   bool prevent_vectorization = false);

  // Add loops to iterate through the indices within the specified
  // shape. The returned index collects the induction variables of the
  // loops so that it will iterate through all coordinates within the
  // specified shape.
  //
  // E.g. if you pass in a 2x3 shape, you will get back an index with
  // two entries that are induction variables of the two loops that
  // will be added. That index will iterate through the 6 coordinates
  // within the shape. One possible order for that sequence would be:
  //
  //   (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
  IrArray::Index AddLoopsForShape(const Shape& shape,
                                  tensorflow::StringPiece suffix);

  // Add a loop for each dimension in "dimensions". "suffix" is the
  // name suffix of the indvar and basic blocks in this new loop nest.
  //
  // The return value is an index with the induction variables. The
  // size equals the rank of shape and there is a null for each
  // dimension that is not in "dimensions".
  IrArray::Index AddLoopsForShapeOnDimensions(
      const Shape& shape, tensorflow::gtl::ArraySlice<int64> dimensions,
      tensorflow::StringPiece suffix);

  // Convenience methods which return particular basic blocks of the outermost
  // or innermost loops. These methods return nullptr if no loops have been
  // added yet.
  llvm::BasicBlock* GetOuterLoopPreheaderBasicBlock() {
    return outer_loop_preheader_bb_;
  }
  llvm::BasicBlock* GetOuterLoopExitBasicBlock() { return outer_loop_exit_bb_; }
  llvm::BasicBlock* GetInnerLoopBodyBasicBlock() { return inner_loop_body_bb_; }

 private:
  // Human-friendly name of the loop nest.
  string name_;

  // The preheader and exit basic block of the outermost loop, or nullptr if no
  // loop has been added yet.
  llvm::BasicBlock* outer_loop_preheader_bb_;
  llvm::BasicBlock* outer_loop_exit_bb_;

  // The body basic block of the most-recently added loop, or nullptr if no loop
  // has been added yet.
  llvm::BasicBlock* inner_loop_body_bb_;

  llvm::IRBuilder<>* ir_builder_;

  TF_DISALLOW_COPY_AND_ASSIGN(ForLoopNest);
};

}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_LOOP_H_
