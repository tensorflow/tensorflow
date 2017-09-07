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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE__CALL_INLINER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE__CALL_INLINER_H_

#include <deque>

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// For every kCall operation in the main computation, we inline the body of the
// called function, and proceed recursively.
class CallInliner : public HloPassInterface {
 public:
  ~CallInliner() override = default;
  tensorflow::StringPiece name() const override { return "CallInliner"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  // Replaces the given call operation -- which must be an operation inside the
  // entry computation with opcode kCall -- with the called computation's body,
  // such that the called computation is inline in the entry computation.
  //
  // On successful inlining, the inlined computation may have itself contained
  // calls; if so, they are added to the work_queue.
  Status ReplaceWithInlinedBody(HloInstruction* call,
                                std::deque<HloInstruction*>* work_queue);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE__CALL_INLINER_H_
