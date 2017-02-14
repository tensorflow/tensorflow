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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REMATERIALIZATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REMATERIALIZATION_H_

#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {

class HloRematerialization {
 public:
  using ShapeSizeFunction = std::function<int64(const Shape&)>;

  // Rematerialize HLO instructions in the entry computation of the given module
  // to reduce maximum memory use below memory_limit_bytes where memory use is
  // defined as the total size of all live HLO instruction values. Parameters
  // and constants are included in memory use estimates. Method parameters:
  //
  //   size_function: Function which returns the size in bytes of the top-level
  //     buffer of the given shape.
  //
  //   memory_limit_bytes: The threshold number of bytes to reduce memory use to
  //     via rematerialization.
  //
  //   hlo_module: HLO module to rematerialize instructions in.
  //
  //   sequence: Should point to an empty HloModuleSequence. Upon return
  //     contains the HLO instruction order which was used for
  //     rematerialization. This is the order in which HLO instructions should
  //     be emitted to minimize memory use.
  //
  // Returns whether any instructions were rematerialized. If memory use cannot
  // be reduced to the given limit then a ResourceExhausted error is
  // returned. If memory use is already below the given limit then no
  // instructions are rematerialized and false is returned.
  //
  // CSE will undo the effects of this optimization and should not be run after
  // this pass. In general, this pass should be run very late immediately before
  // code generation.
  static StatusOr<bool> RematerializeAndSchedule(
      const ShapeSizeFunction& size_function, int64 memory_limit_bytes,
      HloModule* hlo_module,
      SequentialHloOrdering::HloModuleSequence* sequence);

 protected:
  HloRematerialization(const ShapeSizeFunction& size_function,
                       int64 memory_limit_bytes)
      : size_function_(size_function),
        memory_limit_bytes_(memory_limit_bytes) {}
  ~HloRematerialization() {}

  // Runs rematerialization on the given module. Returns whether the module was
  // changed.
  StatusOr<bool> Run(HloModule* module,
                     SequentialHloOrdering::HloModuleSequence* sequence);

  // Rematerializes instructions within the given computation. 'order' is the
  // order in which the computation's instructions will be emitted in the
  // backend. Rematerialized instructions will be added to the HLO computation
  // and inserted into 'order'.
  StatusOr<bool> RematerializeComputation(
      HloComputation* computation, std::vector<const HloInstruction*>* order);

  // Returns the total size of the shape (including nested elements) in bytes.
  int64 TotalSizeBytes(const Shape& shape);

  const ShapeSizeFunction size_function_;
  const int64 memory_limit_bytes_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REMATERIALIZATION_H_
