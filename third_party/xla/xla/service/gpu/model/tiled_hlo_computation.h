/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_MODEL_TILED_HLO_COMPUTATION_H_
#define XLA_SERVICE_GPU_MODEL_TILED_HLO_COMPUTATION_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xla/iterator_util.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "tsl/lib/gtl/iterator_range.h"

namespace xla {
namespace gpu {

// Stores TiledHloInstructions in the computation.
//  * Instructions reference each other with non-owning pointers.
//  * Instructions with the same tiling parameters are CSE-ed during
//  construction.
//  * Instructions are stored in def-before-use order.
//  * The last element in the vector in the root instruction.
class TiledHloComputation {
 public:
  // Creates a computation from a list of instructions. The instructions are
  // expected to be sorted in def-before-use order.
  static TiledHloComputation FromSortedTiledHloInstructions(
      std::vector<std::unique_ptr<TiledHloInstruction>> instructions) {
    return TiledHloComputation(std::move(instructions));
  }

  // Returns an iterator range over the instructions in the computation in
  // def-before-use order.
  tsl::gtl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<TiledHloInstruction>>::const_iterator>>
  instructions() const {
    return {MakeUnwrappingIterator(instructions_.begin()),
            MakeUnwrappingIterator(instructions_.end())};
  }

  // Returns the root instruction of the computation.
  const TiledHloInstruction* GetRoot() const {
    return instructions_.back().get();
  }

  // Returns a string representation of the computation. Used only for error
  // messages and debugging.
  std::string ToString() const;

 private:
  explicit TiledHloComputation(
      std::vector<std::unique_ptr<TiledHloInstruction>> instructions)
      : instructions_(std::move(instructions)) {}

  // Stores instructions in the computation in def-before-use order.
  std::vector<std::unique_ptr<TiledHloInstruction>> instructions_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TILED_HLO_COMPUTATION_H_
