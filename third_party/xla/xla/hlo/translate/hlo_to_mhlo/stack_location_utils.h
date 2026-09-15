/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSLATE_HLO_TO_MHLO_STACK_LOCATION_UTILS_H_
#define XLA_HLO_TRANSLATE_HLO_TO_MHLO_STACK_LOCATION_UTILS_H_

#include <cstddef>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/service/hlo.pb.h"

namespace mlir {
namespace hlo {

// Memo of the locations GetLocationFromFrameIndex built for the stack frames
// of one HloModule in one MLIRContext. Frames are shared by many instructions,
// so memoizing per frame id builds each frame's location once instead of once
// per instruction that refers to it. A cache must only be used with the module
// it was built for and with one context.
class StackFrameLocationCache {
 public:
  // Frame ids are dense and start at 1, so the memo is indexed by id and sized
  // once from the module's frame table.
  explicit StackFrameLocationCache(const xla::HloModule& module)
      : locations_(module.stack_frames().proto().stack_frames_size() + 1) {}

  // Returns the location of the frame chain starting at `id`, or null if it
  // has not been built yet.
  mlir::LocationAttr Lookup(xla::StackFrameId id) const {
    size_t index = static_cast<size_t>(id.value);
    return index < locations_.size() ? locations_[index] : mlir::LocationAttr();
  }

  // Ids outside the frame table are never memoized.
  void Insert(xla::StackFrameId id, mlir::Location location) {
    size_t index = static_cast<size_t>(id.value);
    if (index < locations_.size()) {
      locations_[index] = location;
    }
  }

 private:
  llvm::SmallVector<mlir::LocationAttr> locations_;
};

// Construct MLIR location from frame index.
// Returns unknown location if frame is not presented.
// When `cache` is given, the locations of all frames on the chain of `frame_id`
// are read from and stored into it.
mlir::Location GetLocationFromFrameIndex(
    int frame_id, mlir::Builder& builder, const xla::HloModule* hlo_module,
    StackFrameLocationCache* cache = nullptr);

}  // namespace hlo
}  // namespace mlir

#endif  // XLA_HLO_TRANSLATE_HLO_TO_MHLO_STACK_LOCATION_UTILS_H_
