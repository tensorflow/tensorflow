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

#include "xla/hlo/translate/hlo_to_mhlo/stack_location_utils.h"

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"

namespace mlir {
namespace hlo {
mlir::Location GetLocationFromFrameIndex(int frame_id, mlir::Builder& builder,
                                         const xla::HloModule* hlo_module,
                                         StackFrameLocationCache* cache) {
  // Location of the memoized tail of the chain, if any; extended one frame at a
  // time below.
  mlir::LocationAttr location;

  // Walk towards the root until the chain ends or reaches a memoized frame.
  llvm::SmallVector<std::pair<xla::StackFrameId, xla::HloStackFrame>> frames;
  xla::StackFrameId current{frame_id};
  while (current.valid()) {
    if (cache != nullptr) {
      if (mlir::LocationAttr cached = cache->Lookup(current)) {
        location = cached;
        break;
      }
    }
    xla::HloStackFrame frame = hlo_module->get_stack_frame(current);
    if (frame.empty()) {
      break;
    }
    frames.emplace_back(current, frame);
    current = frame.parent_frame_id;
  }

  // Build from the root inward: each frame is the callee of its parent chain,
  // so the result nests as CallSiteLoc(leaf, CallSiteLoc(parent, ... root)),
  // the shape CallSiteLoc::get(leaf, parents) produces.
  for (const auto& [id, frame] : llvm::reverse(frames)) {
    mlir::Location frame_location = mlir::NameLoc::get(
        builder.getStringAttr(xla::ToStringRef(frame.function_name)),
        mlir::FileLineColLoc::get(
            builder.getStringAttr(xla::ToStringRef(frame.file_name)),
            frame.line, frame.column));
    if (location) {
      location = mlir::CallSiteLoc::get(frame_location, location);
    } else {
      location = frame_location;
    }
    if (cache != nullptr) {
      cache->Insert(id, location);
    }
  }

  if (!location) {
    return mlir::UnknownLoc::get(builder.getContext());
  }
  return location;
}
}  // namespace hlo
}  // namespace mlir
