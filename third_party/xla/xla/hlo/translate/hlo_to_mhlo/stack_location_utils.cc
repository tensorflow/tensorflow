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

#include <cstddef>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"

namespace mlir {
namespace hlo {
mlir::Location GetLocationFromFrameIndex(
    int frame_id, mlir::Builder& builder, const xla::HloModule* hlo_module,
    llvm::SmallVectorImpl<mlir::LocationAttr>* frame_locations) {
  // Frame ids are dense and start at 1, so the memo is indexed by id and sized
  // once from the module's frame table.
  if (frame_locations != nullptr && frame_locations->empty()) {
    frame_locations->resize(
        hlo_module->stack_frames().proto().stack_frames_size() + 1);
  }
  auto memoized = [&](int id) -> mlir::LocationAttr* {
    if (frame_locations == nullptr ||
        static_cast<size_t>(id) >= frame_locations->size()) {
      return nullptr;
    }
    return &(*frame_locations)[id];
  };

  // Location of the memoized tail of the chain, if any; extended one frame at a
  // time below.
  mlir::LocationAttr location;

  // Walk towards the root until the chain ends or reaches a memoized frame.
  // Most walks stop within a few frames: the leaf is new and its parents are
  // memoized, so the frames stay inline.
  llvm::SmallVector<std::pair<int, xla::HloModule::StackFrame>, 8> frames;
  xla::StackFrameId id{frame_id};
  while (id.valid()) {
    if (mlir::LocationAttr* cached = memoized(id.value); cached && *cached) {
      location = *cached;
      break;
    }
    xla::HloModule::StackFrame frame = hlo_module->get_stack_frame(id);
    if (frame.empty()) {
      break;
    }
    frames.emplace_back(id.value, frame);
    id = frame.parent_frame_id;
  }

  // Build from the root inward: each frame is the callee of its parent chain,
  // so the result nests as CallSiteLoc(leaf, CallSiteLoc(parent, ... root)),
  // the shape CallSiteLoc::get(leaf, parents) produces.
  for (const auto& [pending_id, frame] : llvm::reverse(frames)) {
    mlir::Location frame_location = mlir::NameLoc::get(
        builder.getStringAttr(xla::ToStringRef(frame.function_name)),
        mlir::FileLineColLoc::get(
            builder.getStringAttr(xla::ToStringRef(frame.file_name)),
            frame.line, frame.column));
    location = location ? mlir::CallSiteLoc::get(frame_location, location)
                        : frame_location;
    if (mlir::LocationAttr* slot = memoized(pending_id)) {
      *slot = location;
    }
  }

  if (!location) {
    return mlir::UnknownLoc::get(builder.getContext());
  }
  return location;
}
}  // namespace hlo
}  // namespace mlir
