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

#include <vector>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"

namespace mlir {
namespace hlo {
mlir::Location GetLocationFromFrameIndex(int frame_id, mlir::Builder& builder,
                                         const xla::HloModule* hlo_module) {
  std::vector<mlir::Location> stack_locations;
  while (frame_id != 0) {
    xla::HloModule::StackFrame frame = hlo_module->get_stack_frame(frame_id);

    if (frame.empty()) {
      break;
    }

    stack_locations.push_back(mlir::NameLoc::get(
        builder.getStringAttr(xla::ToStringRef(frame.function_name)),
        mlir::FileLineColLoc::get(
            builder.getStringAttr(xla::ToStringRef(frame.file_name)),
            frame.line, frame.column)));

    frame_id = frame.parent_frame_id;
  }

  if (stack_locations.empty()) {
    return mlir::UnknownLoc::get(builder.getContext());
  }

  if (stack_locations.size() == 1) {
    return stack_locations[0];
  }

  ArrayRef stack_locations_ref = stack_locations;
  return mlir::CallSiteLoc::get(stack_locations[0],
                                stack_locations_ref.drop_front());
}
}  // namespace hlo
}  // namespace mlir
