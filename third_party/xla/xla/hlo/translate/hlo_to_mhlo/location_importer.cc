/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/translate/hlo_to_mhlo/location_importer.h"

#include <string>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/hlo/translate/hlo_to_mhlo/stack_location_utils.h"
#include "xla/mlir_hlo/utils/unregistered_attributes.h"

namespace mlir {
namespace hlo {

mlir::Location GenerateInstructionLocation(
    const xla::HloInstruction* instruction, mlir::MLIRContext* context) {
  mlir::Builder b(context);

  auto fuse_original_value_if_present = [&](mlir::Location loc) {
    auto original_value = instruction->original_value();
    if (original_value) {
      return b.getFusedLoc(
          {loc, mlir::NameLoc::get(
                    b.getStringAttr(std::string(xla::kMhloOriginalValueAttr) +
                                    "={" + original_value->ToString() + "}"))});
    }
    return loc;
  };

  const std::string& op_name = instruction->metadata().op_name();
  if (op_name.empty()) {
    return fuse_original_value_if_present(mlir::NameLoc::get(
        b.getStringAttr(xla::ToStringRef(instruction->name()))));
  }

  if (instruction->metadata().stack_frame_id() != 0) {
    mlir::Location frame_location =
        GetLocationFromFrameIndex(instruction->metadata().stack_frame_id(), b,
                                  instruction->parent()->parent());

    if (!isa<mlir::UnknownLoc>(frame_location)) {
      return fuse_original_value_if_present(
          mlir::NameLoc::get(b.getStringAttr(op_name), frame_location));
    }
  }

  mlir::Location op_name_loc = mlir::NameLoc::get(b.getStringAttr(op_name));
  const std::string& source_file = instruction->metadata().source_file();

  if (source_file.empty()) {
    return fuse_original_value_if_present(op_name_loc);
  }

  return fuse_original_value_if_present(b.getFusedLoc(
      {op_name_loc,
       mlir::FileLineColLoc::get(b.getContext(), source_file,
                                 instruction->metadata().source_line(), 0)}));
}

}  // namespace hlo
}  // namespace mlir
