/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/location_importer.h"

#include "mlir/IR/Builders.h"  // from @llvm-project

namespace mlir {
namespace mhlo {

// TODO(herhut): Refactor the format.
mlir::Location GenerateInstructionLocation(
    const xla::HloInstruction* instruction, mlir::MLIRContext* context) {
  mlir::Builder b(context);
  const std::string& op_name = instruction->metadata().op_name();
  if (op_name.empty()) {
    return mlir::NameLoc::get(b.getStringAttr(instruction->name()));
  }

  mlir::Location op_name_loc = mlir::NameLoc::get(b.getStringAttr(op_name));
  const std::string& source_file = instruction->metadata().source_file();
  if (source_file.empty()) {
    return op_name_loc;
  }

  return b.getFusedLoc(
      {op_name_loc,
       mlir::FileLineColLoc::get(b.getContext(), source_file,
                                 instruction->metadata().source_line(), 0)});
}

}  // namespace mhlo
}  // namespace mlir
