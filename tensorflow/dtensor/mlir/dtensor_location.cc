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

#include "tensorflow/dtensor/mlir/dtensor_location.h"

#include <algorithm>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace tensorflow {
namespace dtensor {

mlir::Location DTensorLocation(mlir::Location loc, llvm::StringRef file,
                               unsigned int line) {
  // Strip dirname.
  auto split = file.rsplit("/");
  if (!split.second.empty()) file = split.second;
  mlir::Location callee_loc =
      mlir::FileLineColLoc::get(loc.getContext(), file, line, 0);
  return mlir::CallSiteLoc::get(/*callee=*/callee_loc, /*caller=*/loc);
}

mlir::Location DTensorLocation(mlir::Operation* op, llvm::StringRef file,
                               unsigned int line) {
  return DTensorLocation(op->getLoc(), file, line);
}

std::string CreateLocalLocationString(mlir::FileLineColLoc loc) {
  return llvm::formatv(">> {0}:{1}:{2}", loc.getFilename(), loc.getLine(),
                       loc.getColumn())
      .str();
}

std::string DTensorLocationToString(mlir::Location loc) {
  llvm::SmallVector<std::string, 4> stack;
  while (auto callsite_loc = loc.dyn_cast<mlir::CallSiteLoc>()) {
    if (auto callee_loc =
            callsite_loc.getCallee().dyn_cast<mlir::FileLineColLoc>())
      stack.push_back(CreateLocalLocationString(callee_loc));

    loc = callsite_loc.getCaller();
  }

  if (auto file_line_col_loc = loc.dyn_cast<mlir::FileLineColLoc>())
    stack.push_back(CreateLocalLocationString(file_line_col_loc));

  std::reverse(stack.begin(), stack.end());
  std::string s;
  llvm::raw_string_ostream ss(s);
  llvm::interleave(stack, ss, "\n");
  return ss.str();
}

}  // namespace dtensor
}  // namespace tensorflow
