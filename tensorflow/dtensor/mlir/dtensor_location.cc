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
#include <queue>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/utils/name_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {
std::string CreateLocalLocationString(mlir::FileLineColLoc loc) {
  return llvm::formatv(">> {0}:{1}:{2}", loc.getFilename(), loc.getLine(),
                       loc.getColumn())
      .str();
}
}  // namespace

mlir::Location DTensorLocation(mlir::Location loc, llvm::StringRef file,
                               unsigned int line, llvm::StringRef name) {
  // Strip dirname.
  auto split = file.rsplit("/");
  if (!split.second.empty()) file = split.second;
  mlir::Location callee_loc =
      mlir::FileLineColLoc::get(loc.getContext(), file, line, 0);
  std::string new_name = GetNameFromLoc(loc);
  if (!new_name.empty()) {
    if (!name.empty()) {
      new_name = llvm::formatv("{0}/{1}", new_name, name).str();
    }
    callee_loc = mlir::NameLoc::get(
        mlir::StringAttr::get(loc.getContext(), new_name), callee_loc);
  }
  return mlir::CallSiteLoc::get(/*callee=*/callee_loc, /*caller=*/loc);
}

mlir::Location DTensorLocation(mlir::Operation* op, llvm::StringRef file,
                               unsigned int line, llvm::StringRef name) {
  return DTensorLocation(op->getLoc(), file, line, name);
}

std::string DTensorLocationToString(mlir::Location loc) {
  llvm::SmallVector<std::string, 4> stack;
  std::queue<mlir::Location> queue;
  queue.push(loc);

  while (!queue.empty()) {
    mlir::Location& front = queue.front();
    if (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(front)) {
      queue.push(name_loc.getChildLoc());
    } else if (auto callsite_loc = mlir::dyn_cast<mlir::CallSiteLoc>(front)) {
      queue.push(callsite_loc.getCallee());
      queue.push(callsite_loc.getCaller());
    } else if (auto line_loc = mlir::dyn_cast<mlir::FileLineColLoc>(front)) {
      stack.push_back(CreateLocalLocationString(line_loc));
    }
    queue.pop();
  }

  std::reverse(stack.begin(), stack.end());
  std::string s;
  llvm::raw_string_ostream ss(s);
  llvm::interleave(stack, ss, "\n");
  return ss.str();
}

}  // namespace dtensor
}  // namespace tensorflow
