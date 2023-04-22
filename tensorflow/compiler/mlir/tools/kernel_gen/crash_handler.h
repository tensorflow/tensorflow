// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_CRASH_HANDLER_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_CRASH_HANDLER_H_

#include "llvm/Support/PrettyStackTrace.h"
#include "tensorflow/core/platform/platform.h"

namespace tensorflow {
namespace kernel_gen {

inline void SetCrashReportMessage() {
#if defined(PLATFORM_GOOGLE)
  llvm::setBugReportMsg(
      "The TensorFlow Kernel Generator crashed, see the docs at "
      "go/tf-kernel-gen for debug hints and contact information.\n");
#else
  llvm::setBugReportMsg(
      "The TensorFlow Kernel Generator crashed, please report a bug with the "
      "trace below on https://github.com/tensorflow/tensorflow/issues.\n");
#endif
}
}  // namespace kernel_gen
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_CRASH_HANDLER_H_
