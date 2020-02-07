/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DUMP_MLIR_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DUMP_MLIR_UTIL_H_

#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Creates a file to use for dumping and returns success if a file could be
// created. The opened file is placed in 'os' and the path of the file used is
// placed in 'filepath'.
//
// If the TF_DUMP_GRAPH_PREFIX environment variable is "-", then the LOG(INFO)
// macro is used instead.
//
// This will create a file name via prefixing `name` with the value of the
// TF_DUMP_GRAPH_PREFIX environment variable if `dirname` is empty and
// suffixing `name` with ".mlir".
Status CreateFileForDumping(llvm::StringRef name,
                            std::unique_ptr<llvm::raw_ostream>* os,
                            std::string* filepath,
                            llvm::StringRef dirname = "");

// Dumps MLIR operation to a file and returns the file name used.
//
// If the TF_DUMP_GRAPH_PREFIX environment variable is "-", then the MLIR
// operation will be logged (using the LOG(INFO) macro) instead.
//
// This will create a file name via prefixing `name` with the value of the
// TF_DUMP_GRAPH_PREFIX environment variable if `dirname` is empty and
// suffixing `name` with ".mlir".
std::string DumpMlirOpToFile(llvm::StringRef name, mlir::Operation* op,
                             llvm::StringRef dirname = "");

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DUMP_MLIR_UTIL_H_
