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
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
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

// Reads the directory to dump the MLIR module from environment variables.
// Default is reading from TF_DUMP_GRAPH_PREFIX, and if the string is 'sponge'
// read from TEST_UNDECLARED_OUTPUTS_DIR. Returns nullptr if the directory
// cannot be determined and generates a warning message.
std::string GetDumpDirFromEnvVar();

// Dumps a raw string to a file and returns the file name used.
//
// This will create a file name via prefixing `name` with the value of the
// TF_DUMP_GRAPH_PREFIX environment variable if `dirname` is empty and
// suffixing `name` with ".mlir".
std::string DumpRawStringToFile(llvm::StringRef name, llvm::StringRef content,
                                llvm::StringRef dirname = "");

// Enable the crash reproducer on the provided PassManager to the provided
// directory path. If the provided path is empty, it is retrieved from the
// environment variable `MLIR_CRASH_REPRODUCER_DIRECTORY`. If the provided path
// is the string "sponge", the file will be included in the sponge "Output
// Files" by looking up the environment to infer the directory path.
void SetCrashReproducer(mlir::PassManager& pm, llvm::StringRef dir_path = "");

// This applies both the PassManagerCLOptions provided by MLIR along with any
// tensorflow specific options.
//
// Note that this function should be in a more appropriate file, but it is
// unclear what a proper file would be as no other functions would currently be
// in the file also.
void applyTensorflowAndCLOptions(mlir::PassManager& pm,
                                 llvm::StringRef dir_path = "");

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DUMP_MLIR_UTIL_H_
