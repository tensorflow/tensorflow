/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TRANSLATE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TRANSLATE_UTILS_H_

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

// Populates the tf.versions attribute on a module, given a corresponding
// graph VersionDef proto.
void PopulateTfVersions(mlir::ModuleOp module, const VersionDef& versions);

// Extracts TensorFlow GraphDef version information from the given module.
// Returns failure if version attribute is missing or any of the sub attributes
// are invalid.
mlir::LogicalResult ExtractTfVersions(mlir::ModuleOp module,
                                      VersionDef* versions);

// Returns TensorFlow GraphDef producer version for the given module. Returns an
// error if the version information is missing for the module or is not valid.
::stream_executor::port::StatusOr<int64_t> GetTfGraphProducerVersion(
    mlir::ModuleOp module);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TRANSLATE_UTILS_H_
