/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_MLIR_IMPORT_OPTIONS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_MLIR_IMPORT_OPTIONS_H_

namespace tensorflow {

// TODO(jpienaar): This file and class are confusingly named. This seems to be
// a SavedModel only import options file that exposes a subset of the
// GraphImportConfig options, but the naming would make one think it is more
// general.
struct MLIRImportOptions {
  // If true, functionalize the input graph before importing it into MLIR.
  bool upgrade_legacy = false;

  // Whether to unconditionally use the shape set via _output_shapes on import.
  bool unconditionally_use_set_output_shapes = false;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_MLIR_IMPORT_OPTIONS_H_
