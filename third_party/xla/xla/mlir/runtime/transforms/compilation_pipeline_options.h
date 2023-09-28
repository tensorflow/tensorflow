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

#ifndef XLA_MLIR_RUNTIME_TRANSFORMS_COMPILATION_PIPELINE_OPTIONS_H_
#define XLA_MLIR_RUNTIME_TRANSFORMS_COMPILATION_PIPELINE_OPTIONS_H_

#include <functional>

#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "xla/runtime/type_id.h"

namespace xla {
namespace runtime {

// Common options for default XLA-{CPU,GPU} compilation pipelines.
struct CompilationPipelineOptions {
  // Register names for the TypeIDs used for encoding types of custom arguments
  // and attributes.
  std::function<void(TypeIDNameRegistry&)> populate_type_id_names;

  // Add type conversions from user-defined types to LLVM types. These
  // conversions are required for lowering runtime operations to the
  // corresponding runtime APIs (including custom calls).
  std::function<void(mlir::TypeConverter&)> populate_type_conversions;

  // Add user-defined encoding for JitRt custom call arguments and attributes.
  //
  // Custom encodings allow to pass dialect-specific attributes (enums and
  // structs) to the custom calls, and decode them into dialect-specific runtime
  // values in the custom call handlers (see custom_call_to_llvm.h for details).
  std::function<void(CustomCallArgEncodingSet&)> populate_arg_encodings;
  std::function<void(CustomCallRetEncodingSet&)> populate_ret_encodings;
  std::function<void(CustomCallAttrEncodingSet&)> populate_attr_encodings;
};

}  // namespace runtime
}  // namespace xla

#endif  // XLA_MLIR_RUNTIME_TRANSFORMS_COMPILATION_PIPELINE_OPTIONS_H_
