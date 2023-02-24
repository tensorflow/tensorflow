/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_SERIALIZER_FLATBUFFER_EXPORT_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_SERIALIZER_FLATBUFFER_EXPORT_H_

#include <map>
#include <string>
#include <unordered_set>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"

namespace mlir {
namespace odml {

// Options for exporting to Flatbuffer.
struct FlatbufferExportOptions {
  // TocoFlags proto. The following fields are migrated.
  // bool emit_builtin_tflite_ops  -> !toco_flags.force_select_tf_ops()
  // bool emit_select_tf_ops       -> toco_flags.enable_select_tf_ops()
  // bool emit_custom_ops          -> toco_flags.allow_custom_ops()
  // bool allow_all_select_tf_ops  -> toco_flags.allow_all_select_tf_ops()
  // std::set<> select_user_tf_ops -> toco_flags.select_user_tf_ops()
  toco::TocoFlags toco_flags;
  // When exporting from SavedModel, this will have the requested tags.
  std::unordered_set<std::string> saved_model_tags;
  // Metadata key/value pairs to write to the flatbuffer.
  std::map<std::string, std::string> metadata;
  // OpOrArgNameMapper to convert location of the op to name in flatbuffer.
  // If not set, a default mapper will be used.
  tensorflow::OpOrArgNameMapper* op_or_arg_name_mapper = nullptr;
};

// Translates the given MLIR `module` into a FlatBuffer and stores the
// serialized flatbuffer into the string.
// Returns true on successful exporting, false otherwise.
bool MlirToFlatBufferTranslateFunction(mlir::ModuleOp module,
                                       const FlatbufferExportOptions& options,
                                       std::string* serialized_flatbuffer);

}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_SERIALIZER_FLATBUFFER_EXPORT_H_
