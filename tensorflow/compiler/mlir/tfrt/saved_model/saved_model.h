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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_SAVED_MODEL_SAVED_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_SAVED_MODEL_SAVED_MODEL_H_

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime

namespace tfrt {
class CoreRuntime;
}

namespace mlir {
class ModuleOp;
}

namespace tensorflow {

// TFRTSavedModelSignatureInfo contains the metadata for a signature in the
// savedmodel such as function name, inputs/outputs' names and types. This can
// be used to retrieve these information in a tf_saved_model module.
struct TFRTSavedModelSignatureInfo {
  llvm::StringRef func_name;

  // The following are metadata for inputs.
  llvm::ArrayRef<llvm::StringRef> input_names;
  llvm::ArrayRef<
      std::pair<tensorflow::DataType, tensorflow::PartialTensorShape>>
      input_specs;
  llvm::ArrayRef<llvm::StringRef> input_devices;

  // The following are metadata for outputs.
  llvm::ArrayRef<llvm::StringRef> output_names;
  llvm::ArrayRef<
      std::pair<tensorflow::DataType, tensorflow::PartialTensorShape>>
      output_specs;

  // The following are metadata for bound_inputs, ie. captures.
  llvm::ArrayRef<mlir::Operation*> bound_inputs;
};

// Apply `map_fn` on every exported function in the module with the
// corresponding signature metadata populated in TFRTSavedModelSignatureInfo for
// the function.
absl::Status MapFunctionSignaturesFromTFSavedModelMLIR(
    mlir::ModuleOp module,
    llvm::function_ref<void(const TFRTSavedModelSignatureInfo&)> map_fn);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_SAVED_MODEL_SAVED_MODEL_H_
