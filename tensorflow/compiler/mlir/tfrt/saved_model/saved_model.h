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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_SAVED_MODEL_SAVED_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_SAVED_MODEL_SAVED_MODEL_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/core/platform/status.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/support/aligned_buffer.h"

namespace tfrt {
class CoreRuntime;
}

namespace mlir {
class ModuleOp;
}

namespace tensorflow {

struct TFRTSavedModelCompileOptions {
  // TODO(tf-runtime-team): Ideally, compiler should make the decision where
  // to place the variable.
  std::string variable_device = "cpu";
  std::string default_device = "cpu";

  // Enable compiler optimization in TFRT dialect.
  bool enable_optimizer = true;

  // Force data format for all layout sensitive operations, eg. setting it to
  // "NHWC" will changes all data format in the graph to "NHWC" by inserting
  // or removing related tf.Transpose op. Currently the supported formats are
  // "NHWC" and "NCHW".
  //
  // TODO(tf-runtime-team): Ideally compiler should figure out whether the
  // data format should be changed, instead of controlled by users.
  std::string force_data_format;
};

// Map captured global tensors for each function.
void MapFunctionGlobalTensorCapturesFromTFSavedModelMLIR(
    mlir::ModuleOp module,
    llvm::function_ref<
        void(llvm::StringRef func_name,
             llvm::ArrayRef<mlir::tf_saved_model::GlobalTensorOp> captures)>
        map_fn);

// Compile MLIR in TF saved model dialect into BEF.
Status CompileTFSavedModelMLIRToBEF(const TFRTSavedModelCompileOptions& options,
                                    mlir::ModuleOp module,
                                    tfrt::AlignedBuffer<8>* bef_buffer);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_SAVED_MODEL_SAVED_MODEL_H_
