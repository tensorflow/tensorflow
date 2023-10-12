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
#ifndef TENSORFLOW_COMPILER_MLIR_TFR_INTEGRATION_NODE_EXPANSION_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_TFR_INTEGRATION_NODE_EXPANSION_PASS_H_

#include "tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.h"
#include "tensorflow/core/common_runtime/eager/eager_op_rewrite_registry.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tfr {

// An optimization pass that decompose the composite ops in a module according
// to the decomposition library. Currently the decomposition library is loaded
// each time the pass runs. A special environment variable is set to locate the
// decomposition library.
class CompositeOpExpansion : public EagerOpRewrite {
 public:
  CompositeOpExpansion(string name, string file, string line)
      : EagerOpRewrite(name, file, line) {}

  Status Run(EagerOperation* orig_op,
             std::unique_ptr<tensorflow::EagerOperation>* out_op) override;

 private:
  // Whether to run this pass. If this is enabled, the NodeDef will be imported
  // to MLIR even no tf composition file is found.
  bool IsEnabled() {
    const char* tfr_lib_env_val = getenv(string(kTFRLibEnv).c_str());
    return tfr_lib_env_val != nullptr;
  }
};

}  // namespace tfr
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFR_INTEGRATION_NODE_EXPANSION_PASS_H_
