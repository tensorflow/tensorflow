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

#include "tensorflow/compiler/mlir/tf2xla/api/v2/cluster_tf.h"

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/api/v2/device_type.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {

using mlir::ModuleOp;

tensorflow::Status RunFunctionTf2xlaClusteringBridge(ModuleOp module,
                                                     DeviceType device_type) {
  return tsl::OkStatus();
}

}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow
