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

#include <gtest/gtest.h>
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tsl/lib/core/status_test_util.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {
namespace {

using mlir::ModuleOp;

TEST(FunctionTf2xlaClusteringBridgeTest, ClustersTf) {
  ModuleOp module;
  TF_ASSERT_OK(
      RunFunctionTf2xlaClusteringBridge(module, DeviceType::XLA_TPU_JIT));
}

}  // namespace
}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow
