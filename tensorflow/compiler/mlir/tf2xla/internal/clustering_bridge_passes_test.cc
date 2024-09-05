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

#include "tensorflow/compiler/mlir/tf2xla/internal/clustering_bridge_passes.h"

#include <gtest/gtest.h>
#include "mlir/Pass/PassManager.h"  // from @llvm-project

namespace tensorflow {
namespace tf2xla {
namespace internal {

using mlir::OpPassManager;

TEST(ClusteringBridgePassesTest, AddsBridgePasses) {
  OpPassManager pass_manager;
  AddReplicatedBridgeClusteringPipelinePasses(pass_manager);

  EXPECT_EQ(pass_manager.size(), 45);
}

TEST(ClusteringBridgePassesTest, AddsNonTPUBridgePasses) {
  OpPassManager pass_manager;
  AddNonReplicatedBridgeClusteringPipelinePasses(pass_manager);

  EXPECT_EQ(pass_manager.size(), 15);
}

};  // namespace internal
};  // namespace tf2xla
};  // namespace tensorflow
