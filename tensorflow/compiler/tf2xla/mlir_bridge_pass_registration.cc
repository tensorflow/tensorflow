/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "tensorflow/compiler/mlir/mlir_graph_optimization_pass.h"
#include "tensorflow/compiler/tf2xla/mlir_bridge_pass.h"

namespace tensorflow {
namespace {
constexpr int kMlirBridgePriority = 10;
}

static mlir_pass_registration::MlirOptimizationPassRegistration
    register_mlir_bridge_pass(kMlirBridgePriority,
                              std::make_unique<MlirBridgePass>());

static mlir_pass_registration::MlirV1CompatOptimizationPassRegistration
    register_v1_compat_mlir_bridge_pass(
        std::make_unique<MlirBridgeV1CompatPass>());

}  // namespace tensorflow
