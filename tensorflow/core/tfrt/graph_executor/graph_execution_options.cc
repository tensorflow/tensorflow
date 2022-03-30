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
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"

#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/tfrt/utils/bridge_graph_analysis.h"

namespace tensorflow {
namespace tfrt_stub {

tensorflow::SessionOptions CreateDefaultSessionOptions(
    const GraphExecutionOptions& options) {
  tensorflow::SessionOptions session_options;
  auto& config = session_options.config;

  config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_disable_meta_optimizer(!options.compile_options.enable_grappler);

  // The following configs are constant.

  // Setting use_tfrt to true avoids grappler logic that lowers to v1 control
  // flow. Note that other function inlining (e.g. on StatefulPartitionedCall)
  // is still enabled.
  config.mutable_experimental()->set_use_tfrt(true);
  if (options.enable_grappler_function_optimizer) {
    config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_function_optimization(tensorflow::RewriterConfig::ON);
  } else {
    config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_function_optimization(tensorflow::RewriterConfig::OFF);
  }
  // Do not skip grappler optimization even for small graphs.
  config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_min_graph_nodes(-1);

  return session_options;
}

void UpdateTpuTargetByBridgeCompatibility(
    tensorflow::tfrt_stub::GraphExecutionOptions& options,
    const tensorflow::GraphDef& graph_def) {
  if (options.compile_options.tpu_target ==
      tensorflow::TfrtTpuInfraTarget::kBridgeFallback) {
    auto s = tfrt::CheckTpuMlirBridgeCompatibility(graph_def);
    if (!s.ok()) {
      LOG(INFO)
          << "TFRT detected Bridge unsupported feature, using TF fallback";
      options.compile_options.tpu_target =
          tensorflow::TfrtTpuInfraTarget::kTfFallback;
    } else {
      options.compile_options.tpu_target =
          tensorflow::TfrtTpuInfraTarget::kTpurt;
    }
  }
  LOG(INFO) << "TFRT uses TPU target " << options.compile_options.tpu_target;
}

}  // namespace tfrt_stub
}  // namespace tensorflow
