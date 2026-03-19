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

#include <ostream>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
// TODO(b/200579737): using FunctionRegistry is simpler than the OSS trick.
#include "tensorflow/core/tfrt/utils/bridge_graph_analysis.h"

namespace tensorflow {
namespace tfrt_stub {

tensorflow::SessionOptions CreateDefaultSessionOptions(
    const GraphExecutionOptions& options) {
  tensorflow::SessionOptions session_options;
  auto& config = session_options.config;

  *config.mutable_experimental()->mutable_session_metadata() =
      options.model_metadata;

  *config.mutable_graph_options() = options.compile_options.graph_options;

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

  if (options.tfrt_gpu_parallelism > 1) {
    if (!options.compile_options.use_gpu_compile_and_execute_op) {
      LOG(WARNING)
          << "tfrt_gpu_parallelism > 1, but fused GPU kernel is not used. "
             "Non-fused GPU kernel does not support multiple GPU devices.";
    }
    config.mutable_gpu_options()
        ->mutable_experimental()
        ->set_num_virtual_devices_per_gpu(options.tfrt_gpu_parallelism);
  }
  if (options.gpu_system_memory_size_in_mb > 0) {
    config.mutable_gpu_options()
        ->mutable_experimental()
        ->set_gpu_system_memory_size_in_mb(
            options.gpu_system_memory_size_in_mb);
  }
  return session_options;
}

void UpdateTpuTargetByBridgeCompatibility(
    tensorflow::tfrt_stub::GraphExecutionOptions& options,
    const tensorflow::GraphDef& graph_def) {
  if (options.compile_options.device_target ==
      tensorflow::TfrtDeviceInfraTarget::kBridgeFallback) {
    auto s = tfrt::CheckTpuMlirBridgeCompatibility(graph_def);
    if (!s.ok()) {
      LOG(INFO)
          << "TFRT detected Bridge unsupported feature, using TF fallback";
      options.compile_options.device_target =
          tensorflow::TfrtDeviceInfraTarget::kTfFallback;
    } else {
      options.compile_options.device_target =
          tensorflow::TfrtDeviceInfraTarget::kTpurt;
    }
  }

  // We don't need to check for SPMD fallback for non TFRT TPU path.
  //
  // TODO(b/288096487): Clean up the enums to reflect the device target better.
  // One option is to use a  custom target enum for the opaque backend.
  if (options.compile_options.device_target !=
          tensorflow::TfrtDeviceInfraTarget::kCpu &&
      options.compile_options.device_target !=
          tensorflow::TfrtDeviceInfraTarget::kGpu) {
    // TODO(linchai): Once native support for SPMD models is fully rollout,
    // remove the fallback logic.
    if (!(tfrt::CheckSpmdGraph(graph_def).ok() ||
          options.compile_options.tpu_fuse_ops)) {
      options.compile_options.device_target =
          tensorflow::TfrtDeviceInfraTarget::kTfFallback;
    }
    LOG(INFO) << "TFRT uses device target "
              << options.compile_options.device_target;
  }
}

std::ostream& operator<<(std::ostream& os,
                         const GraphExecutionOptions& options) {
  return os << "{" << "run_placer_grappler_on_functions = "
            << options.run_placer_grappler_on_functions
            << ", enable_grappler_function_optimizer = "
            << options.enable_grappler_function_optimizer
            << ", enable_tfrt_gpu = " << options.enable_tfrt_gpu
            << ", use_ifrt = " << options.use_ifrt << ", runtime = "
            << options.runtime
            // clang-tidy off
            << ", model_metadata = "
            << options.model_metadata.DebugString()
            // clang-tidy on
            << ", compile_options = " << options.compile_options << "}";
}

}  // namespace tfrt_stub
}  // namespace tensorflow
