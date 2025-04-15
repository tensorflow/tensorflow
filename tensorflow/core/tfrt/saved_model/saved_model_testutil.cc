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
#include "tensorflow/core/tfrt/saved_model/saved_model_testutil.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"

#if defined(PLATFORM_GOOGLE)
ABSL_FLAG(bool, enable_optimizer, true,
          "enable optimizations in CoreRT dialect (e.g., constant-folding)");
ABSL_FLAG(std::string, force_data_format, "",
          "force data format for all layout sensitive operations. Currently "
          "the supported formats are 'NHWC' and 'NCHW'");

ABSL_FLAG(
    bool, enable_grappler, false,
    "If true, run grappler passes before importing the SavedModel into MLIR.");

ABSL_FLAG(bool, enable_mlrt, false,
          "If true, the runtime will use MLRT interpreter for host execution.");
#endif

namespace tensorflow {
namespace tfrt_stub {

std::unique_ptr<tensorflow::tfrt_stub::Runtime> DefaultTfrtRuntime(
    int num_threads) {
  return tensorflow::tfrt_stub::Runtime::Create(
      tensorflow::tfrt_stub::WrapDefaultWorkQueue(
          tfrt::CreateMultiThreadedWorkQueue(num_threads, num_threads)));
}

SavedModel::Options DefaultSavedModelOptions(
    tensorflow::tfrt_stub::Runtime* runtime,
    std::optional<UserSavedModelOptions> user_options) {
  SavedModel::Options options(runtime);
  auto& compile_options = options.graph_execution_options.compile_options;
#if defined(PLATFORM_GOOGLE)
  options.graph_execution_options.enable_mlrt =
      absl::GetFlag(FLAGS_enable_mlrt);
  compile_options.enable_optimizer = absl::GetFlag(FLAGS_enable_optimizer);
  compile_options.enable_grappler = absl::GetFlag(FLAGS_enable_grappler);
  compile_options.force_data_format = absl::GetFlag(FLAGS_force_data_format);
#endif

  if (user_options) {
    options.graph_execution_options.enable_mlrt = user_options->enable_mlrt;
    compile_options.enable_optimizer = user_options->enable_optimizer;
    compile_options.enable_grappler = user_options->enable_grappler;
    compile_options.force_data_format = user_options->force_data_format;
  }
  return options;
}

TFRTSavedModelTest::TFRTSavedModelTest(const std::string& saved_model_dir)
    : TFRTSavedModelTest(saved_model_dir,
                         DefaultTfrtRuntime(/*num_threads=*/1)) {}

TFRTSavedModelTest::TFRTSavedModelTest(
    const std::string& saved_model_dir,
    std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime)
    : runtime_(std::move(runtime)) {
  CHECK(runtime_);
  auto options = DefaultSavedModelOptions(runtime_.get());

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_DCHECK_OK(saved_model.status());
  saved_model_ = *std::move(saved_model);
}

// Compute the results using TF1 session loaded from the saved model. In
// addition to returning the result tensors, it also fills `bundle` with the
// loaded savedmodel. This is useful as sometimes the result tensors may only be
// valid when the bundle is alive.
void ComputeCurrentTFResult(const std::string& saved_model_dir,
                            const std::string& signature_name,
                            const std::vector<std::string>& input_names,
                            const std::vector<tensorflow::Tensor>& inputs,
                            const std::vector<std::string>& output_names,
                            std::vector<tensorflow::Tensor>* outputs,
                            tensorflow::SavedModelBundle* bundle,
                            bool enable_mlir_bridge, bool disable_grappler) {
  DCHECK(bundle);
  tensorflow::SessionOptions session_options;
  session_options.config.mutable_experimental()->set_enable_mlir_bridge(
      enable_mlir_bridge);
  // Disable grappler optimization for numerical analysis.
  if (disable_grappler) {
    session_options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_disable_meta_optimizer(true);
  }
  TF_CHECK_OK(tensorflow::LoadSavedModel(session_options, /*run_options=*/{},
                                         saved_model_dir,
                                         /* tags = */ {"serve"}, bundle));

  const auto& signature_def =
      bundle->meta_graph_def.signature_def().at(signature_name);

  std::vector<std::pair<std::string, tensorflow::Tensor>> session_inputs;
  session_inputs.reserve(inputs.size());
  for (const auto& iter : llvm::zip(input_names, inputs)) {
    const auto& node_name = signature_def.inputs().at(std::get<0>(iter)).name();
    session_inputs.emplace_back(node_name, std::get<1>(iter));
  }

  std::vector<std::string> session_output_names;
  session_output_names.reserve(output_names.size());
  for (const auto& output_name : output_names) {
    const auto& node_name = signature_def.outputs().at(output_name).name();
    session_output_names.push_back(node_name);
  }

  TF_CHECK_OK(bundle->GetSession()->Run(session_inputs, session_output_names,
                                        {}, outputs));
}

void ComputeCurrentTFResult(const std::string& saved_model_dir,
                            const std::string& signature_name,
                            const std::vector<std::string>& input_names,
                            const std::vector<tensorflow::Tensor>& inputs,
                            const std::vector<std::string>& output_names,
                            std::vector<tensorflow::Tensor>* outputs,
                            bool enable_mlir_bridge, bool disable_grappler) {
  tensorflow::SavedModelBundle bundle;
  ComputeCurrentTFResult(saved_model_dir, signature_name, input_names, inputs,
                         output_names, outputs, &bundle, enable_mlir_bridge,
                         disable_grappler);
}

void ExpectTensorEqual(const tensorflow::Tensor& x, const tensorflow::Tensor& y,
                       std::optional<double> error) {
  DCHECK_EQ(x.dtype(), y.dtype());
  VLOG(1) << "TFRT result: " << x.DebugString();
  VLOG(1) << "TF result  : " << y.DebugString();
  switch (y.dtype()) {
    case tensorflow::DT_STRING:
      tensorflow::test::ExpectTensorEqual<tensorflow::tstring>(x, y);
      break;
    case tensorflow::DT_FLOAT:
    case tensorflow::DT_DOUBLE:
      if (error) {
        tensorflow::test::ExpectClose(x, y, *error, /*rtol=*/0.0);
      } else {
        tensorflow::test::ExpectEqual(x, y);
      }
      break;
    default:
      tensorflow::test::ExpectEqual(x, y);
      break;
  }
}

SavedModel::Options DefaultTpuModelOptions(
    tensorflow::tfrt_stub::Runtime* runtime,
    tensorflow::TfrtDeviceInfraTarget device_target) {
  SavedModel::Options options(runtime);
#if defined(PLATFORM_GOOGLE)
  options.graph_execution_options.enable_mlrt =
      absl::GetFlag(FLAGS_enable_mlrt);
#endif
  auto& compile_options = options.graph_execution_options.compile_options;
  compile_options.variable_device =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  compile_options.enable_optimizer = false;
  compile_options.enable_grappler = true;
  compile_options.device_target = device_target;
  compile_options.hoist_invariant_ops = true;
  compile_options.sink_in_invariant_ops = true;
  compile_options.cost_threshold =
      1024;  // Servo currently uses 1024 as threshold for TPU models

  return options;
}

}  // namespace tfrt_stub
}  // namespace tensorflow
