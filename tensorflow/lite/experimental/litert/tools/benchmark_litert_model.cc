/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/litert/tools/benchmark_litert_model.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_compilation_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"

namespace litert::benchmark {
namespace {
using ::litert::CompilationOptions;
using ::litert::CompiledModel;
using ::litert::TensorBuffer;

CompilationOptions CreateCompiledModelOptions(const BenchmarkParams& params) {
  auto use_gpu = params.Get<bool>("use_gpu");
  CompilationOptions compilation_options =
      *litert::CompilationOptions::Create();
  if (use_gpu) {
    compilation_options.SetHardwareAccelerators(
        LiteRtHwAccelerators::kLiteRtHwAcceleratorGpu);
  }
  return compilation_options;
}
}  // namespace

TfLiteStatus BenchmarkLiteRtModel::Init() {
  std::string fd_or_graph_path = params_.Get<std::string>("graph");
  LITERT_LOG(LITERT_INFO, "Loading model from: %s", fd_or_graph_path.c_str());
  model_ = *litert::Model::CreateFromFile(fd_or_graph_path);
  if (!model_) {
    LITERT_LOG(LITERT_ERROR, "Failed to load model: %s",
               fd_or_graph_path.c_str());
    return kTfLiteError;
  }

  auto env = Environment::Create({});
  if (!env) {
    LITERT_LOG(LITERT_ERROR, "Failed to create litert environment.");
    return kTfLiteError;
  }

  auto compilation_options = CreateCompiledModelOptions(params_);
  auto compiled_model_result =
      litert::CompiledModel::Create(*env, model_, compilation_options);
  if (!compiled_model_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to create compiled model.");
    return kTfLiteError;
  }

  compiled_model_ = std::make_unique<litert::CompiledModel>(
      std::move(*compiled_model_result));
  auto signature = params_.Get<std::string>("signature_to_run_for");
  auto input_buffers_result = compiled_model_->CreateInputBuffers(signature);
  if (!input_buffers_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to create input buffers.");
    return kTfLiteError;
  }
  input_buffers_ = std::make_unique<std::vector<litert::TensorBuffer>>(
      std::move(*input_buffers_result));

  auto output_buffers_result = compiled_model_->CreateOutputBuffers(signature);
  if (!output_buffers_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to create output buffers.");
    return kTfLiteError;
  }
  output_buffers_ = std::make_unique<std::vector<litert::TensorBuffer>>(
      std::move(*output_buffers_result));

  return kTfLiteOk;
}
}  // namespace litert::benchmark
