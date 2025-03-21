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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_BENCHMARK_LITERT_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_BENCHMARK_LITERT_MODEL_H_

#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"
#include "tensorflow/lite/tools/utils.h"

namespace litert {
namespace benchmark {

using ::litert::CompiledModel;
using ::litert::Environment;
using ::litert::Model;
using ::litert::TensorBuffer;
using ::tflite::benchmark::BenchmarkModel;
using ::tflite::benchmark::BenchmarkParam;
using ::tflite::benchmark::BenchmarkParams;
using ::tflite::utils::InputTensorData;

class BenchmarkLiteRtModel : public BenchmarkModel {
 public:
  BenchmarkLiteRtModel() = default;
  explicit BenchmarkLiteRtModel(BenchmarkParams params)
      : BenchmarkModel(std::move(params)) {}
  ~BenchmarkLiteRtModel() override = default;
  static BenchmarkParams DefaultParams() {
    BenchmarkParams default_params = BenchmarkModel::DefaultParams();
    default_params.AddParam("graph", BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("signature_to_run_for",
                            BenchmarkParam::Create<std::string>(""));
    default_params.AddParam("use_xnnpack", BenchmarkParam::Create<bool>(true));
    default_params.AddParam("use_gpu", BenchmarkParam::Create<bool>(false));

    return default_params;
  }

  TfLiteStatus Init() override;

  int64_t MayGetModelFileSize() override {
    std::string fd_or_graph_path = params_.Get<std::string>("graph");
    // Path can be one of the following:
    // 1) File descriptor path: path must be in the format of
    // "fd:%model_fd%:%model_offset%:%model_size%".
    // 2) File path: path to the model file.
    // Please see tensorflow/lite/tools/model_loader.h for more information.
    std::vector<absl::string_view> parts =
        absl::StrSplit(fd_or_graph_path, ':');
    if (!parts.empty() && parts[0] == "fd") {
      int64_t model_size = -1;
      if (parts.size() != 4 || !absl::SimpleAtoi(parts[3], &model_size)) {
        LITERT_LOG(LITERT_ERROR, "Failed to parse model file size: %s",
                   fd_or_graph_path.c_str());
      }
      return model_size;
    }
    std::ifstream in_file(fd_or_graph_path, std::ios::binary | std::ios::ate);
    return in_file.tellg();
  }

  TfLiteStatus RunImpl() override {
    if (!compiled_model_) {
      LITERT_LOG(LITERT_ERROR, "Compiled model not initialized");
      return kTfLiteError;
    }
    auto signature = params_.Get<std::string>("signature_to_run_for");
    if (compiled_model_->Run(signature, *input_buffers_, *output_buffers_)) {
      return kTfLiteOk;
    } else {
      LITERT_LOG(LITERT_ERROR, "Run failed");
      return kTfLiteError;
    }
  }

  uint64_t ComputeInputBytes() override {
    uint64_t total_bytes = 0;
    for (const auto& buffer : *input_buffers_) {
      total_bytes += *buffer.Size();
    }
    return total_bytes;
  }

  InputTensorData CreateRandomTensorData(const litert::TensorBuffer& t,
                                         std::string name) {
    float low_range = 0;
    float high_range = 0;
    tflite::utils::GetDataRangesForType(
        static_cast<TfLiteType>(t.TensorType()->ElementType()), &low_range,
        &high_range);
    return tflite::utils::CreateRandomTensorData(
        name, static_cast<TfLiteType>(t.TensorType()->ElementType()), *t.Size(),
        low_range, high_range);
  }

  TfLiteStatus PrepareInputData() override {
    int index = 0;
    for (auto& buffer : *input_buffers_) {
      auto t_data =
          CreateRandomTensorData(buffer, "input_" + std::to_string(index));
      buffer.Write<char>(absl::MakeSpan(
          reinterpret_cast<char*>(t_data.data.get()), t_data.bytes));
      ++index;
    }
    return kTfLiteOk;
  }

  TfLiteStatus ResetInputsAndOutputs() override { return kTfLiteOk; }

 private:
  Model model_;
  std::unique_ptr<litert::CompiledModel> compiled_model_;
  std::unique_ptr<std::vector<litert::TensorBuffer>> input_buffers_;
  std::unique_ptr<std::vector<litert::TensorBuffer>> output_buffers_;
};

}  // namespace benchmark
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_BENCHMARK_LITERT_MODEL_H_
