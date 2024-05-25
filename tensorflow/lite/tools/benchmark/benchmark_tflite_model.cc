/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "ruy/profiler/profiler.h"  // from @ruy
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/core/signature_runner.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profile_summary_formatter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/benchmark/profiling_listener.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/tools/model_loader.h"
#include "tensorflow/lite/tools/utils.h"

void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);

// Version with Weak linker attribute doing nothing: if someone links this
// library with another definition of this function (presumably to actually
// register custom ops), that version will be used instead.
void ABSL_ATTRIBUTE_WEAK
RegisterSelectedOps(::tflite::MutableOpResolver* resolver) {}

namespace tflite {
namespace benchmark {
namespace {
using utils::InputTensorData;
using utils::VoidUniquePtr;

// Backward compat with previous approach to enabling op profiling.
#if defined(TFLITE_PROFILING_ENABLED)
constexpr bool kOpProfilingEnabledDefault = true;
#else
constexpr bool kOpProfilingEnabledDefault = false;
#endif

// Op profiling output modes.
constexpr char kOpProfilingOutputModeStdout[] = "stdout";
constexpr char kOpProfilingOutputModeCsv[] = "csv";
constexpr char kOpProfilingOutputModeProto[] = "proto";

const char* kOpProfilingOutputModes[] = {kOpProfilingOutputModeStdout,
                                         kOpProfilingOutputModeCsv,
                                         kOpProfilingOutputModeProto};

// Dumps ruy profiling events if the ruy profiler is enabled.
class RuyProfileListener : public BenchmarkListener {
 public:
  void OnBenchmarkStart(const BenchmarkParams& params) override;

  void OnBenchmarkEnd(const BenchmarkResults& results) override;

 private:
  std::unique_ptr<ruy::profiler::ScopeProfile> ruy_profile_;
};

void RuyProfileListener::OnBenchmarkStart(const BenchmarkParams& params) {
  ruy_profile_ = std::make_unique<ruy::profiler::ScopeProfile>();
}

void RuyProfileListener::OnBenchmarkEnd(const BenchmarkResults& results) {
  ruy_profile_ = nullptr;
}

class InterpreterStatePrinter : public BenchmarkListener {
 public:
  explicit InterpreterStatePrinter(Interpreter* interpreter)
      : interpreter_(interpreter) {}

  void OnBenchmarkStart(const BenchmarkParams& params) override {
    params_ = &params;
    if (params_->Get<bool>("print_preinvoke_state")) {
      TFLITE_LOG(INFO) << "\n====Printing out TfLite interpreter pre-invoke "
                          "state begins====";
      tflite::PrintInterpreterState(
          interpreter_, params_->Get<int32_t>("tensor_name_display_length"),
          params_->Get<int32_t>("tensor_type_display_length"),
          params_->Get<int32_t>("alloc_type_display_length"));
      TFLITE_LOG(INFO) << "====Printing out TfLite interpreter pre-invoke "
                          "state ends====\n";
    }
  }

  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    if (params_->Get<bool>("print_postinvoke_state")) {
      TFLITE_LOG(INFO) << "\n====Printing out TfLite interpreter post-invoke "
                          "state begins====";
      tflite::PrintInterpreterState(
          interpreter_, params_->Get<int32_t>("tensor_name_display_length"),
          params_->Get<int32_t>("tensor_type_display_length"),
          params_->Get<int32_t>("alloc_type_display_length"));
      TFLITE_LOG(INFO) << "====Printing out TfLite interpreter post-invoke "
                          "state ends====\n";
    }
  }

 private:
  Interpreter* const interpreter_ = nullptr;  // not own the memory.
  const BenchmarkParams* params_ = nullptr;   // not own the memory.
};

class OutputSaver : public BenchmarkListener {
 public:
  explicit OutputSaver(BenchmarkInterpreterRunner* runner)
      : interpreter_runner_(runner) {}

  void OnBenchmarkStart(const BenchmarkParams& params) override {
    params_ = &params;
  }

  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    std::string path = params_->Get<std::string>("output_filepath");
    if (path.empty()) return;

    std::ofstream ofs(path, std::ofstream::out);
    if (ofs.good()) {
      for (int i = 0; i < interpreter_runner_->outputs().size(); i++) {
        int tensor_index = interpreter_runner_->outputs()[i];
        ofs.write(interpreter_runner_->tensor(tensor_index)->data.raw,
                  interpreter_runner_->tensor(tensor_index)->bytes);
      }
      ofs.close();
    }
  }

 private:
  BenchmarkInterpreterRunner* const interpreter_runner_;
  const BenchmarkParams* params_ = nullptr;
};

std::vector<std::string> Split(const std::string& str, const char delim) {
  if (str.empty()) {
    return {};
  }
  return absl::StrSplit(str, delim);
}

int GetNumElements(const TfLiteIntArray* dim_array) {
  int num_elements = 1;
  for (size_t i = 0; i < dim_array->size; i++) {
    num_elements *= dim_array->data[i];
  }
  return num_elements;
}

void FillRandomString(tflite::DynamicBuffer* buffer,
                      const TfLiteIntArray* dim_array,
                      const std::function<std::string()>& random_func) {
  int num_elements = GetNumElements(dim_array);
  for (int i = 0; i < num_elements; ++i) {
    auto str = random_func();
    buffer->AddString(str.data(), str.length());
  }
}

int FindLayerInfoIndex(std::vector<BenchmarkTfLiteModel::InputLayerInfo>* info,
                       const std::string& input_name,
                       const string& names_string) {
  for (int i = 0; i < info->size(); ++i) {
    if (info->at(i).name == input_name) {
      return i;
    }
  }
  TFLITE_LOG(FATAL) << "Cannot find the corresponding input_layer name("
                    << input_name << ") in --input_layer as " << names_string;
  return -1;
}

TfLiteStatus PopulateInputValueRanges(
    const std::string& names_string, const std::string& value_ranges_string,
    std::vector<BenchmarkTfLiteModel::InputLayerInfo>* info) {
  std::vector<std::string> value_ranges = Split(value_ranges_string, ':');
  for (const auto& val : value_ranges) {
    std::vector<std::string> name_range = Split(val, ',');
    if (name_range.size() != 3) {
      TFLITE_LOG(ERROR) << "Wrong input value range item specified: " << val;
      return kTfLiteError;
    }

    // Ensure the specific input layer name exists.
    int layer_info_idx = FindLayerInfoIndex(info, name_range[0], names_string);

    // Parse the range value.
    int low, high;
    bool has_low = absl::SimpleAtoi(name_range[1], &low);
    bool has_high = absl::SimpleAtoi(name_range[2], &high);
    if (!has_low || !has_high || low > high) {
      TFLITE_LOG(ERROR)
          << "Wrong low and high value of the input value range specified: "
          << val;
      return kTfLiteError;
    }
    info->at(layer_info_idx).has_value_range = true;
    info->at(layer_info_idx).low = low;
    info->at(layer_info_idx).high = high;
  }
  return kTfLiteOk;
}

TfLiteStatus PopulateInputValueFiles(
    const std::string& names_string, const std::string& value_files_string,
    std::vector<BenchmarkTfLiteModel::InputLayerInfo>* info) {
  std::vector<std::string> value_files = Split(value_files_string, ',');
  for (const auto& val : value_files) {
    std::pair<std::string, std::string> name_file_pair;
    TfLiteStatus status = SplitInputLayerNameAndValueFile(val, name_file_pair);
    if (status != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Wrong input value file item specified: " << val;
      TFLITE_LOG(ERROR) << status;
      return status;
    }

    // Ensure the specific input layer name exists.
    int layer_info_idx =
        FindLayerInfoIndex(info, name_file_pair.first, names_string);
    if (info->at(layer_info_idx).has_value_range) {
      TFLITE_LOG(WARN)
          << "The input_name:" << info->at(layer_info_idx).name
          << " appears both in input_layer_value_files and "
             "input_layer_value_range. The input_layer_value_range of the "
             "input_name will be ignored.";
    }
    info->at(layer_info_idx).input_file_path = name_file_pair.second;
  }
  return kTfLiteOk;
}

TfLiteStatus PopulateInputLayerInfo(
    const std::string& names_string, const std::string& shapes_string,
    const std::string& value_ranges_string,
    const std::string& value_files_string,
    std::vector<BenchmarkTfLiteModel::InputLayerInfo>* info) {
  info->clear();
  std::vector<std::string> names = Split(names_string, ',');
  std::vector<std::string> shapes = Split(shapes_string, ':');

  if (names.size() != shapes.size()) {
    TFLITE_LOG(ERROR)
        << "The number of items in --input_layer_shape (" << shapes_string
        << ", with " << shapes.size()
        << " items) must match the number of items in --input_layer ("
        << names_string << ", with " << names.size()
        << " items). For example --input_layer=input1,input2 "
           "--input_layer_shape=1,224,224,4:1,20";
    return kTfLiteError;
  }

  for (int i = 0; i < names.size(); ++i) {
    info->push_back(BenchmarkTfLiteModel::InputLayerInfo());
    BenchmarkTfLiteModel::InputLayerInfo& input = info->back();

    input.name = names[i];

    TFLITE_TOOLS_CHECK(util::SplitAndParse(shapes[i], ',', &input.shape))
        << "Incorrect size string specified: " << shapes[i];
    for (int dim : input.shape) {
      if (dim == -1) {
        TFLITE_LOG(ERROR)
            << "Any unknown sizes in the shapes (-1's) must be replaced"
            << " with the size you want to benchmark with.";
        return kTfLiteError;
      }
    }
  }

  // Populate input value range if it's specified.
  TF_LITE_ENSURE_STATUS(
      PopulateInputValueRanges(names_string, value_ranges_string, info));

  // Populate input value files if it's specified.
  TF_LITE_ENSURE_STATUS(
      PopulateInputValueFiles(names_string, value_files_string, info));

  return kTfLiteOk;
}

std::shared_ptr<profiling::ProfileSummaryFormatter>
CreateProfileSummaryFormatter(const std::string& output_mode) {
  if (output_mode == kOpProfilingOutputModeCsv) {
    return std::make_shared<profiling::ProfileSummaryCSVFormatter>();
  } else if (output_mode == kOpProfilingOutputModeProto) {
    return std::make_shared<profiling::ProfileSummaryProtoFormatter>();
  } else {
    return std::make_shared<profiling::ProfileSummaryDefaultFormatter>();
  }
}

}  // namespace

TfLiteStatus SplitInputLayerNameAndValueFile(
    const std::string& name_and_value_file,
    std::pair<std::string, std::string>& name_file_pair) {
  // 1. split the string by ':' and ignore escaped characters
  int delim_index = -1;
  for (int i = 0; i < name_and_value_file.length() - 1; ++i) {
    if (name_and_value_file[i] == ':') {
      if (name_and_value_file[i + 1] == ':') {
        ++i;
      } else {
        if (delim_index == -1) {
          delim_index = i;
        } else {
          TFLITE_LOG(ERROR)
              << name_and_value_file << " contains more than one delimiter.";
          return kTfLiteError;
        }
      }
    }
  }
  if (delim_index == -1) {
    TFLITE_LOG(ERROR) << name_and_value_file
                      << " doesn't contain any delimiter.";
    return kTfLiteError;
  }
  // 2. replace escaped "::" string to ":"
  name_file_pair.first = absl::StrReplaceAll(
      name_and_value_file.substr(0, delim_index), {{"::", ":"}});
  name_file_pair.second = absl::StrReplaceAll(
      name_and_value_file.substr(delim_index + 1), {{"::", ":"}});
  return kTfLiteOk;
}

std::pair<TfLiteStatus, std::unique_ptr<BenchmarkInterpreterRunner>>
BenchmarkInterpreterRunner::Create(tflite::Interpreter* const interpreter,
                                   std::string signature_key) {
  if (!signature_key.empty()) {
    const std::vector<const std::string*>& keys = interpreter->signature_keys();
    bool found = std::any_of(
        keys.begin(), keys.end(),
        [&signature_key](const auto& k) { return *k == signature_key; });

    if (keys.size() > 1 && (signature_key.empty() || !found)) {
      TFLITE_LOG(ERROR)
          << "Signature not specified or incorrect for graph with multiple "
             "signatures. Pass one of the following to the flag "
             "\"--signature_to_run_for\"";
      for (const std::string* k : keys) {
        TFLITE_LOG(ERROR) << " #> Signature key: " << *k;
      }
      return {kTfLiteError, nullptr};
    } else if (keys.size() == 1 && signature_key.empty()) {
      signature_key = *keys[0];
    }

    if (!signature_key.empty() && !keys.empty()) {
      TFLITE_LOG(INFO) << "Using signature: " << signature_key;
      auto signature_runner =
          interpreter->GetSignatureRunner(signature_key.c_str());
      if (signature_runner == nullptr) {
        return {kTfLiteError, nullptr};
      } else {
        int subgraph_index =
            interpreter->GetSubgraphIndexFromSignature(signature_key.c_str());

        return {kTfLiteOk, std::make_unique<BenchmarkInterpreterRunner>(
                               interpreter, signature_runner,
                               interpreter->subgraph(subgraph_index))};
      }
    }
  }
  return {kTfLiteOk, std::make_unique<BenchmarkInterpreterRunner>(
                         interpreter, nullptr, nullptr)};
}

TfLiteStatus BenchmarkInterpreterRunner::AllocateTensors() {
  if (signature_runner_ != nullptr) {
    return signature_runner_->AllocateTensors();
  } else {
    return interpreter_->AllocateTensors();
  }
}

TfLiteStatus BenchmarkInterpreterRunner::Invoke() {
  if (signature_runner_ != nullptr) {
    return signature_runner_->Invoke();
  } else {
    return interpreter_->Invoke();
  }
}

const std::vector<int>& BenchmarkInterpreterRunner::execution_plan() const {
  if (signature_runner_ != nullptr) {
    return subgraph_->execution_plan();
  } else {
    return interpreter_->execution_plan();
  }
}

const std::vector<int>& BenchmarkInterpreterRunner::inputs() const {
  if (signature_runner_ != nullptr) {
    return subgraph_->inputs();
  } else {
    return interpreter_->inputs();
  }
}

const std::vector<int>& BenchmarkInterpreterRunner::outputs() const {
  if (signature_runner_ != nullptr) {
    return subgraph_->outputs();
  } else {
    return interpreter_->outputs();
  }
}
TfLiteTensor* BenchmarkInterpreterRunner::tensor(int tensor_index) {
  if (signature_runner_ != nullptr) {
    return subgraph_->tensor(tensor_index);
  } else {
    return interpreter_->tensor(tensor_index);
  }
}

const std::pair<TfLiteNode, TfLiteRegistration>*
BenchmarkInterpreterRunner::node_and_registration(int node_index) const {
  if (signature_runner_ != nullptr) {
    return subgraph_->node_and_registration(node_index);
  } else {
    return interpreter_->node_and_registration(node_index);
  }
}

TfLiteStatus BenchmarkInterpreterRunner::ResizeInputTensor(
    int tensor_index, const std::vector<int>& new_size) {
  if (signature_runner_ != nullptr) {
    return subgraph_->ResizeInputTensor(tensor_index, new_size);
  } else {
    return interpreter_->ResizeInputTensor(tensor_index, new_size);
  }
}

BenchmarkParams BenchmarkTfLiteModel::DefaultParams() {
  BenchmarkParams default_params = BenchmarkModel::DefaultParams();
  default_params.AddParam("graph", BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("signature_to_run_for",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("list_signatures",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam("input_layer",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("input_layer_shape",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("input_layer_value_range",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("input_layer_value_files",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("allow_fp16", BenchmarkParam::Create<bool>(false));
  default_params.AddParam("require_full_delegation",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam(
      "enable_op_profiling",
      BenchmarkParam::Create<bool>(kOpProfilingEnabledDefault));
  default_params.AddParam(
      "op_profiling_output_mode",
      BenchmarkParam::Create<std::string>(kOpProfilingOutputModeStdout));
  default_params.AddParam("op_profiling_output_file",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("max_profiling_buffer_entries",
                          BenchmarkParam::Create<int32_t>(1024));
  default_params.AddParam("allow_dynamic_profiling_buffer_increase",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam("profiling_output_csv_file",
                          BenchmarkParam::Create<std::string>(""));

  default_params.AddParam("print_preinvoke_state",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam("print_postinvoke_state",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam("release_dynamic_tensors",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam("optimize_memory_for_large_tensors",
                          BenchmarkParam::Create<int32_t>(0));
  default_params.AddParam("disable_delegate_clustering",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam("enable_builtin_cast_constant_cache",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam("output_filepath",
                          BenchmarkParam::Create<std::string>(""));

  default_params.AddParam("tensor_name_display_length",
                          BenchmarkParam::Create<int32_t>(25));
  default_params.AddParam("tensor_type_display_length",
                          BenchmarkParam::Create<int32_t>(15));
  default_params.AddParam("alloc_type_display_length",
                          BenchmarkParam::Create<int32_t>(18));

  tools::ProvidedDelegateList delegate_providers(&default_params);
  delegate_providers.AddAllDelegateParams();

  return default_params;
}

BenchmarkTfLiteModel::BenchmarkTfLiteModel(BenchmarkParams params)
    : BenchmarkModel(std::move(params)),
      random_engine_(std::random_device()()) {
  AddListener(&log_output_);
}

void BenchmarkTfLiteModel::CleanUp() {
  // Free up any pre-allocated tensor data during PrepareInputData.
  inputs_data_.clear();
}

BenchmarkTfLiteModel::~BenchmarkTfLiteModel() {
  CleanUp();

  // Release the pointer to the interpreter_runner_ before the interpreter is
  // destroyed.
  interpreter_runner_.reset();

  // Destory the owned interpreter earlier than other objects (specially
  // 'owned_delegates_').
  interpreter_.reset();
}

std::vector<Flag> BenchmarkTfLiteModel::GetFlags() {
  std::vector<Flag> flags = BenchmarkModel::GetFlags();
  std::vector<Flag> specific_flags = {
      CreateFlag<std::string>("graph", &params_, "graph file name"),
      CreateFlag<std::string>("input_layer", &params_, "input layer names"),
      CreateFlag<std::string>("input_layer_shape", &params_,
                              "input layer shape"),
      CreateFlag<std::string>(
          "input_layer_value_range", &params_,
          "A map-like string representing value range for *integer* input "
          "layers. Each item is separated by ':', and the item value consists "
          "of input layer name and integer-only range values (both low and "
          "high are inclusive) separated by ',', e.g. input1,1,2:input2,0,254"),
      CreateFlag<std::string>(
          "input_layer_value_files", &params_,
          "A map-like string representing value file. Each item is separated "
          "by ',', and the item value consists "
          "of input layer name and value file path separated by ':', e.g. "
          "input1:file_path1,input2:file_path2. In case the input layer name "
          "contains ':' e.g. \"input:0\", escape it with \"\\:\". If the "
          "input_name appears both in input_layer_value_range and "
          "input_layer_value_files, input_layer_value_range of the input_name "
          "will be ignored. The file format is binary and it should be array "
          "format or null separated strings format."),
      CreateFlag<bool>("allow_fp16", &params_, "allow fp16"),
      CreateFlag<bool>("require_full_delegation", &params_,
                       "require delegate to run the entire graph"),
      CreateFlag<bool>("enable_op_profiling", &params_, "enable op profiling"),
      CreateFlag<std::string>(
          "op_profiling_output_mode", &params_,
          "Output mode for op profiling results. Supported values are: "
          "'stdout', 'csv' and 'proto'."),
      CreateFlag<std::string>("op_profiling_output_file", &params_,
                              "Output file for op profiling results."),
      CreateFlag<int32_t>("max_profiling_buffer_entries", &params_,
                          "max initial profiling buffer entries"),
      CreateFlag<bool>("allow_dynamic_profiling_buffer_increase", &params_,
                       "allow dynamic increase on profiling buffer entries"),
      CreateFlag<std::string>("profiling_output_csv_file", &params_,
                              "[DEPRECATED: Use op_profiling_output_file and "
                              "op_profiling_output_mode instead] File path to "
                              "export profile data as CSV, if not set "
                              "prints to stdout."),
      CreateFlag<bool>(
          "print_preinvoke_state", &params_,
          "print out the interpreter internals just before calling Invoke. The "
          "internals will include allocated memory size of each tensor etc."),
      CreateFlag<bool>(
          "print_postinvoke_state", &params_,
          "print out the interpreter internals just before benchmark completes "
          "(i.e. after all repeated Invoke calls complete). The internals will "
          "include allocated memory size of each tensor etc."),
      CreateFlag<bool>("release_dynamic_tensors", &params_,
                       "Ensure dynamic tensor's memory is released when they "
                       "are not used."),
      CreateFlag<int32_t>(
          "optimize_memory_for_large_tensors", &params_,
          "Optimize memory usage for large tensors with sacrificing latency."),
      CreateFlag<bool>("disable_delegate_clustering", &params_,
                       "Disable delegate clustering."),
      CreateFlag<bool>(
          "enable_builtin_cast_constant_cache", &params_,
          "Cache the output of the builtin cast operation when its input "
          "is a constant tensor."),
      CreateFlag<std::string>(
          "output_filepath", &params_,
          "File path to export outputs layer as binary data."),
      CreateFlag<int32_t>(
          "tensor_name_display_length", &params_,
          "The number of characters to show for the tensor's name when "
          "printing the interpeter's state, defaults to 25."),
      CreateFlag<int32_t>(
          "tensor_type_display_length", &params_,
          "The number of characters to show for the tensor's type when "
          "printing the interpeter's state, defaults to 15."),
      CreateFlag<int32_t>(
          "alloc_type_display_length", &params_,
          "The number of characters to show for the tensor's allocation type "
          "when printing the interpeter's state, defaults to 18."),
      CreateFlag<std::string>(
          "signature_to_run_for", &params_,
          "If the model contains multiple signatures, use this flag to specify "
          "the signature to benchmark. If multiple signatures are present and "
          "this flag is not specified, the benchmark will throw an error. If "
          "only one signature is present and this flag is not specified, the "
          "default signature will be used."),
      CreateFlag<bool>("list_signatures", &params_,
                       "Displays all signatures present in the model and then "
                       "terminates the program.")};

  flags.insert(flags.end(), specific_flags.begin(), specific_flags.end());

  tools::ProvidedDelegateList delegate_providers(&params_);
  delegate_providers.AppendCmdlineFlags(flags);

  return flags;
}

void BenchmarkTfLiteModel::LogParams() {
  BenchmarkModel::LogParams();
  const bool verbose = params_.Get<bool>("verbose");
  // Always log the value of --graph.
  LOG_BENCHMARK_PARAM(std::string, "graph", "Graph", /*verbose*/ true);
  LOG_BENCHMARK_PARAM(std::string, "signature_to_run_for", "Signature to run",
                      /*verbose*/ true);
  LOG_BENCHMARK_PARAM(bool, "list_signatures",
                      "List signatures from the provided model", false);
  LOG_BENCHMARK_PARAM(std::string, "input_layer", "Input layers", verbose);
  LOG_BENCHMARK_PARAM(std::string, "input_layer_shape", "Input shapes",
                      verbose);
  LOG_BENCHMARK_PARAM(std::string, "input_layer_value_range",
                      "Input value ranges", verbose);
  LOG_BENCHMARK_PARAM(std::string, "input_layer_value_files",
                      "Input value files", verbose);

  LOG_BENCHMARK_PARAM(bool, "allow_fp16", "Allow fp16", verbose);
  LOG_BENCHMARK_PARAM(bool, "require_full_delegation",
                      "Require full delegation", verbose);
  LOG_BENCHMARK_PARAM(bool, "enable_op_profiling", "Enable op profiling",
                      verbose);
  LOG_BENCHMARK_PARAM(std::string, "op_profiling_output_mode",
                      "Op profiling output mode.", verbose);
  LOG_BENCHMARK_PARAM(std::string, "op_profiling_output_file",
                      "Op profiling output file.", verbose);
  LOG_BENCHMARK_PARAM(int32_t, "max_profiling_buffer_entries",
                      "Max initial profiling buffer entries", verbose);
  LOG_BENCHMARK_PARAM(bool, "allow_dynamic_profiling_buffer_increase",
                      "Allow dynamic increase on profiling buffer entries",
                      verbose);
  LOG_BENCHMARK_PARAM(std::string, "profiling_output_csv_file",
                      "CSV File to export profiling data to", verbose);
  LOG_BENCHMARK_PARAM(bool, "print_preinvoke_state",
                      "Print pre-invoke interpreter state", verbose);
  LOG_BENCHMARK_PARAM(bool, "print_postinvoke_state",
                      "Print post-invoke interpreter state", verbose);
  LOG_BENCHMARK_PARAM(bool, "release_dynamic_tensors",
                      "Release dynamic tensor memory", verbose);
  LOG_BENCHMARK_PARAM(int32_t, "optimize_memory_for_large_tensors",
                      "Optimize memory usage for large tensors", verbose);
  LOG_BENCHMARK_PARAM(bool, "disable_delegate_clustering",
                      "Disable delegate clustering", verbose);
  LOG_BENCHMARK_PARAM(bool, "enable_builtin_cast_constant_cache",
                      "Constant CAST output cache", verbose);
  LOG_BENCHMARK_PARAM(std::string, "output_filepath",
                      "File path to export outputs layer to", verbose);
  LOG_BENCHMARK_PARAM(int32_t, "tensor_name_display_length",
                      "Tensor name display length", verbose);
  LOG_BENCHMARK_PARAM(int32_t, "tensor_type_display_length",
                      "Tensor type display length", verbose);
  LOG_BENCHMARK_PARAM(int32_t, "alloc_type_display_length",
                      "Tensor allocation type display length", verbose);

  for (const auto& delegate_provider :
       tools::GetRegisteredDelegateProviders()) {
    delegate_provider->LogParams(params_, verbose);
  }
}

TfLiteStatus BenchmarkTfLiteModel::ValidateParams() {
  TF_LITE_ENSURE_STATUS(BenchmarkModel::ValidateParams());

  if (params_.Get<std::string>("graph").empty()) {
    TFLITE_LOG(ERROR)
        << "Please specify the name of your TF Lite input file with --graph";
    return kTfLiteError;
  }

  if (params_.Get<bool>("enable_op_profiling")) {
    bool found =
        std::find(std::begin(kOpProfilingOutputModes),
                  std::end(kOpProfilingOutputModes),
                  params_.Get<std::string>("op_profiling_output_mode")) !=
        std::end(kOpProfilingOutputModes);

    if (!found) {
      TFLITE_LOG(ERROR) << "Output mode"
                        << params_.Get<std::string>("op_profiling_output_mode")
                        << " is not supported. Supported values are: 'stdout', "
                           "'csv' and 'proto'.";
      return kTfLiteError;
    }

    if (!params_.Get<std::string>("profiling_output_csv_file").empty()) {
      // Backward compatibility for profiling_output_csv_file.
      params_.Set<std::string>("op_profiling_output_mode",
                               kOpProfilingOutputModeCsv);
      params_.Set<std::string>(
          "op_profiling_output_file",
          params_.Get<std::string>("profiling_output_csv_file"));
    }
  }

  return PopulateInputLayerInfo(
      params_.Get<std::string>("input_layer"),
      params_.Get<std::string>("input_layer_shape"),
      params_.Get<std::string>("input_layer_value_range"),
      params_.Get<std::string>("input_layer_value_files"), &inputs_);
}

uint64_t BenchmarkTfLiteModel::ComputeInputBytes() {
  TFLITE_TOOLS_CHECK(interpreter_runner_);
  uint64_t total_input_bytes = 0;
  for (int input : interpreter_runner_->inputs()) {
    auto* t = interpreter_runner_->tensor(input);
    total_input_bytes += t->bytes;
  }
  return total_input_bytes;
}

int64_t BenchmarkTfLiteModel::MayGetModelFileSize() {
  std::string fd_or_graph_path = params_.Get<std::string>("graph");
  // Path can be one of the following:
  // 1) File descriptor path: path must be in the format of
  // "fd:%model_fd%:%model_offset%:%model_size%".
  // 2) File path: path to the model file.
  // Please see tensorflow/lite/tools/model_loader.h for more information.
  std::vector<absl::string_view> parts = absl::StrSplit(fd_or_graph_path, ':');
  if (!parts.empty() && parts[0] == "fd") {
    int64_t model_size = -1;
    if (parts.size() != 4 || !absl::SimpleAtoi(parts[3], &model_size)) {
      TFLITE_LOG(ERROR) << "Failed to parse model file size: "
                        << fd_or_graph_path;
    }
    return model_size;
  }
  std::ifstream in_file(fd_or_graph_path, std::ios::binary | std::ios::ate);
  return in_file.tellg();
}

InputTensorData BenchmarkTfLiteModel::LoadInputTensorData(
    const TfLiteTensor& t, const std::string& input_file_path) {
  std::ifstream value_file(input_file_path, std::ios::binary);
  if (!value_file.good()) {
    TFLITE_LOG(FATAL) << "Failed to read the input_layer_value_file:"
                      << input_file_path;
  }
  InputTensorData t_data;
  if (t.type == kTfLiteString) {
    t_data.data = VoidUniquePtr(
        static_cast<void*>(new tflite::DynamicBuffer()),
        [](void* ptr) { delete static_cast<DynamicBuffer*>(ptr); });
    if (input_file_path.size() > 3 &&
        input_file_path.substr(input_file_path.size() - 3) == ".pb") {
      // If input file is ".pb" file, read data as a binary.
      std::stringstream buffer;
      buffer << value_file.rdbuf();
      static_cast<DynamicBuffer*>(t_data.data.get())
          ->AddString(buffer.str().data(), buffer.str().length());
      TFLITE_LOG(INFO) << "Read " << buffer.str().length()
                       << " bytes data from " << input_file_path << ".";
    } else {
      // Read input as a text.
      std::string line;
      size_t num_line = 0;
      // Read the line with the delimiter '\0'.
      while (std::getline(value_file, line, '\0')) {
        num_line++;
        static_cast<DynamicBuffer*>(t_data.data.get())
            ->AddString(line.data(), line.length());
      }
      int num_elements = GetNumElements(t.dims);
      if (num_line != num_elements) {
        TFLITE_LOG(FATAL)
            << "The number of string in the input_layer_value_file("
            << input_file_path << ") is " << num_line << ". It should be "
            << num_elements << ".";
      }
    }
  } else {
    value_file.seekg(0, std::ios_base::end);
    if (value_file.tellg() != t.bytes) {
      TFLITE_LOG(FATAL) << "The size of " << input_file_path << " is "
                        << value_file.tellg() << " bytes. It should be "
                        << t.bytes << " bytes.";
    }
    t_data.bytes = t.bytes;
    t_data.data =
        VoidUniquePtr(static_cast<void*>(new char[t.bytes]),
                      [](void* ptr) { delete[] static_cast<char*>(ptr); });
    value_file.clear();
    value_file.seekg(0, std::ios_base::beg);
    value_file.read(static_cast<char*>(t_data.data.get()), t.bytes);
  }
  return t_data;
}

InputTensorData BenchmarkTfLiteModel::CreateRandomTensorData(
    const TfLiteTensor& t, const InputLayerInfo* layer_info) {
  float low_range = 0;
  float high_range = 0;
  if (layer_info && layer_info->has_value_range) {
    low_range = layer_info->low;
    high_range = layer_info->high;
  } else {
    utils::GetDataRangesForType(t.type, &low_range, &high_range);
  }
  return utils::CreateRandomTensorData(t, low_range, high_range);
}

TfLiteStatus BenchmarkTfLiteModel::PrepareInputData() {
  CleanUp();

  // Note the corresponding relation between 'runner_inputs' and
  // 'inputs_' (i.e. the specified input layer info) has been checked in
  // BenchmarkTfLiteModel::Init() before calling this function. So, we
  // simply use the corresponding input layer info to initialize the input
  // data value properly.
  const std::vector<int>& runner_inputs = interpreter_runner_->inputs();
  for (int i = 0; i < runner_inputs.size(); ++i) {
    int tensor_index = runner_inputs[i];
    const TfLiteTensor& t = *(interpreter_runner_->tensor(tensor_index));
    const InputLayerInfo* input_layer_info = nullptr;
    // Note that when input layer parameters (i.e. --input_layer,
    // --input_layer_shape) are not specified, inputs_ is empty.
    if (!inputs_.empty()) input_layer_info = &inputs_[i];

    InputTensorData t_data;
    if (input_layer_info && !input_layer_info->input_file_path.empty()) {
      t_data = LoadInputTensorData(t, input_layer_info->input_file_path);
    } else {
      t_data = CreateRandomTensorData(t, input_layer_info);
    }
    inputs_data_.push_back(std::move(t_data));
  }
  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::ResetInputsAndOutputs() {
  const std::vector<int>& runner_inputs = interpreter_runner_->inputs();
  // Set the values of the input tensors from inputs_data_.
  for (int j = 0; j < runner_inputs.size(); ++j) {
    int i = runner_inputs[j];
    TfLiteTensor* t = interpreter_runner_->tensor(i);
    if (t->type == kTfLiteString) {
      if (inputs_data_[j].data) {
        static_cast<DynamicBuffer*>(inputs_data_[j].data.get())
            ->WriteToTensor(t, /*new_shape=*/nullptr);
      } else {
        tflite::DynamicBuffer buffer;
        FillRandomString(&buffer, t->dims, []() {
          return "we're have some friends over saturday to hang out in "
                 "the "
                 "yard";
        });
        buffer.WriteToTensor(t, /*new_shape=*/nullptr);
      }
    } else {
      std::memcpy(t->data.raw, inputs_data_[j].data.get(),
                  inputs_data_[j].bytes);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::InitInterpreter() {
  auto resolver = GetOpResolver();
  const int32_t num_threads = params_.Get<int32_t>("num_threads");
  const bool use_caching = params_.Get<bool>("use_caching");

  InterpreterOptions options;
  options.SetEnsureDynamicTensorsAreReleased(
      params_.Get<bool>("release_dynamic_tensors"));
  options.OptimizeMemoryForLargeTensors(
      params_.Get<int32_t>("optimize_memory_for_large_tensors"));
  options.SetDisableDelegateClustering(
      params_.Get<bool>("disable_delegate_clustering"));
  options.SetCacheConstantCastOp(
      params_.Get<bool>("enable_builtin_cast_constant_cache"));

  tflite::InterpreterBuilder builder(*model_, *resolver, &options);
  if (builder.SetNumThreads(num_threads) != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to set thread number";
    return kTfLiteError;
  }

  builder(&interpreter_);
  if (!interpreter_) {
    TFLITE_LOG(ERROR) << "Failed to initialize the interpreter";
    return kTfLiteError;
  }
  // Manually enable caching behavior in TF Lite interpreter.
  if (use_caching) {
    external_context_ = std::make_unique<tflite::ExternalCpuBackendContext>();
    std::unique_ptr<tflite::CpuBackendContext> cpu_backend_context(
        new tflite::CpuBackendContext());
    cpu_backend_context->SetUseCaching(true);
    cpu_backend_context->SetMaxNumThreads(num_threads);
    external_context_->set_internal_backend_context(
        std::move(cpu_backend_context));
    interpreter_->SetExternalContext(kTfLiteCpuBackendContext,
                                     external_context_.get());
  }

  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::Init() {
  TF_LITE_ENSURE_STATUS(LoadModel());
  TF_LITE_ENSURE_STATUS(InitInterpreter());

  if (params_.Get<bool>("list_signatures")) {
    const std::vector<const std::string*>& keys =
        interpreter_->signature_keys();
    TFLITE_LOG(INFO) << "The Model contains " << keys.size()
                     << " signature key(s).";
    if (!keys.empty()) {
      TFLITE_LOG(INFO) << "They are listed below: ";
    }
    for (const std::string* key : keys) {
      TFLITE_LOG(INFO) << "-> Signature Key: " << *key;
    }
    return kTfLiteError;
  }

  // Install profilers if necessary right after interpreter is created so that
  // any memory allocations inside the TFLite runtime could be recorded if the
  // installed profiler profile memory usage information.

  // Adjust "max_profiling_buffer_entries" according to the loaded model.
  int total_nodes = 0;
  for (int i = 0; i < interpreter_->subgraphs_size(); ++i) {
    // subgraph(...) is non-const member method.
    total_nodes += static_cast<int>(interpreter_->subgraph(i)->nodes_size());
  }
  if (total_nodes > params_.Get<int32_t>("max_profiling_buffer_entries")) {
    constexpr int kProfilingBufferHeadrooms = 512;
    params_.Set<int32_t>("max_profiling_buffer_entries",
                         total_nodes + kProfilingBufferHeadrooms);
  }

  AddOwnedListener(MayCreateProfilingListener());
  AddOwnedListener(std::unique_ptr<BenchmarkListener>(
      new InterpreterStatePrinter(interpreter_.get())));

  interpreter_->SetAllowFp16PrecisionForFp32(params_.Get<bool>("allow_fp16"));

  std::pair<TfLiteStatus, std::unique_ptr<BenchmarkInterpreterRunner>>
      status_and_runner = BenchmarkInterpreterRunner::Create(
          interpreter_.get(), params_.Get<std::string>("signature_to_run_for"));

  TF_LITE_ENSURE_STATUS(status_and_runner.first);
  interpreter_runner_ = std::move(status_and_runner.second);

  const std::vector<int>& runner_inputs = interpreter_runner_->inputs();

  if (!inputs_.empty()) {
    TFLITE_TOOLS_CHECK_EQ(inputs_.size(), runner_inputs.size())
        << "Inputs mismatch: Model inputs #:" << inputs_.size()
        << " expected: " << runner_inputs.size();
  }

  // Check if the tensor names match, and log a warning if it doesn't.
  for (int j = 0; j < inputs_.size(); ++j) {
    const InputLayerInfo& input = inputs_[j];
    int i = runner_inputs[j];
    TfLiteTensor* t = interpreter_runner_->tensor(i);
    if (input.name != t->name) {
      TFLITE_LOG(WARN) << "Tensor # " << i << " is named " << t->name
                       << " but flags call it " << input.name;
    }

    if (t->type != kTfLiteString && input.shape.size() != t->dims->size) {
      TFLITE_LOG(ERROR) << "Input tensor #" << i << " should have "
                        << t->dims->size << " dimensions!";
      return kTfLiteError;
    }
  }

  // Resize all non-string tensors.
  for (int j = 0; j < inputs_.size(); ++j) {
    const InputLayerInfo& input = inputs_[j];
    int i = runner_inputs[j];
    TfLiteTensor* t = interpreter_runner_->tensor(i);
    if (t->type != kTfLiteString) {
      interpreter_runner_->ResizeInputTensor(i, input.shape);
    }
  }

  owned_delegates_.clear();

  // Contains all ids of TfLiteNodes that have been checked to see whether
  // it's delegated or not.
  std::unordered_set<int> checked_node_ids;
  tools::ProvidedDelegateList delegate_providers(&params_);
  auto created_delegates = delegate_providers.CreateAllRankedDelegates();
  TFLITE_MAY_LOG(INFO, (created_delegates.size() >= 2))
      << "Going to apply " << created_delegates.size()
      << " delegates one after another.";

  // If created_delegates is empty, 'require_full_delegation' flag will
  // not be checked, thus CPU fallback will happen. Adding check here to
  // avoid fallback in this situation.
  if (created_delegates.empty() &&
      params_.Get<bool>("require_full_delegation")) {
    TFLITE_LOG(ERROR) << "Disallowed CPU fallback detected.";
    return kTfLiteError;
  }
  for (auto& created_delegate : created_delegates) {
    const auto* delegate_provider = created_delegate.provider;
    TfLiteDelegate* delegate = created_delegate.delegate.get();
    TFLITE_TOOLS_CHECK(delegate != nullptr)
        << "The created delegate by the delegate provider should not be "
           "nullptr!";
    // The interpreter becomes dependent on the delegate once the delegate
    // is used, so the order of destruction must be interpreter first,
    // delegate later. Moving the delegate to a list of owned delegates to
    // guarantee that.
    owned_delegates_.emplace_back(std::move(created_delegate.delegate));
    if (interpreter_->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to apply " << delegate_provider->GetName()
                        << " delegate.";
      return kTfLiteError;
    } else {
      // Ideally, such delegate info should already be computed when the
      // delegate is being applied to the model graph.
      int num_delegated_kernels = 0;
      for (int i = 0; i < interpreter_runner_->execution_plan().size(); ++i) {
        int node_id = interpreter_runner_->execution_plan()[i];
        if (checked_node_ids.find(node_id) != checked_node_ids.end()) {
          continue;
        }
        const TfLiteNode& node =
            interpreter_runner_->node_and_registration(node_id)->first;

        // Note that the 'delegate' here could be an
        // ExternalDelegateWrapper object that wraps an actual external
        // delegate, in which case, 'node.delegate' will be different from
        // 'delegate' because 'node.delegate' refers to the actual
        // external delegate.
        if (node.delegate != nullptr) {
          num_delegated_kernels++;
          checked_node_ids.insert(node_id);
        }
      }
      bool fully_delegated =
          (num_delegated_kernels == 1 &&
           interpreter_runner_->execution_plan().size() == 1);

      if (params_.Get<bool>("require_full_delegation") && !fully_delegated) {
        TFLITE_LOG(ERROR) << "Disallowed CPU fallback detected.";
        return kTfLiteError;
      }
      if (fully_delegated) {
        TFLITE_LOG(INFO) << "Explicitly applied "
                         << delegate_provider->GetName()
                         << " delegate, and the model graph will be completely"
                         << " executed by the delegate.";
      } else if (num_delegated_kernels > 0) {
        TFLITE_LOG(INFO) << "Explicitly applied "
                         << delegate_provider->GetName()
                         << " delegate, and the model graph will be partially"
                         << " executed by the delegate w/ "
                         << num_delegated_kernels << " delegate kernels.";
      } else {
        TFLITE_LOG(INFO) << "Though " << delegate_provider->GetName()
                         << " delegate is explicitly applied, the model "
                            "graph will not be"
                         << " executed by the delegate.";
      }
    }
  }

  if (interpreter_runner_->AllocateTensors() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to allocate tensors!";
    return kTfLiteError;
  }

  AddOwnedListener(
      std::unique_ptr<BenchmarkListener>(new RuyProfileListener()));

  AddOwnedListener(std::unique_ptr<BenchmarkListener>(
      new OutputSaver(interpreter_runner_.get())));

  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::LoadModel() {
  std::string fd_or_graph_path = params_.Get<std::string>("graph");
  model_loader_ = tools::CreateModelLoaderFromPath(fd_or_graph_path);
  if (!model_loader_) {
    TFLITE_LOG(ERROR) << "Failed to initialize model loader with path "
                      << fd_or_graph_path;
    return kTfLiteError;
  }
  if (!model_loader_->Init()) {
    TFLITE_LOG(ERROR) << "Failed to load model " << fd_or_graph_path;
    return kTfLiteError;
  }
  model_ = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(
          model_loader_->GetModel()->allocation()->base()),
      model_loader_->GetModel()->allocation()->bytes());
  TFLITE_LOG(INFO) << "Loaded model " << fd_or_graph_path;
  return kTfLiteOk;
}

std::unique_ptr<tflite::OpResolver> BenchmarkTfLiteModel::GetOpResolver()
    const {
  tflite::ops::builtin::BuiltinOpResolver* resolver = nullptr;
  // When --use_xnnpack is explicitly set to false, skip applying the default
  // XNNPACK delegate in TfLite runtime so that the original execution path
  // based on the unmodified model graph is still exercised.
  if (params_.HasParam("use_xnnpack") &&
      params_.HasValueSet<bool>("use_xnnpack") &&
      !params_.Get<bool>("use_xnnpack")) {
    resolver =
        new tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates();
  } else {
    resolver = new tflite::ops::builtin::BuiltinOpResolver();
  }
  RegisterSelectedOps(resolver);
  return std::unique_ptr<tflite::OpResolver>(resolver);
}

std::unique_ptr<BenchmarkListener>
BenchmarkTfLiteModel::MayCreateProfilingListener() const {
  if (!params_.Get<bool>("enable_op_profiling")) return nullptr;

  return std::unique_ptr<BenchmarkListener>(new ProfilingListener(
      interpreter_.get(), params_.Get<int32_t>("max_profiling_buffer_entries"),
      params_.Get<bool>("allow_dynamic_profiling_buffer_increase"),
      params_.Get<std::string>("op_profiling_output_file"),
      CreateProfileSummaryFormatter(
          params_.Get<std::string>("op_profiling_output_mode"))));
}

TfLiteStatus BenchmarkTfLiteModel::RunImpl() {
  return interpreter_runner_->Invoke();
}

}  // namespace benchmark
}  // namespace tflite
