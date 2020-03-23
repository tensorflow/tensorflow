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

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/numbers.h"
#include "tensorflow/lite/experimental/ruy/profiler/profiler.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/profiling/platform_profiler.h"
#include "tensorflow/lite/profiling/profile_summary_formatter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/benchmark/delegate_provider.h"
#include "tensorflow/lite/tools/benchmark/logging.h"
#include "tensorflow/lite/tools/benchmark/profiling_listener.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);

// Version with Weak linker attribute doing nothing: if someone links this
// library with another definition of this function (presumably to actually
// register custom ops), that version will be used instead.
void ABSL_ATTRIBUTE_WEAK
RegisterSelectedOps(::tflite::MutableOpResolver* resolver) {}

namespace tflite {
namespace benchmark {
namespace {

// Backward compat with previous approach to enabling op profiling.
#if defined(TFLITE_PROFILING_ENABLED)
constexpr int kOpProfilingEnabledDefault = true;
#else
constexpr int kOpProfilingEnabledDefault = false;
#endif

// Dumps platform-wide tracing files via a platform-based profiler that's built
// upon platform tracing tools, like ATrace on Android etc.
class PlatformProfilingListener : public BenchmarkListener {
 public:
  explicit PlatformProfilingListener(Interpreter* interpreter) {
    TFLITE_BENCHMARK_CHECK(interpreter);
    platform_profiler_ = profiling::CreatePlatformProfiler();
    interpreter->SetProfiler(platform_profiler_.get());
  }

 private:
  std::unique_ptr<tflite::Profiler> platform_profiler_;
};

// Dumps ruy profiling events if the ruy profiler is enabled.
class RuyProfileListener : public BenchmarkListener {
 public:
  void OnBenchmarkStart(const BenchmarkParams& params) override;

  void OnBenchmarkEnd(const BenchmarkResults& results) override;

 private:
  std::unique_ptr<ruy::profiler::ScopeProfile> ruy_profile_;
};

void RuyProfileListener::OnBenchmarkStart(const BenchmarkParams& params) {
  ruy_profile_.reset(new ruy::profiler::ScopeProfile);
}

void RuyProfileListener::OnBenchmarkEnd(const BenchmarkResults& results) {
  ruy_profile_ = nullptr;
}

std::vector<std::string> Split(const std::string& str, const char delim) {
  std::vector<std::string> results;
  if (!util::SplitAndParse(str, delim, &results)) {
    results.clear();
  }
  return results;
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
    std::vector<std::string> name_file = Split(val, ':');
    if (name_file.size() != 2) {
      TFLITE_LOG(ERROR) << "Wrong input value file item specified: " << val;
      return kTfLiteError;
    }

    // Ensure the specific input layer name exists.
    int layer_info_idx = FindLayerInfoIndex(info, name_file[0], names_string);
    if (info->at(layer_info_idx).has_value_range) {
      TFLITE_LOG(WARN)
          << "The input_name:" << info->at(layer_info_idx).name
          << " appears both in input_layer_value_files and "
             "input_layer_value_range. The input_layer_value_range of the "
             "input_name will be ignored.";
    }
    info->at(layer_info_idx).input_file_path = name_file[1];
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
    TFLITE_LOG(ERROR) << "The number of items in"
                      << " --input_layer_shape (" << shapes_string << ", with "
                      << shapes.size() << " items)"
                      << " must match the number of items in"
                      << " --input_layer (" << names_string << ", with "
                      << names.size() << " items)."
                      << " For example --input_layer=input1,input2"
                      << " --input_layer_shape=1,224,224,4:1,20";
    return kTfLiteError;
  }

  for (int i = 0; i < names.size(); ++i) {
    info->push_back(BenchmarkTfLiteModel::InputLayerInfo());
    BenchmarkTfLiteModel::InputLayerInfo& input = info->back();

    input.name = names[i];

    TFLITE_BENCHMARK_CHECK(util::SplitAndParse(shapes[i], ',', &input.shape))
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
CreateProfileSummaryFormatter(bool format_as_csv) {
  return format_as_csv
             ? std::make_shared<profiling::ProfileSummaryCSVFormatter>()
             : std::make_shared<profiling::ProfileSummaryDefaultFormatter>();
}

}  // namespace

BenchmarkParams BenchmarkTfLiteModel::DefaultParams() {
  BenchmarkParams default_params = BenchmarkModel::DefaultParams();
  default_params.AddParam("graph", BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("input_layer",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("input_layer_shape",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("input_layer_value_range",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("input_layer_value_files",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("use_legacy_nnapi",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam("allow_fp16", BenchmarkParam::Create<bool>(false));
  default_params.AddParam("require_full_delegation",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam(
      "enable_op_profiling",
      BenchmarkParam::Create<bool>(kOpProfilingEnabledDefault));
  default_params.AddParam("max_profiling_buffer_entries",
                          BenchmarkParam::Create<int32_t>(1024));
  default_params.AddParam("profiling_output_csv_file",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("max_delegated_partitions",
                          BenchmarkParam::Create<int32_t>(0));
  default_params.AddParam("enable_platform_tracing",
                          BenchmarkParam::Create<bool>(false));

  for (const auto& delegate_util : GetRegisteredDelegateProviders()) {
    delegate_util->AddParams(&default_params);
  }

  return default_params;
}

BenchmarkTfLiteModel::BenchmarkTfLiteModel(BenchmarkParams params)
    : BenchmarkModel(std::move(params)),
      random_engine_(std::random_device()()) {}

void BenchmarkTfLiteModel::CleanUp() {
  // Free up any pre-allocated tensor data during PrepareInputData.
  inputs_data_.clear();
}

BenchmarkTfLiteModel::~BenchmarkTfLiteModel() { CleanUp(); }

std::vector<Flag> BenchmarkTfLiteModel::GetFlags() {
  std::vector<Flag> flags = BenchmarkTfLiteModel::BenchmarkModel::GetFlags();
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
          "input1:file_path1,input2:file_path2. If the input_name appears both "
          "in input_layer_value_range and input_layer_value_files, "
          "input_layer_value_range of the input_name will be ignored. The file "
          "format is binary and it should be array format or null separated "
          "strings format."),
      CreateFlag<bool>("use_legacy_nnapi", &params_, "use legacy nnapi api"),
      CreateFlag<bool>("allow_fp16", &params_, "allow fp16"),
      CreateFlag<bool>("require_full_delegation", &params_,
                       "require delegate to run the entire graph"),
      CreateFlag<bool>("enable_op_profiling", &params_, "enable op profiling"),
      CreateFlag<int32_t>("max_profiling_buffer_entries", &params_,
                          "max profiling buffer entries"),
      CreateFlag<std::string>(
          "profiling_output_csv_file", &params_,
          "File path to export profile data as CSV, if not set "
          "prints to stdout."),
      CreateFlag<int>("max_delegated_partitions", &params_,
                      "Max partitions to be delegated."),
      CreateFlag<bool>("enable_platform_tracing", &params_,
                       "enable platform-wide tracing, only meaningful when "
                       "--enable_op_profiling is set to true.")};

  flags.insert(flags.end(), specific_flags.begin(), specific_flags.end());

  for (const auto& delegate_util : GetRegisteredDelegateProviders()) {
    auto delegate_flags = delegate_util->CreateFlags(&params_);
    flags.insert(flags.end(), delegate_flags.begin(), delegate_flags.end());
  }

  return flags;
}

void BenchmarkTfLiteModel::LogParams() {
  BenchmarkModel::LogParams();
  TFLITE_LOG(INFO) << "Graph: [" << params_.Get<std::string>("graph") << "]";
  TFLITE_LOG(INFO) << "Input layers: ["
                   << params_.Get<std::string>("input_layer") << "]";
  TFLITE_LOG(INFO) << "Input shapes: ["
                   << params_.Get<std::string>("input_layer_shape") << "]";
  TFLITE_LOG(INFO) << "Input value ranges: ["
                   << params_.Get<std::string>("input_layer_value_range")
                   << "]";
  TFLITE_LOG(INFO) << "Input layer values files: ["
                   << params_.Get<std::string>("input_layer_value_files")
                   << "]";
#if defined(__ANDROID__)
  TFLITE_LOG(INFO) << "Use legacy nnapi : ["
                   << params_.Get<bool>("use_legacy_nnapi") << "]";
#endif
  TFLITE_LOG(INFO) << "Allow fp16 : [" << params_.Get<bool>("allow_fp16")
                   << "]";
  TFLITE_LOG(INFO) << "Require full delegation : ["
                   << params_.Get<bool>("require_full_delegation") << "]";
  TFLITE_LOG(INFO) << "Enable op profiling: ["
                   << params_.Get<bool>("enable_op_profiling") << "]";
  TFLITE_LOG(INFO) << "Max profiling buffer entries: ["
                   << params_.Get<int32_t>("max_profiling_buffer_entries")
                   << "]";
  TFLITE_LOG(INFO) << "CSV File to export profiling data to: ["
                   << params_.Get<std::string>("profiling_output_csv_file")
                   << "]";
  TFLITE_LOG(INFO) << "Max number of delegated partitions : ["
                   << params_.Get<int32_t>("max_delegated_partitions") << "]";
  TFLITE_LOG(INFO) << "Enable platform-wide tracing: ["
                   << params_.Get<bool>("enable_platform_tracing") << "]";

  for (const auto& delegate_util : GetRegisteredDelegateProviders()) {
    delegate_util->LogParams(params_);
  }
}

TfLiteStatus BenchmarkTfLiteModel::ValidateParams() {
  if (params_.Get<std::string>("graph").empty()) {
    TFLITE_LOG(ERROR)
        << "Please specify the name of your TF Lite input file with --graph";
    return kTfLiteError;
  }

  return PopulateInputLayerInfo(
      params_.Get<std::string>("input_layer"),
      params_.Get<std::string>("input_layer_shape"),
      params_.Get<std::string>("input_layer_value_range"),
      params_.Get<std::string>("input_layer_value_files"), &inputs_);
}

uint64_t BenchmarkTfLiteModel::ComputeInputBytes() {
  TFLITE_BENCHMARK_CHECK(interpreter_);
  uint64_t total_input_bytes = 0;
  for (int input : interpreter_->inputs()) {
    auto* t = interpreter_->tensor(input);
    total_input_bytes += t->bytes;
  }
  return total_input_bytes;
}

int64_t BenchmarkTfLiteModel::MayGetModelFileSize() {
  std::ifstream in_file(params_.Get<std::string>("graph"),
                        std::ios::binary | std::ios::ate);
  return in_file.tellg();
}

BenchmarkTfLiteModel::InputTensorData BenchmarkTfLiteModel::LoadInputTensorData(
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
      TFLITE_LOG(FATAL) << "The number of string in the input_layer_value_file("
                        << input_file_path << ") is " << num_line
                        << ". It should be " << num_elements << ".";
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

BenchmarkTfLiteModel::InputTensorData
BenchmarkTfLiteModel::CreateRandomTensorData(const TfLiteTensor& t,
                                             const InputLayerInfo* layer_info) {
  bool has_value_range = false;
  int low_range = 0;
  int high_range = 0;
  if (layer_info) {
    has_value_range = layer_info->has_value_range;
    low_range = layer_info->low;
    high_range = layer_info->high;
  }
  int num_elements = GetNumElements(t.dims);
  switch (t.type) {
    case kTfLiteFloat32: {
      return CreateInputTensorData<float>(
          num_elements, std::uniform_real_distribution<float>(-0.5f, 0.5f));
    }
    case kTfLiteFloat16: {
      // TODO(b/138843274): Remove this preprocessor guard when bug is fixed.
#if TFLITE_ENABLE_FP16_CPU_BENCHMARKS
#if __GNUC__ && \
    (__clang__ || __ARM_FP16_FORMAT_IEEE || __ARM_FP16_FORMAT_ALTERNATIVE)
      // __fp16 is available on Clang or when __ARM_FP16_FORMAT_* is defined.
      return CreateInputTensorData<__fp16>(
          num_elements, std::uniform_real_distribution<float>(-0.5f, 0.5f));
#else
      TFLITE_LOG(FATAL) << "Don't know how to populate tensor " << t->name
                        << " of type FLOAT16 on this platform.";
#endif
#else
      // You need to build with -DTFLITE_ENABLE_FP16_CPU_BENCHMARKS=1 using a
      // compiler that supports __fp16 type. Note: when using Clang and *not*
      // linking with compiler-rt, a definition of __gnu_h2f_ieee and
      // __gnu_f2h_ieee must be supplied.
      TFLITE_LOG(FATAL) << "Populating the tensor " << t.name
                        << " of type FLOAT16 is disabled.";
#endif  // TFLITE_ENABLE_FP16_CPU_BENCHMARKS
      break;
    }
    case kTfLiteInt64: {
      int low = has_value_range ? low_range : 0;
      int high = has_value_range ? high_range : 99;
      return CreateInputTensorData<int64_t>(
          num_elements, std::uniform_int_distribution<int64_t>(low, high));
    }
    case kTfLiteInt32: {
      int low = has_value_range ? low_range : 0;
      int high = has_value_range ? high_range : 99;
      return CreateInputTensorData<int32_t>(
          num_elements, std::uniform_int_distribution<int32_t>(low, high));
    }
    case kTfLiteInt16: {
      int low = has_value_range ? low_range : 0;
      int high = has_value_range ? high_range : 99;
      return CreateInputTensorData<int16_t>(
          num_elements, std::uniform_int_distribution<int16_t>(low, high));
    }
    case kTfLiteUInt8: {
      int low = has_value_range ? low_range : 0;
      int high = has_value_range ? high_range : 254;
      // std::uniform_int_distribution is specified not to support char types.
      return CreateInputTensorData<uint8_t>(
          num_elements, std::uniform_int_distribution<uint32_t>(low, high));
    }
    case kTfLiteInt8: {
      int low = has_value_range ? low_range : -127;
      int high = has_value_range ? high_range : 127;
      // std::uniform_int_distribution is specified not to support char types.
      return CreateInputTensorData<int8_t>(
          num_elements, std::uniform_int_distribution<int32_t>(low, high));
    }
    case kTfLiteString: {
      // TODO(haoliang): No need to cache string tensors right now.
      break;
    }
    default: {
      TFLITE_LOG(FATAL) << "Don't know how to populate tensor " << t.name
                        << " of type " << t.type;
    }
  }
  return InputTensorData();
}

TfLiteStatus BenchmarkTfLiteModel::PrepareInputData() {
  CleanUp();

  // Note the corresponding relation between 'interpreter_inputs' and 'inputs_'
  // (i.e. the specified input layer info) has been checked in
  // BenchmarkTfLiteModel::Init() before calling this function. So, we simply
  // use the corresponding input layer info to initializethe input data value
  // properly.
  auto interpreter_inputs = interpreter_->inputs();
  for (int i = 0; i < interpreter_inputs.size(); ++i) {
    int tensor_index = interpreter_inputs[i];
    const TfLiteTensor& t = *(interpreter_->tensor(tensor_index));
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
  auto interpreter_inputs = interpreter_->inputs();
  // Set the values of the input tensors from inputs_data_.
  for (int j = 0; j < interpreter_inputs.size(); ++j) {
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    if (t->type == kTfLiteString) {
      if (inputs_data_[j].data) {
        static_cast<DynamicBuffer*>(inputs_data_[j].data.get())
            ->WriteToTensor(t, /*new_shape=*/nullptr);
      } else {
        tflite::DynamicBuffer buffer;
        FillRandomString(&buffer, t->dims, []() {
          return "we're have some friends over saturday to hang out in the "
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

TfLiteStatus BenchmarkTfLiteModel::Init() {
  TF_LITE_ENSURE_STATUS(LoadModel());

  auto resolver = GetOpResolver();

  const int32_t num_threads = params_.Get<int32_t>("num_threads");
  tflite::InterpreterBuilder(*model_, *resolver)(&interpreter_, num_threads);
  if (!interpreter_) {
    TFLITE_LOG(ERROR) << "Failed to construct interpreter";
    return kTfLiteError;
  }

  // Install profilers if necessary right after interpreter is created so that
  // any memory allocations inside the TFLite runtime could be recorded if the
  // installed profiler profile memory usage information.
  profiling_listener_ = MayCreateProfilingListener();
  if (profiling_listener_) AddListener(profiling_listener_.get());

  interpreter_->UseNNAPI(params_.Get<bool>("use_legacy_nnapi"));
  interpreter_->SetAllowFp16PrecisionForFp32(params_.Get<bool>("allow_fp16"));

  owned_delegates_.clear();
  for (const auto& delegate_provider : GetRegisteredDelegateProviders()) {
    auto delegate = delegate_provider->CreateTfLiteDelegate(params_);
    // It's possible that a delegate of certain type won't be created as
    // user-specified benchmark params tells not to.
    if (delegate == nullptr) continue;
    if (interpreter_->ModifyGraphWithDelegate(delegate.get()) != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to apply " << delegate_provider->GetName()
                        << " delegate.";
      return kTfLiteError;
    } else {
      bool fully_delegated = true;
      if (interpreter_->execution_plan().size() != 1) {
        fully_delegated = false;
      } else {
        int first_node_id = interpreter_->execution_plan()[0];
        const TfLiteNode first_node =
            interpreter_->node_and_registration(first_node_id)->first;
        if (delegate.get() != first_node.delegate) {
          fully_delegated = false;
        }
      }
      if (params_.Get<bool>("require_full_delegation") && !fully_delegated) {
        TFLITE_LOG(ERROR) << "Disallowed CPU fallback detected.";
        return kTfLiteError;
      }
      const std::string delegate_status =
          fully_delegated ? "completely" : "partially";
      TFLITE_LOG(INFO) << "Applied " << delegate_provider->GetName()
                       << " delegate, and the model graph will be "
                       << delegate_status << " executed w/ the delegate.";
    }
    owned_delegates_.emplace_back(std::move(delegate));
  }

  auto interpreter_inputs = interpreter_->inputs();

  if (!inputs_.empty()) {
    TFLITE_BENCHMARK_CHECK_EQ(inputs_.size(), interpreter_inputs.size())
        << "Inputs mismatch: Model inputs #:" << inputs_.size()
        << " expected: " << interpreter_inputs.size();
  }

  // Check if the tensor names match, and log a warning if it doesn't.
  // TODO(ycling): Consider to make this an error again when the new converter
  // create tensors with consistent naming.
  for (int j = 0; j < inputs_.size(); ++j) {
    const InputLayerInfo& input = inputs_[j];
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    if (input.name != t->name) {
      TFLITE_LOG(WARN) << "Tensor # " << i << " is named " << t->name
                       << " but flags call it " << input.name;
    }
  }

  // Resize all non-string tensors.
  for (int j = 0; j < inputs_.size(); ++j) {
    const InputLayerInfo& input = inputs_[j];
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    if (t->type != kTfLiteString) {
      interpreter_->ResizeInputTensor(i, input.shape);
    }
  }

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to allocate tensors!";
    return kTfLiteError;
  }

  ruy_profiling_listener_.reset(new RuyProfileListener());
  AddListener(ruy_profiling_listener_.get());

  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::LoadModel() {
  std::string graph = params_.Get<std::string>("graph");
  model_ = tflite::FlatBufferModel::BuildFromFile(graph.c_str());
  if (!model_) {
    TFLITE_LOG(ERROR) << "Failed to mmap model " << graph;
    return kTfLiteError;
  }
  TFLITE_LOG(INFO) << "Loaded model " << graph;
  return kTfLiteOk;
}

std::unique_ptr<tflite::OpResolver> BenchmarkTfLiteModel::GetOpResolver()
    const {
  auto resolver = new tflite::ops::builtin::BuiltinOpResolver();
  RegisterSelectedOps(resolver);
  return std::unique_ptr<tflite::OpResolver>(resolver);
}

std::unique_ptr<BenchmarkListener>
BenchmarkTfLiteModel::MayCreateProfilingListener() const {
  if (!params_.Get<bool>("enable_op_profiling")) return nullptr;

  if (params_.Get<bool>("enable_platform_tracing")) {
    return std::unique_ptr<BenchmarkListener>(
        new PlatformProfilingListener(interpreter_.get()));
  }

  return std::unique_ptr<BenchmarkListener>(new ProfilingListener(
      interpreter_.get(), params_.Get<int32_t>("max_profiling_buffer_entries"),
      params_.Get<std::string>("profiling_output_csv_file"),
      CreateProfileSummaryFormatter(
          !params_.Get<std::string>("profiling_output_csv_file").empty())));
}

TfLiteStatus BenchmarkTfLiteModel::RunImpl() { return interpreter_->Invoke(); }

}  // namespace benchmark
}  // namespace tflite
