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
#include "tensorflow/lite/profiling/buffered_profiler.h"
#include "tensorflow/lite/profiling/profile_summarizer.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/benchmark/delegate_provider.h"
#include "tensorflow/lite/tools/benchmark/logging.h"
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

// Dumps profiling events if profiling is enabled.
class ProfilingListener : public BenchmarkListener {
 public:
  ProfilingListener(Interpreter* interpreter, uint32_t max_num_entries)
      : interpreter_(interpreter), profiler_(max_num_entries) {
    TFLITE_BENCHMARK_CHECK(interpreter);
    interpreter_->SetProfiler(&profiler_);

    // We start profiling here in order to catch events that are recorded during
    // the benchmark run preparation stage where TFLite interpreter is
    // initialized and model graph is prepared.
    profiler_.Reset();
    profiler_.StartProfiling();
  }

  void OnBenchmarkStart(const BenchmarkParams& params) override;

  void OnSingleRunStart(RunType run_type) override;

  void OnSingleRunEnd() override;

  void OnBenchmarkEnd(const BenchmarkResults& results) override;

 private:
  Interpreter* interpreter_;
  profiling::BufferedProfiler profiler_;
  profiling::ProfileSummarizer run_summarizer_;
  profiling::ProfileSummarizer init_summarizer_;
};

// Dumps ruy profiling events if the ruy profiler is enabled.
class RuyProfileListener : public BenchmarkListener {
 public:
  void OnBenchmarkStart(const BenchmarkParams& params) override;

  void OnBenchmarkEnd(const BenchmarkResults& results) override;

 private:
  std::unique_ptr<ruy::profiler::ScopeProfile> ruy_profile_;
};

void ProfilingListener::OnBenchmarkStart(const BenchmarkParams& params) {
  // At this point, we have completed the prepration for benchmark runs
  // including TFLite interpreter initialization etc. So we are going to process
  // profiling events recorded during this stage.
  profiler_.StopProfiling();
  auto profile_events = profiler_.GetProfileEvents();
  init_summarizer_.ProcessProfiles(profile_events, *interpreter_);
  profiler_.Reset();
}

void ProfilingListener::OnSingleRunStart(RunType run_type) {
  if (run_type == REGULAR) {
    profiler_.Reset();
    profiler_.StartProfiling();
  }
}

void ProfilingListener::OnBenchmarkEnd(const BenchmarkResults& results) {
  if (init_summarizer_.HasProfiles()) {
    TFLITE_LOG(INFO) << "Profiling Info for Benchmark Initialization:";
    TFLITE_LOG(INFO) << init_summarizer_.GetOutputString();
  }
  if (run_summarizer_.HasProfiles()) {
    TFLITE_LOG(INFO)
        << "Operator-wise Profiling Info for Regular Benchmark Runs:";
    TFLITE_LOG(INFO) << run_summarizer_.GetOutputString();
  }
}

void ProfilingListener::OnSingleRunEnd() {
  profiler_.StopProfiling();
  auto profile_events = profiler_.GetProfileEvents();
  run_summarizer_.ProcessProfiles(profile_events, *interpreter_);
}

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

void FillRandomString(tflite::DynamicBuffer* buffer,
                      const std::vector<int>& sizes,
                      const std::function<std::string()>& random_func) {
  int num_elements = 1;
  for (int dim : sizes) {
    num_elements *= dim;
  }
  for (int i = 0; i < num_elements; ++i) {
    auto str = random_func();
    buffer->AddString(str.data(), str.length());
  }
}

TfLiteStatus PopulateInputLayerInfo(
    const std::string& names_string, const std::string& shapes_string,
    const std::string& value_ranges_string,
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
  std::vector<std::string> value_ranges = Split(value_ranges_string, ':');
  for (const auto& val : value_ranges) {
    std::vector<std::string> name_range = Split(val, ',');
    if (name_range.size() != 3) {
      TFLITE_LOG(FATAL) << "Wrong input value range item specified: " << val;
    }

    // Ensure the specific input layer name exists.
    const std::string& input_name = name_range[0];
    int layer_info_idx = -1;
    for (int i = 0; i < info->size(); ++i) {
      if (info->at(i).name == input_name) {
        layer_info_idx = i;
        break;
      }
    }
    TFLITE_BENCHMARK_CHECK((layer_info_idx != -1))
        << "Cannot find the corresponding input_layer name(" << input_name
        << ") in --input_layer as " << names_string;

    // Parse the range value.
    int low, high;
    bool has_low = absl::SimpleAtoi(name_range[1], &low);
    bool has_high = absl::SimpleAtoi(name_range[2], &high);
    if (!has_low || !has_high || low > high) {
      TFLITE_LOG(FATAL)
          << "Wrong low and high value of the input value range specified: "
          << val;
    }

    info->at(layer_info_idx).has_value_range = true;
    info->at(layer_info_idx).low = low;
    info->at(layer_info_idx).high = high;
  }

  return kTfLiteOk;
}

std::vector<int> TfLiteIntArrayToVector(const TfLiteIntArray* int_array) {
  std::vector<int> values;
  values.reserve(int_array->size);
  for (size_t i = 0; i < int_array->size; i++) {
    values.push_back(int_array->data[i]);
  }
  return values;
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
      CreateFlag<bool>("use_legacy_nnapi", &params_, "use legacy nnapi api"),
      CreateFlag<bool>("allow_fp16", &params_, "allow fp16"),
      CreateFlag<bool>("require_full_delegation", &params_,
                       "require delegate to run the entire graph"),
      CreateFlag<bool>("enable_op_profiling", &params_, "enable op profiling"),
      CreateFlag<int32_t>("max_profiling_buffer_entries", &params_,
                          "max profiling buffer entries")};

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
      params_.Get<std::string>("input_layer_value_range"), &inputs_);
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

TfLiteStatus BenchmarkTfLiteModel::PrepareInputData() {
  auto interpreter_inputs = interpreter_->inputs();
  const size_t input_size = interpreter_inputs.size();
  CleanUp();

  // Note the corresponding relation between 'interpreter_inputs' and 'inputs_'
  // (i.e. the specified input layer info) has been checked in
  // BenchmarkTfLiteModel::Init() before calling this function. So, we simply
  // use the corresponding input layer info to initializethe input data value
  // properly.

  for (int j = 0; j < input_size; ++j) {
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    bool has_value_range = false;
    int low_range = 0;
    int high_range = 0;
    // For tflite files that are benchmarked without input layer parameters
    // inputs_ is empty.
    if (!inputs_.empty()) {
      has_value_range = inputs_[j].has_value_range;
      low_range = inputs_[j].low;
      high_range = inputs_[j].high;
    }
    std::vector<int> sizes = TfLiteIntArrayToVector(t->dims);
    int num_elements = 1;
    for (int i = 0; i < sizes.size(); ++i) {
      num_elements *= sizes[i];
    }
    InputTensorData t_data;
    if (t->type == kTfLiteFloat32) {
      t_data = CreateInputTensorData<float>(
          num_elements, std::uniform_real_distribution<float>(-0.5f, 0.5f));
    } else if (t->type == kTfLiteFloat16) {
// TODO(b/138843274): Remove this preprocessor guard when bug is fixed.
#if TFLITE_ENABLE_FP16_CPU_BENCHMARKS
#if __GNUC__ && \
    (__clang__ || __ARM_FP16_FORMAT_IEEE || __ARM_FP16_FORMAT_ALTERNATIVE)
      // __fp16 is available on Clang or when __ARM_FP16_FORMAT_* is defined.
      t_data = CreateInputTensorData<__fp16>(
          num_elements, std::uniform_real_distribution<float>(-0.5f, 0.5f));
#else
      TFLITE_LOG(FATAL) << "Don't know how to populate tensor " << t->name
                        << " of type FLOAT16 on this platform.";
#endif
#else
      // You need to build with -DTFLITE_ENABLE_FP16_CPU_BENCHMARKS=1 using a
      // compiler that supports __fp16 type. Note: when using Clang and *not*
      // linking with compiler-rt, a defintion of __gnu_h2f_ieee and
      // __gnu_f2h_ieee must be supplied.
      TFLITE_LOG(FATAL) << "Populating the tensor " << t->name
                        << " of type FLOAT16 is disabled.";
#endif  // TFLITE_ENABLE_FP16_CPU_BENCHMARKS
    } else if (t->type == kTfLiteInt64) {
      int low = has_value_range ? low_range : 0;
      int high = has_value_range ? high_range : 99;
      t_data = CreateInputTensorData<int64_t>(
          num_elements, std::uniform_int_distribution<int64_t>(low, high));
    } else if (t->type == kTfLiteInt32) {
      int low = has_value_range ? low_range : 0;
      int high = has_value_range ? high_range : 99;
      t_data = CreateInputTensorData<int32_t>(
          num_elements, std::uniform_int_distribution<int32_t>(low, high));
    } else if (t->type == kTfLiteInt16) {
      int low = has_value_range ? low_range : 0;
      int high = has_value_range ? high_range : 99;
      t_data = CreateInputTensorData<int16_t>(
          num_elements, std::uniform_int_distribution<int16_t>(low, high));
    } else if (t->type == kTfLiteUInt8) {
      int low = has_value_range ? low_range : 0;
      int high = has_value_range ? high_range : 254;
      // std::uniform_int_distribution is specified not to support char types.
      t_data = CreateInputTensorData<uint8_t>(
          num_elements, std::uniform_int_distribution<uint32_t>(low, high));
    } else if (t->type == kTfLiteInt8) {
      int low = has_value_range ? low_range : -127;
      int high = has_value_range ? high_range : 127;
      // std::uniform_int_distribution is specified not to support char types.
      t_data = CreateInputTensorData<int8_t>(
          num_elements, std::uniform_int_distribution<int32_t>(low, high));
    } else if (t->type == kTfLiteString) {
      // TODO(haoliang): No need to cache string tensors right now.
    } else {
      TFLITE_LOG(ERROR) << "Don't know how to populate tensor " << t->name
                        << " of type " << t->type;
      return kTfLiteError;
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
      tflite::DynamicBuffer buffer;
      std::vector<int> sizes = TfLiteIntArrayToVector(t->dims);
      FillRandomString(&buffer, sizes, []() {
        return "we're have some friends over saturday to hang out in the yard";
      });
      buffer.WriteToTensor(t, /*new_shape=*/nullptr);
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

  delegates_ = GetDelegates();
  for (const auto& delegate : delegates_) {
    if (interpreter_->ModifyGraphWithDelegate(delegate.second.get()) !=
        kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to apply " << delegate.first << " delegate.";
      return kTfLiteError;
    } else {
      if (params_.Get<bool>("require_full_delegation")) {
        bool fully_delegated = true;
        if (interpreter_->execution_plan().size() != 1) {
          fully_delegated = false;
        } else {
          int first_node_id = interpreter_->execution_plan()[0];
          const TfLiteNode first_node =
              interpreter_->node_and_registration(first_node_id)->first;
          if (delegate.second.get() != first_node.delegate) {
            fully_delegated = false;
          }
        }

        if (!fully_delegated) {
          TFLITE_LOG(ERROR) << "Disallowed CPU fallback detected.";
          return kTfLiteError;
        }
      }

      TFLITE_LOG(INFO) << "Applied " << delegate.first << " delegate.";
    }
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

BenchmarkTfLiteModel::TfLiteDelegatePtrMap BenchmarkTfLiteModel::GetDelegates()
    const {
  TfLiteDelegatePtrMap delegates;
  for (const auto& delegate_util : GetRegisteredDelegateProviders()) {
    auto delegate = delegate_util->CreateTfLiteDelegate(params_);
    if (delegate != nullptr) {
      delegates.emplace(delegate_util->GetName(), std::move(delegate));
    }
  }

  return delegates;
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
  return std::unique_ptr<BenchmarkListener>(new ProfilingListener(
      interpreter_.get(),
      params_.Get<int32_t>("max_profiling_buffer_entries")));
}

TfLiteStatus BenchmarkTfLiteModel::RunImpl() { return interpreter_->Invoke(); }

}  // namespace benchmark
}  // namespace tflite
