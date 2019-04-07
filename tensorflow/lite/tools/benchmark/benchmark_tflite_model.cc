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
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/benchmark/logging.h"

#ifdef GEMMLOWP_PROFILING
#include "gemmlowp/profiling/profiler.h"
#endif

#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif

#ifdef TFLITE_CUSTOM_OPS_HEADER
void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);
#endif

namespace tflite {
namespace benchmark {
namespace {

#if defined(__ANDROID__)
Interpreter::TfLiteDelegatePtr CreateGPUDelegate(
    tflite::FlatBufferModel* model) {
  TfLiteGpuDelegateOptions options;
  options.metadata = TfLiteGpuDelegateGetModelMetadata(model->GetModel());
  options.compile_options.precision_loss_allowed = 1;
  options.compile_options.preferred_gl_object_type =
      TFLITE_GL_OBJECT_TYPE_FASTEST;
  options.compile_options.dynamic_batch_enabled = 0;
  return Interpreter::TfLiteDelegatePtr(TfLiteGpuDelegateCreate(&options),
                                        &TfLiteGpuDelegateDelete);
}

Interpreter::TfLiteDelegatePtr CreateNNAPIDelegate() {
  return Interpreter::TfLiteDelegatePtr(
      NnApiDelegate(),
      // NnApiDelegate() returns a singleton, so provide a no-op deleter.
      [](TfLiteDelegate*) {});
}

#endif  // defined(__ANDROID__)

}  // namespace

void ProfilingListener::SetInterpreter(tflite::Interpreter* interpreter) {
  TFLITE_BENCHMARK_CHECK(interpreter);
  interpreter_ = interpreter;
  interpreter_->SetProfiler(&profiler_);
}

void ProfilingListener::OnSingleRunStart(RunType run_type) {
  if (run_type == REGULAR) {
    profiler_.Reset();
    profiler_.StartProfiling();
  }
}

void ProfilingListener::OnBenchmarkEnd(const BenchmarkResults& results) {
  if (has_profiles_) {
    TFLITE_LOG(INFO) << summarizer_.GetOutputString();
  }
}

void ProfilingListener::OnSingleRunEnd() {
  profiler_.StopProfiling();
  auto profile_events = profiler_.GetProfileEvents();
  has_profiles_ = !profile_events.empty();
  summarizer_.ProcessProfiles(profile_events, *interpreter_);
}

void GemmlowpProfilingListener::OnBenchmarkStart(
    const BenchmarkParams& params) {
#ifdef GEMMLOWP_PROFILING
  gemmlowp::RegisterCurrentThreadForProfiling();
  gemmlowp::StartProfiling();
#endif
}

void GemmlowpProfilingListener::OnBenchmarkEnd(
    const BenchmarkResults& results) {
#ifdef GEMMLOWP_PROFILING
  gemmlowp::FinishProfiling();
#endif
}

namespace {

std::vector<std::string> Split(const std::string& str, const char delim) {
  std::istringstream input(str);
  std::vector<std::string> results;
  std::string item;
  while (std::getline(input, item, delim)) {
    results.push_back(item);
  }
  return results;
}

template <typename T>
bool SplitAndParse(const std::string& str, char delim, std::vector<T>* values) {
  std::istringstream input(str);
  bool first = true;
  while (!input.eof()) {
    if (!first) {
      char c;
      input >> c;
      if (c != delim) {
        return false;
      }
    } else {
      first = false;
    }
    T val;
    input >> val;
    if (!input.eof() && !input.good()) {
      return false;
    }
    values->push_back(val);
  }
  return true;
}

template <typename T>
void FillRandomValue(T* ptr, int num_elements,
                     const std::function<T()>& random_func) {
  for (int i = 0; i < num_elements; ++i) {
    *ptr++ = random_func();
  }
}

void FillRandomString(tflite::DynamicBuffer* buffer,
                      const std::vector<int>& sizes,
                      const std::function<string()>& random_func) {
  int num_elements = 1;
  for (int dim : sizes) {
    num_elements *= dim;
  }
  for (int i = 0; i < num_elements; ++i) {
    auto str = random_func();
    buffer->AddString(str.data(), str.length());
  }
}

bool PopulateInputLayerInfo(
    const string& names_string, const string& shapes_string,
    std::vector<BenchmarkTfLiteModel::InputLayerInfo>* info) {
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
    return false;
  }

  for (int i = 0; i < names.size(); ++i) {
    info->push_back(BenchmarkTfLiteModel::InputLayerInfo());
    BenchmarkTfLiteModel::InputLayerInfo& input = info->back();

    input.name = names[i];

    TFLITE_BENCHMARK_CHECK(SplitAndParse(shapes[i], ',', &input.shape))
        << "Incorrect size string specified: " << shapes[i];
    for (int dim : input.shape) {
      if (dim == -1) {
        TFLITE_LOG(ERROR)
            << "Any unknown sizes in the shapes (-1's) must be replaced"
            << " with the size you want to benchmark with.";
        return false;
      }
    }
  }

  return true;
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
  default_params.AddParam("use_nnapi", BenchmarkParam::Create<bool>(false));
  default_params.AddParam("use_legacy_nnapi",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam("use_gpu", BenchmarkParam::Create<bool>(false));
  default_params.AddParam("allow_fp16", BenchmarkParam::Create<bool>(false));
  return default_params;
}

BenchmarkTfLiteModel::BenchmarkTfLiteModel()
    : BenchmarkTfLiteModel(DefaultParams()) {}

BenchmarkTfLiteModel::BenchmarkTfLiteModel(BenchmarkParams params)
    : BenchmarkModel(std::move(params)) {
  AddListener(&profiling_listener_);
  AddListener(&gemmlowp_profiling_listener_);
}

void BenchmarkTfLiteModel::CleanUp() {
  if (inputs_data_.empty()) {
    return;
  }
  // Free up any pre-allocated tensor data during PrepareInputData.
  for (int i = 0; i < inputs_data_.size(); ++i) {
    delete[] inputs_data_[i].data.raw;
  }
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
      CreateFlag<bool>("use_nnapi", &params_, "use nnapi delegate api"),
      CreateFlag<bool>("use_legacy_nnapi", &params_, "use legacy nnapi api"),
      CreateFlag<bool>("use_gpu", &params_, "use gpu"),
      CreateFlag<bool>("allow_fp16", &params_, "allow fp16")};

  flags.insert(flags.end(), specific_flags.begin(), specific_flags.end());
  return flags;
}

void BenchmarkTfLiteModel::LogParams() {
  BenchmarkModel::LogParams();
  TFLITE_LOG(INFO) << "Graph: [" << params_.Get<std::string>("graph") << "]";
  TFLITE_LOG(INFO) << "Input layers: ["
                   << params_.Get<std::string>("input_layer") << "]";
  TFLITE_LOG(INFO) << "Input shapes: ["
                   << params_.Get<std::string>("input_layer_shape") << "]";
  TFLITE_LOG(INFO) << "Use nnapi : [" << params_.Get<bool>("use_nnapi") << "]";
  TFLITE_LOG(INFO) << "Use legacy nnapi : ["
                   << params_.Get<bool>("use_legacy_nnapi") << "]";
  TFLITE_LOG(INFO) << "Use gpu : [" << params_.Get<bool>("use_gpu") << "]";
  TFLITE_LOG(INFO) << "Allow fp16 : [" << params_.Get<bool>("allow_fp16")
                   << "]";
}

bool BenchmarkTfLiteModel::ValidateParams() {
  if (params_.Get<std::string>("graph").empty()) {
    TFLITE_LOG(ERROR)
        << "Please specify the name of your TF Lite input file with --graph";
    return false;
  }
  return PopulateInputLayerInfo(params_.Get<std::string>("input_layer"),
                                params_.Get<std::string>("input_layer_shape"),
                                &inputs);
}

uint64_t BenchmarkTfLiteModel::ComputeInputBytes() {
  TFLITE_BENCHMARK_CHECK(interpreter);
  uint64_t total_input_bytes = 0;
  for (int input : interpreter->inputs()) {
    auto* t = interpreter->tensor(input);
    total_input_bytes += t->bytes;
  }
  return total_input_bytes;
}

void BenchmarkTfLiteModel::PrepareInputData() {
  auto interpreter_inputs = interpreter->inputs();
  const size_t input_size = interpreter_inputs.size();
  CleanUp();

  for (int j = 0; j < input_size; ++j) {
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter->tensor(i);
    std::vector<int> sizes = TfLiteIntArrayToVector(t->dims);
    int num_elements = 1;
    // TODO(haoliang): Ignore the 0-th dimension (number of batches).
    for (int i = 1; i < sizes.size(); ++i) {
      num_elements *= sizes[i];
    }
    InputTensorData t_data;
    if (t->type == kTfLiteFloat32) {
      t_data.bytes = sizeof(float) * num_elements;
      t_data.data.raw = new char[t_data.bytes];
      FillRandomValue<float>(t_data.data.f, num_elements, []() {
        return static_cast<float>(rand()) / RAND_MAX - 0.5f;
      });
    } else if (t->type == kTfLiteInt32) {
      // TODO(yunluli): This is currently only used for handling embedding input
      // for speech models. Generalize if necessary.
      t_data.bytes = sizeof(int32_t) * num_elements;
      t_data.data.raw = new char[t_data.bytes];
      FillRandomValue<int32_t>(t_data.data.i32, num_elements, []() {
        return static_cast<int32_t>(rand()) % 100;
      });
    } else if (t->type == kTfLiteUInt8) {
      t_data.bytes = sizeof(uint8_t) * num_elements;
      t_data.data.raw = new char[t_data.bytes];
      FillRandomValue<uint8_t>(t_data.data.uint8, num_elements, []() {
        return static_cast<uint8_t>(rand()) % 255;
      });
    } else if (t->type == kTfLiteInt8) {
      t_data.bytes = sizeof(int8_t) * num_elements;
      t_data.data.raw = new char[t_data.bytes];
      FillRandomValue<int8_t>(t_data.data.int8, num_elements, []() {
        return static_cast<int8_t>(rand()) % 255 - 127;
      });
    } else if (t->type == kTfLiteString) {
      // TODO(haoliang): No need to cache string tensors right now.
    } else {
      TFLITE_LOG(FATAL) << "Don't know how to populate tensor " << t->name
                        << " of type " << t->type;
    }
    inputs_data_.push_back(t_data);
  }
}

void BenchmarkTfLiteModel::ResetInputsAndOutputs() {
  auto interpreter_inputs = interpreter->inputs();
  // Set the values of the input tensors from inputs_data_.
  for (int j = 0; j < interpreter_inputs.size(); ++j) {
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter->tensor(i);
    if (t->type == kTfLiteFloat32) {
      std::memcpy(interpreter->typed_tensor<float>(i), inputs_data_[j].data.f,
                  inputs_data_[j].bytes);
    } else if (t->type == kTfLiteInt32) {
      std::memcpy(interpreter->typed_tensor<int32_t>(i),
                  inputs_data_[j].data.i32, inputs_data_[j].bytes);
    } else if (t->type == kTfLiteUInt8) {
      std::memcpy(interpreter->typed_tensor<uint8_t>(i),
                  inputs_data_[j].data.uint8, inputs_data_[j].bytes);
    } else if (t->type == kTfLiteInt8) {
      std::memcpy(interpreter->typed_tensor<int8_t>(i),
                  inputs_data_[j].data.int8, inputs_data_[j].bytes);
    } else if (t->type == kTfLiteString) {
      tflite::DynamicBuffer buffer;
      std::vector<int> sizes = TfLiteIntArrayToVector(t->dims);
      FillRandomString(&buffer, sizes, []() {
        return "we're have some friends over saturday to hang out in the yard";
      });
      buffer.WriteToTensor(interpreter->tensor(i), /*new_shape=*/nullptr);
    } else {
      TFLITE_LOG(FATAL) << "Don't know how to populate tensor " << t->name
                        << " of type " << t->type;
    }
  }
}

void BenchmarkTfLiteModel::Init() {
  std::string graph = params_.Get<std::string>("graph");
  model = tflite::FlatBufferModel::BuildFromFile(graph.c_str());
  if (!model) {
    TFLITE_LOG(FATAL) << "Failed to mmap model " << graph;
  }
  TFLITE_LOG(INFO) << "Loaded model " << graph;
  model->error_reporter();
  TFLITE_LOG(INFO) << "resolved reporter";

#ifdef TFLITE_CUSTOM_OPS_HEADER
  tflite::MutableOpResolver resolver;
  RegisterSelectedOps(&resolver);
#else
  tflite::ops::builtin::BuiltinOpResolver resolver;
#endif

  const int32_t num_threads = params_.Get<int32_t>("num_threads");
  tflite::InterpreterBuilder(*model, resolver)(&interpreter, num_threads);
  if (!interpreter) {
    TFLITE_LOG(FATAL) << "Failed to construct interpreter";
  }
  profiling_listener_.SetInterpreter(interpreter.get());

  interpreter->UseNNAPI(params_.Get<bool>("use_legacy_nnapi"));

  delegates_ = GetDelegates();
  for (const auto& delegate : delegates_) {
    if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) !=
        kTfLiteOk) {
      TFLITE_LOG(FATAL) << "Failed to apply " << delegate.first << " delegate.";
    } else {
      TFLITE_LOG(INFO) << "Applied " << delegate.first << " delegate.";
    }
  }

  interpreter->SetAllowFp16PrecisionForFp32(params_.Get<bool>("allow_fp16"));

  auto interpreter_inputs = interpreter->inputs();

  if (!inputs.empty()) {
    TFLITE_BENCHMARK_CHECK_EQ(inputs.size(), interpreter_inputs.size())
        << "Inputs mismatch: Model inputs #:" << interpreter_inputs.size()
        << " expected: " << inputs.size();
  }

  // Check if the tensor names match, and log a warning if it doesn't.
  // TODO(ycling): Consider to make this an error again when the new converter
  // create tensors with consistent naming.
  for (int j = 0; j < inputs.size(); ++j) {
    const InputLayerInfo& input = inputs[j];
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter->tensor(i);
    if (input.name != t->name) {
      TFLITE_LOG(WARN) << "Tensor # " << i << " is named " << t->name
                       << " but flags call it " << input.name;
    }
  }

  // Resize all non-string tensors.
  for (int j = 0; j < inputs.size(); ++j) {
    const InputLayerInfo& input = inputs[j];
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter->tensor(i);
    if (t->type != kTfLiteString) {
      interpreter->ResizeInputTensor(i, input.shape);
    }
  }

  // Don't allocate tensors if we have delegates.
  if (delegates_.empty() && interpreter->AllocateTensors() != kTfLiteOk) {
    TFLITE_LOG(FATAL) << "Failed to allocate tensors!";
  }
}

BenchmarkTfLiteModel::TfLiteDelegatePtrMap BenchmarkTfLiteModel::GetDelegates()
    const {
  TfLiteDelegatePtrMap delegates;
  if (params_.Get<bool>("use_gpu")) {
#if defined(__ANDROID__)
    delegates.emplace("GPU", CreateGPUDelegate(model.get()));
#else
    TFLITE_LOG(WARN) << "GPU acceleration is unsupported on this platform.";
#endif
  }
  if (params_.Get<bool>("use_nnapi")) {
#if defined(__ANDROID__)
    delegates.emplace("NNAPI", CreateNNAPIDelegate());
#else
    TFLITE_LOG(WARN) << "NNAPI acceleration is unsupported on this platform.";
#endif
  }
  return delegates;
}

void BenchmarkTfLiteModel::RunImpl() {
  if (interpreter->Invoke() != kTfLiteOk) {
    TFLITE_LOG(FATAL) << "Failed to invoke!";
  }
}

}  // namespace benchmark
}  // namespace tflite
