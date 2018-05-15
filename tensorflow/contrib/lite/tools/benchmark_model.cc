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

#include <cstdarg>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

#ifdef TFLITE_CUSTOM_OPS_HEADER
void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);
#endif

namespace tflite {

using ::tensorflow::Env;
using ::tensorflow::str_util::Split;
using ::tensorflow::str_util::SplitAndParseAsFloats;
using ::tensorflow::str_util::SplitAndParseAsInts;

struct InputLayerInfo {
  string name;
  TfLiteType data_type;
  std::vector<int> shape;
  // Note that initialization_values is currently unused.
  std::vector<float> initialization_values;
};

template <typename T>
void FillRandomValue(T* ptr, const std::vector<int>& sizes,
                     const std::function<T()>& random_func) {
  int num_elements = 1;
  for (int dim : sizes) {
    num_elements *= dim;
  }
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

TfLiteType TfLiteTypeFromString(const string& input_layer_type) {
  if (input_layer_type == "string")
    return kTfLiteString;
  else if (input_layer_type == "float")
    return kTfLiteFloat32;
  else if (input_layer_type == "uint8")
    return kTfLiteUInt8;
  else if (input_layer_type == "int32")
    return kTfLiteInt32;
  else if (input_layer_type == "int64")
    return kTfLiteInt64;
  else
    return kTfLiteNoType;
}

std::vector<int> ShapeFromTfLiteTensor(TfLiteTensor* t) {
  std::vector<int> result;
  result.reserve(t->dims->size);
  for (int i = 0; i < t->dims->size; ++i) {
    result.push_back(t->dims->data[i]);
  }
  CHECK(!result.empty()) << "Found no shapes in model";
  return result;
}

bool CreateInterpreter(const string& graph,
                       std::unique_ptr<FlatBufferModel>* model,
                       std::unique_ptr<Interpreter>* interpreter) {
  *model = tflite::FlatBufferModel::BuildFromFile(graph.c_str());
  if (!model) {
    std::cerr << "Failed to load model " << graph << std::endl;
    return false;
  }

#ifdef TFLITE_CUSTOM_OPS_HEADER
  tflite::MutableOpResolver resolver;
  RegisterSelectedOps(&resolver);
#else
  tflite::ops::builtin::BuiltinOpResolver resolver;
#endif

  tflite::InterpreterBuilder(*(model->get()), resolver)(interpreter);
  if (!(*interpreter)) {
    std::cerr << "Failed to construct interpreter" << std::endl;
    return false;
  }

  return true;
}

bool PrepareInterpreter(const std::vector<InputLayerInfo> inputs,
                        int num_threads, bool use_nnapi,
                        Interpreter* interpreter) {
  if (num_threads != -1) {
    interpreter->SetNumThreads(num_threads);
  }

  interpreter->UseNNAPI(use_nnapi);

  // Check that all names and types match
  for (const InputLayerInfo& input : inputs) {
    for (int i : interpreter->inputs()) {
      TfLiteTensor* t = interpreter->tensor(i);
      CHECK_EQ(t->name, input.name)
          << "Tensor # " << i << " is named " << t->name
          << " but flags call it " << input.name;
      CHECK_EQ(t->type, input.data_type)
          << "Could not match the type of input tensor " << t->name;
    }
  }

  // Resize all non-string tensors.
  for (const InputLayerInfo& input : inputs) {
    for (int i : interpreter->inputs()) {
      TfLiteTensor* t = interpreter->tensor(i);
      if (t->type != kTfLiteString) {
        interpreter->ResizeInputTensor(i, input.shape);
      }
    }
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors!" << std::endl;
    return false;
  }

  // Set the values of the input tensors.
  for (int i : interpreter->inputs()) {
    TfLiteTensor* t = interpreter->tensor(i);
    std::vector<int> sizes = ShapeFromTfLiteTensor(t);

    // TODO(ahentz): below we ignore the O-th dimension (number of batches).
    if (t->type == kTfLiteFloat32) {
      FillRandomValue<float>(
          interpreter->typed_tensor<float>(i),
          std::vector<int>(sizes.begin() + 1, sizes.end()),
          []() { return static_cast<float>(rand()) / RAND_MAX - 0.5f; });
    } else if (t->type == kTfLiteUInt8) {
      FillRandomValue<uint8_t>(
          interpreter->typed_tensor<uint8_t>(i),
          std::vector<int>(sizes.begin() + 1, sizes.end()),
          []() { return static_cast<uint8_t>(rand()) % 255; });
    } else if (t->type == kTfLiteString) {
      tflite::DynamicBuffer buffer;
      FillRandomString(&buffer, sizes, []() {
        return "we're have some friends over saturday to hang out in the yard";
      });
      buffer.WriteToTensor(interpreter->tensor(i));
    } else {
      std::cerr << "Don't know how to populate tensor " << t->name
                << " of type " << t->type << std::endl;
      return false;
    }
  }
  return true;
}

bool PopulateInputLayerInfo(const string& names_string,
                            const string& shapes_string,
                            const string& types_string,
                            const string& values_string,
                            std::vector<InputLayerInfo>* info) {
  std::vector<string> names = Split(names_string, ',');
  std::vector<string> shapes = Split(shapes_string, ':');
  std::vector<string> types = Split(types_string, ',');
  std::vector<string> values = Split(values_string, ':');

  if (names.size() != shapes.size()) {
    LOG(ERROR) << "The number of items in"
               << " --input_layer_shape (" << shapes_string << ", with "
               << shapes.size() << " items)"
               << " must match the number of items in"
               << " --input_layer (" << names_string << ", with "
               << names.size() << " items)."
               << " For example --input_layer=input1,input2"
               << " --input_layer_shape=1,224,224,4:1,20";
    return false;
  }
  if (names.size() != types.size()) {
    LOG(ERROR) << "The number of items in"
               << " --input_layer_type (" << types_string << ", with "
               << types.size() << " items)"
               << " must match the number of items in"
               << " --input_layer (" << names_string << ", with "
               << names.size() << " items)."
               << " For example --input_layer=input1,input2"
               << " --input_layer_type=float,int";
    return false;
  }

  for (int i = 0; i < names.size(); ++i) {
    info->push_back(InputLayerInfo());
    InputLayerInfo& input = info->back();

    input.name = names[i];

    input.data_type = TfLiteTypeFromString(types[i]);
    CHECK(input.data_type != kTfLiteNoType)
        << types[i] << " was an invalid type";

    CHECK(SplitAndParseAsInts(shapes[i], ',', &input.shape))
        << "Incorrect size string specified: " << shapes[i];
    for (int dim : input.shape) {
      if (dim == -1) {
        LOG(ERROR) << "Any unknown sizes in the shapes (-1's) must be replaced"
                   << " with the size you want to benchmark with.";
        return false;
      }
    }

    if (i < values.size()) {
      CHECK(SplitAndParseAsFloats(values[i], ',', &input.initialization_values))
          << "Incorrect initialization values string specified: " << values[i];
    }
  }

  return true;
}

bool RunBenchmark(Interpreter* interpreter, int64_t* inference_time_us) {
  const int64_t start_time = Env::Default()->NowMicros();

  if (interpreter->Invoke() != kTfLiteOk) {
    std::cerr << "Failed to invoke!";
    return false;
  }

  const int64_t end_time = Env::Default()->NowMicros();
  *inference_time_us = end_time - start_time;
  return true;
}

class Latencies {
 public:
  void AddMeasurement(int64_t time_us) {
    max_ = std::max(time_us, max_);
    min_ = std::min(time_us, min_);
    ++count_;
    sum_ += time_us;
    squared_sum_ += static_cast<double>(time_us) * time_us;
  }

  double avg() const {
    if (count_ == 0) return std::numeric_limits<int64_t>::quiet_NaN();
    return static_cast<double>(sum_) / count_;
  }

  int64_t std_deviation() const {
    if (count_ == 0 || min_ == max_) return 0;
    return sqrt(squared_sum_ / count_ - avg() * avg());
  }

  void OutputToStream(std::ostream* stream) const {
    *stream << "count=" << count_;
    if (count_ == 0) return;
    *stream << " min=" << min_ << " max=" << max_;
    *stream << " avg=" << avg() << " std=" << std_deviation();
  }

 private:
  int64_t count_ = 0;
  int64_t min_ = std::numeric_limits<int64_t>::max();
  int64_t max_ = std::numeric_limits<int64_t>::min();
  int64_t sum_ = 0;
  double squared_sum_ = 0;
};

bool TimeMultipleRuns(Interpreter* interpreter, double sleep_seconds,
                      int num_runs, int64* total_time_us) {
  // Convert the run_delay string into a timespec.
  timespec req;
  req.tv_sec = static_cast<time_t>(sleep_seconds);
  req.tv_nsec = (sleep_seconds - req.tv_sec) * 1000000000;

  *total_time_us = 0;

  std::cout << "Running benchmark for " << num_runs
            << " iterations: " << std::endl;

  Latencies latencies;
  for (int i = 0; i < num_runs; ++i) {
    int64_t time_us;
    bool run_status = RunBenchmark(interpreter, &time_us);
    latencies.AddMeasurement(time_us);
    *total_time_us += time_us;
    if (!run_status) {
      std::cout << "Failed on run " << i << std::endl;
      return false;
    }

    // If requested, sleep between runs for an arbitrary amount of time.
    // This can be helpful to determine the effect of mobile processor
    // scaling and thermal throttling.
    if (sleep_seconds > 0.0) {
#ifdef PLATFORM_WINDOWS
      Sleep(sleep_seconds * 1000);
#else
      nanosleep(&req, nullptr);
#endif
    }
  }
  latencies.OutputToStream(&std::cout);
  std::cout << std::endl;

  return true;
}

int Main(int argc, char** argv) {
  using tensorflow::Flag;
  using tensorflow::Flags;

  string graph;               // e.g.: /data/local/tmp/tfl_inception-v1_model.fb
  string input_layer_string;  // e.g.: input
  string input_layer_shape_string;  // e.g.: 1,224,224,3
  string input_layer_type_string;   // e.g.: float
  string input_layer_values_string;
  string output_layer_string;  // e.g.: output
  int num_runs = 50;
  string run_delay = "-1.0";
  int num_threads = 1;
  string benchmark_name = "";
  string output_prefix = "";
  int warmup_runs = 1;
  bool use_nnapi = false;

  std::vector<Flag> flag_list = {
      Flag("graph", &graph, "graph file name"),
      // All the following flags are optional, but can be used in order
      // to benchmark different input shapes.
      Flag("input_layer", &input_layer_string, "input layer names"),
      Flag("input_layer_shape", &input_layer_shape_string, "input layer shape"),
      Flag("input_layer_type", &input_layer_type_string, "input layer type"),
      Flag("input_layer_values", &input_layer_values_string,
           "values to initialize the inputs with"),
      Flag("output_layer", &output_layer_string, "output layer name"),
      Flag("num_runs", &num_runs, "number of runs"),
      Flag("run_delay", &run_delay, "delay between runs in seconds"),
      Flag("num_threads", &num_threads, "number of threads"),
      Flag("benchmark_name", &benchmark_name, "benchmark name"),
      Flag("output_prefix", &output_prefix, "benchmark output prefix"),
      Flag("warmup_runs", &warmup_runs, "how many runs to initialize model"),
      Flag("use_nnapi", &use_nnapi, "use nnapi api"),
  };
  string usage = Flags::Usage(argv[0], flag_list);
  const bool parse_result = Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  if (!parse_result) {
    std::cerr << usage << std::endl;
    return -1;
  }

  std::cout << "Graph: [" << graph << "]" << std::endl;
  if (!input_layer_string.empty()) {
    std::cout << "Input layers: [" << input_layer_string << "]" << std::endl;
    std::cout << "Input shapes: [" << input_layer_shape_string << "]"
              << std::endl;
    std::cout << "Input types: [" << input_layer_type_string << "]"
              << std::endl;
  }
  if (!output_layer_string.empty()) {
    std::cout << "Output layers: [" << output_layer_string << "]" << std::endl;
  }
  std::cout << "Num runs: [" << num_runs << "]" << std::endl;
  std::cout << "Inter-run delay (seconds): [" << run_delay << "]" << std::endl;
  std::cout << "Num threads: [" << num_threads << "]" << std::endl;
  if (!benchmark_name.empty()) {
    std::cout << "Benchmark name: [" << benchmark_name << "]" << std::endl;
    std::cout << "Output prefix: [" << output_prefix << "]" << std::endl;
  }
  std::cout << "Warmup runs: [" << warmup_runs << "]" << std::endl;
  std::cout << "Use nnapi : [" << use_nnapi << "]" << std::endl;

  if (graph.empty()) {
    std::cout
        << "Please specify the name of your TF Lite input file with --graph"
        << std::endl;
    return -1;
  }

  std::vector<InputLayerInfo> inputs;
  if (!PopulateInputLayerInfo(input_layer_string, input_layer_shape_string,
                              input_layer_type_string,
                              input_layer_values_string, &inputs)) {
    return -1;
  }

  int64 initialization_start_us = Env::Default()->NowMicros();

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (!CreateInterpreter(graph, &model, &interpreter)) {
    return -1;
  }
  if (!PrepareInterpreter(inputs, num_threads, use_nnapi, interpreter.get())) {
    return -1;
  }

  int64 initialization_end_us = Env::Default()->NowMicros();

  const double initialization_time_s =
      (initialization_end_us - initialization_start_us) / 1000000.0f;
  std::cout << "Initialized session in " << initialization_time_s << "s"
            << std::endl;

  const double sleep_seconds = std::strtod(run_delay.c_str(), nullptr);

  // If requested, run through the graph first to preinitialize everything
  // before the benchmarking runs.
  int64 warmup_time_us = 0;
  if (warmup_runs > 0) {
    if (!TimeMultipleRuns(interpreter.get(), sleep_seconds, warmup_runs,
                          &warmup_time_us)) {
      std::cerr << "Warmup failed" << std::endl;
      return -1;
    }
  }

  // Capture overall inference time without stat logging overhead. This is the
  // timing data that can be compared to other libaries.
  int64 no_stat_time_us = 0;
  if (!TimeMultipleRuns(interpreter.get(), sleep_seconds, num_runs,
                        &no_stat_time_us)) {
    std::cerr << "Timing failed." << std::endl;
    return -1;
  }

  std::cout << "Average inference timings in us: " << no_stat_time_us / num_runs
            << " , Warmup: "
            << (warmup_runs > 0 ? warmup_time_us / warmup_runs : 0) << ", "
            << std::endl;

  return 0;
}

}  // namespace tflite

int main(int argc, char** argv) { return ::tflite::Main(argc, argv); }
