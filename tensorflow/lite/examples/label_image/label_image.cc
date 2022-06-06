/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/examples/label_image/label_image.h"

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/examples/label_image/log.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace label_image {

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using ProvidedDelegateList = tflite::tools::ProvidedDelegateList;

class DelegateProviders {
 public:
  DelegateProviders() : delegate_list_util_(&params_) {
    delegate_list_util_.AddAllDelegateParams();
    delegate_list_util_.AppendCmdlineFlags(flags_);

    // Remove the "help" flag to avoid printing "--help=false"
    params_.RemoveParam("help");
    delegate_list_util_.RemoveCmdlineFlag(flags_, "help");
  }

  // Initialize delegate-related parameters from parsing command line arguments,
  // and remove the matching arguments from (*argc, argv). Returns true if all
  // recognized arg values are parsed correctly.
  bool InitFromCmdlineArgs(int* argc, const char** argv) {
    // Note if '--help' is in argv, the Flags::Parse return false,
    // see the return expression in Flags::Parse.
    return Flags::Parse(argc, argv, flags_);
  }

  // According to passed-in settings `s`, this function sets corresponding
  // parameters that are defined by various delegate execution providers. See
  // lite/tools/delegates/README.md for the full list of parameters defined.
  void MergeSettingsIntoParams(const Settings& s) {
    // Parse settings related to GPU delegate.
    // Note that GPU delegate does support OpenCL. 'gl_backend' was introduced
    // when the GPU delegate only supports OpenGL. Therefore, we consider
    // setting 'gl_backend' to true means using the GPU delegate.
    if (s.gl_backend) {
      if (!params_.HasParam("use_gpu")) {
        LOG(WARN) << "GPU delegate execution provider isn't linked or GPU "
                     "delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_gpu", true);
        // The parameter "gpu_inference_for_sustained_speed" isn't available for
        // iOS devices.
        if (params_.HasParam("gpu_inference_for_sustained_speed")) {
          params_.Set<bool>("gpu_inference_for_sustained_speed", true);
        }
        params_.Set<bool>("gpu_precision_loss_allowed", s.allow_fp16);
      }
    }

    // Parse settings related to NNAPI delegate.
    if (s.accel) {
      if (!params_.HasParam("use_nnapi")) {
        LOG(WARN) << "NNAPI delegate execution provider isn't linked or NNAPI "
                     "delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_nnapi", true);
        params_.Set<bool>("nnapi_allow_fp16", s.allow_fp16);
      }
    }

    // Parse settings related to Hexagon delegate.
    if (s.hexagon_delegate) {
      if (!params_.HasParam("use_hexagon")) {
        LOG(WARN) << "Hexagon delegate execution provider isn't linked or "
                     "Hexagon delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_hexagon", true);
        params_.Set<bool>("hexagon_profiling", s.profiling);
      }
    }

    // Parse settings related to XNNPACK delegate.
    if (s.xnnpack_delegate) {
      if (!params_.HasParam("use_xnnpack")) {
        LOG(WARN) << "XNNPACK delegate execution provider isn't linked or "
                     "XNNPACK delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_xnnpack", true);
        params_.Set<int32_t>("num_threads", s.number_of_threads);
      }
    }
  }

  // Create a list of TfLite delegates based on what have been initialized (i.e.
  // 'params_').
  std::vector<ProvidedDelegateList::ProvidedDelegate> CreateAllDelegates()
      const {
    return delegate_list_util_.CreateAllRankedDelegates();
  }

  std::string GetHelpMessage(const std::string& cmdline) const {
    return Flags::Usage(cmdline, flags_);
  }

 private:
  // Contain delegate-related parameters that are initialized from command-line
  // flags.
  tflite::tools::ToolParams params_;

  // A helper to create TfLite delegates.
  ProvidedDelegateList delegate_list_util_;

  // Contains valid flags
  std::vector<tflite::Flag> flags_;
};

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(ERROR) << "Labels file " << file_name << " not found";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void PrintProfilingInfo(const profiling::ProfileEvent* e,
                        uint32_t subgraph_index, uint32_t op_index,
                        TfLiteRegistration registration) {
  // output something like
  // time (ms) , Node xxx, OpCode xxx, symbolic name
  //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

  LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->elapsed_time) / 1000.0 << ", Subgraph " << std::setw(3)
            << std::setprecision(3) << subgraph_index << ", Node "
            << std::setw(3) << std::setprecision(3) << op_index << ", OpCode "
            << std::setw(3) << std::setprecision(3) << registration.builtin_code
            << ", "
            << EnumNameBuiltinOperator(
                   static_cast<BuiltinOperator>(registration.builtin_code));
}

void RunInference(Settings* settings,
                  const DelegateProviders& delegate_providers) {
  if (!settings->model_name.c_str()) {
    LOG(ERROR) << "no model file name";
    exit(-1);
  }

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(settings->model_name.c_str());
  if (!model) {
    LOG(ERROR) << "Failed to mmap model " << settings->model_name;
    exit(-1);
  }
  settings->model = model.get();
  LOG(INFO) << "Loaded model " << settings->model_name;
  model->error_reporter();
  LOG(INFO) << "resolved reporter";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(ERROR) << "Failed to construct interpreter";
    exit(-1);
  }

  interpreter->SetAllowFp16PrecisionForFp32(settings->allow_fp16);

  if (settings->verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size();
    LOG(INFO) << "nodes size: " << interpreter->nodes_size();
    LOG(INFO) << "inputs: " << interpreter->inputs().size();
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0);

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point;
    }
  }

  if (settings->number_of_threads != -1) {
    interpreter->SetNumThreads(settings->number_of_threads);
  }

  int image_width = 224;
  int image_height = 224;
  int image_channels = 3;
  std::vector<uint8_t> in = read_bmp(settings->input_bmp_name, &image_width,
                                     &image_height, &image_channels, settings);

  int input = interpreter->inputs()[0];
  if (settings->verbose) LOG(INFO) << "input: " << input;

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (settings->verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size();
    LOG(INFO) << "number of outputs: " << outputs.size();
  }

  auto delegates = delegate_providers.CreateAllDelegates();
  for (auto& delegate : delegates) {
    const auto delegate_name = delegate.provider->GetName();
    if (interpreter->ModifyGraphWithDelegate(std::move(delegate.delegate)) !=
        kTfLiteOk) {
      LOG(ERROR) << "Failed to apply " << delegate_name << " delegate.";
      exit(-1);
    } else {
      LOG(INFO) << "Applied " << delegate_name << " delegate.";
    }
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Failed to allocate tensors!";
    exit(-1);
  }

  if (settings->verbose) PrintInterpreterState(interpreter.get());

  // get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  settings->input_type = interpreter->tensor(input)->type;
  switch (settings->input_type) {
    case kTfLiteFloat32:
      resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels, settings);
      break;
    case kTfLiteInt8:
      resize<int8_t>(interpreter->typed_tensor<int8_t>(input), in.data(),
                     image_height, image_width, image_channels, wanted_height,
                     wanted_width, wanted_channels, settings);
      break;
    case kTfLiteUInt8:
      resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                      image_height, image_width, image_channels, wanted_height,
                      wanted_width, wanted_channels, settings);
      break;
    default:
      LOG(ERROR) << "cannot handle input type "
                 << interpreter->tensor(input)->type << " yet";
      exit(-1);
  }
  auto profiler = absl::make_unique<profiling::Profiler>(
      settings->max_profiling_buffer_entries);
  interpreter->SetProfiler(profiler.get());

  if (settings->profiling) profiler->StartProfiling();
  for (int i = 0; i < settings->number_of_warmup_runs; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(ERROR) << "Failed to invoke tflite!";
      exit(-1);
    }
  }

  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);
  for (int i = 0; i < settings->loop_count; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(ERROR) << "Failed to invoke tflite!";
      exit(-1);
    }
  }
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "invoked";
  LOG(INFO) << "average time: "
            << (get_us(stop_time) - get_us(start_time)) /
                   (settings->loop_count * 1000)
            << " ms";

  if (settings->profiling) {
    profiler->StopProfiling();
    auto profile_events = profiler->GetProfileEvents();
    for (int i = 0; i < profile_events.size(); i++) {
      auto subgraph_index = profile_events[i]->extra_event_metadata;
      auto op_index = profile_events[i]->event_metadata;
      const auto subgraph = interpreter->subgraph(subgraph_index);
      const auto node_and_registration =
          subgraph->node_and_registration(op_index);
      const TfLiteRegistration registration = node_and_registration->second;
      PrintProfilingInfo(profile_events[i], subgraph_index, op_index,
                         registration);
    }
  }

  const float threshold = 0.001f;

  std::vector<std::pair<float, int>> top_results;

  int output = interpreter->outputs()[0];
  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  switch (interpreter->tensor(output)->type) {
    case kTfLiteFloat32:
      get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                       settings->number_of_results, threshold, &top_results,
                       settings->input_type);
      break;
    case kTfLiteInt8:
      get_top_n<int8_t>(interpreter->typed_output_tensor<int8_t>(0),
                        output_size, settings->number_of_results, threshold,
                        &top_results, settings->input_type);
      break;
    case kTfLiteUInt8:
      get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                         output_size, settings->number_of_results, threshold,
                         &top_results, settings->input_type);
      break;
    default:
      LOG(ERROR) << "cannot handle output type "
                 << interpreter->tensor(output)->type << " yet";
      exit(-1);
  }

  std::vector<string> labels;
  size_t label_count;

  if (ReadLabelsFile(settings->labels_file_name, &labels, &label_count) !=
      kTfLiteOk)
    exit(-1);

  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    LOG(INFO) << confidence << ": " << index << " " << labels[index];
  }

  // Destory the interpreter earlier than delegates objects.
  interpreter.reset();
}

void display_usage(const DelegateProviders& delegate_providers) {
  LOG(INFO)
      << "\n"
      << delegate_providers.GetHelpMessage("label_image")
      << "\t--accelerated, -a: [0|1] use Android NNAPI or not\n"
      << "\t--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or "
         "not\n"
      << "\t--count, -c: loop interpreter->Invoke() for certain times\n"
      << "\t--gl_backend, -g: [0|1]: use GL GPU Delegate on Android\n"
      << "\t--hexagon_delegate, -j: [0|1]: use Hexagon Delegate on Android\n"
      << "\t--input_mean, -b: input mean\n"
      << "\t--input_std, -s: input standard deviation\n"
      << "\t--image, -i: image_name.bmp\n"
      << "\t--labels, -l: labels for the model\n"
      << "\t--tflite_model, -m: model_name.tflite\n"
      << "\t--profiling, -p: [0|1], profiling or not\n"
      << "\t--num_results, -r: number of results to show\n"
      << "\t--threads, -t: number of threads\n"
      << "\t--verbose, -v: [0|1] print more information\n"
      << "\t--warmup_runs, -w: number of warmup runs\n"
      << "\t--xnnpack_delegate, -x [0:1]: xnnpack delegate\n"
      << "\t--help, -h: Print this help message\n";
}

int Main(int argc, char** argv) {
  DelegateProviders delegate_providers;
  bool parse_result = delegate_providers.InitFromCmdlineArgs(
      &argc, const_cast<const char**>(argv));
  if (!parse_result) {
    display_usage(delegate_providers);
    return EXIT_FAILURE;
  }

  Settings s;

  int c;
  while (true) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"allow_fp16", required_argument, nullptr, 'f'},
        {"count", required_argument, nullptr, 'c'},
        {"verbose", required_argument, nullptr, 'v'},
        {"image", required_argument, nullptr, 'i'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"profiling", required_argument, nullptr, 'p'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"num_results", required_argument, nullptr, 'r'},
        {"max_profiling_buffer_entries", required_argument, nullptr, 'e'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"gl_backend", required_argument, nullptr, 'g'},
        {"hexagon_delegate", required_argument, nullptr, 'j'},
        {"xnnpack_delegate", required_argument, nullptr, 'x'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "a:b:c:d:e:f:g:i:j:l:m:p:r:s:t:v:w:x:h",
                    long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'e':
        s.max_profiling_buffer_entries =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'f':
        s.allow_fp16 =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'g':
        s.gl_backend =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_bmp_name = optarg;
        break;
      case 'j':
        s.hexagon_delegate = optarg;
        break;
      case 'l':
        s.labels_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'p':
        s.profiling =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'r':
        s.number_of_results =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'x':
        s.xnnpack_delegate =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage(delegate_providers);
        exit(-1);
      default:
        exit(-1);
    }
  }

  delegate_providers.MergeSettingsIntoParams(s);
  RunInference(&s, delegate_providers);
  return 0;
}

}  // namespace label_image
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::label_image::Main(argc, argv);
}
