/* Copyright 2016 Google Inc. All Rights Reserved.

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

// A C++ binary to benchmark a compute graph and its individual operators,
// both on desktop machines and on Android.
//
// See README.md for usage instructions.

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/stat_summarizer.h"

namespace tensorflow {

// Global variables that holds the Tensorflow classifier.
static std::unique_ptr<tensorflow::Session> session;

static std::unique_ptr<tensorflow::StatSummarizer> g_stats;

struct Flags {
  string graph = "/data/local/tmp/tensorflow_inception_graph.pb";
  string input_layer = "input:0";
  string input_layer_shape = "1,224,224,3";
  string input_layer_type = "float";
  string output_layer = "output:0";
  int num_runs = 50;
  string run_delay = "-1.0";
  int num_threads = -1;
};

static Flags* flags;  // Filled in by main()

static bool InitializeBenchmark() {
  LOG(INFO) << "Loading Tensorflow.";

  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& config = options.config;
  if (flags->num_threads > 0) {
    config.set_intra_op_parallelism_threads(flags->num_threads);
  }
  LOG(INFO) << "Got config, " << config.device_count_size() << " devices";

  session.reset(tensorflow::NewSession(options));
  tensorflow::GraphDef tensorflow_graph;
  Status s = ReadBinaryProto(Env::Default(), flags->graph, &tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Graph: " << s;
    return false;
  }

  g_stats.reset(new tensorflow::StatSummarizer(tensorflow_graph));

  s = session->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Session: " << s;
    return false;
  }

  // Clear the proto to save memory space.
  tensorflow_graph.Clear();
  return true;
}

static bool RunBenchmark() {
  DataType input_data_type;
  CHECK(DataTypeFromString(flags->input_layer_type, &input_data_type))
      << flags->input_layer_type << " was an invalid type";

  std::vector<int32> sizes;
  CHECK(str_util::SplitAndParseAsInts(flags->input_layer_shape, ',', &sizes))
      << "Incorrect size string specified: " << flags->input_layer_shape;
  TensorShape input_shape;
  for (int i = 0; i < sizes.size(); ++i) {
    input_shape.AddDim(sizes[i]);
  }

  Tensor input_tensor(input_data_type, input_shape);

  switch (input_data_type) {
    case DT_INT32: {
      auto int_tensor = input_tensor.flat<int32>();
      int_tensor = int_tensor.constant(0.0);
      break;
    }
    case DT_FLOAT: {
      auto float_tensor = input_tensor.flat<float>();
      float_tensor = float_tensor.constant(0.0);
      break;
    }
    case DT_QUINT8: {
      auto int_tensor = input_tensor.flat<quint8>();
      int_tensor = int_tensor.constant(0.0);
      break;
    }
    default:
      LOG(FATAL) << "Unsupported input type: " << flags->input_layer_type;
  }

  std::vector<std::pair<string, tensorflow::Tensor> > input_tensors(
      {{flags->input_layer, input_tensor}});

  std::vector<tensorflow::Tensor> output_tensors;
  std::vector<string> output_names({flags->output_layer});

  tensorflow::Status s;

  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  RunMetadata run_metadata;

  s = session->Run(run_options, input_tensors, output_names, {},
                   &output_tensors, &run_metadata);

  assert(run_metadata.has_step_stats());

  const StepStats& stats = run_metadata.step_stats();

  g_stats->ProcessStepStats(stats);

  if (!s.ok()) {
    LOG(ERROR) << "Error during inference: " << s;
    return false;
  }
  return true;
}

}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::flags = new tensorflow::Flags();

  const bool parse_result = tensorflow::ParseFlags(
      &argc, argv,
      {
          tensorflow::Flag("graph", &tensorflow::flags->graph),
          tensorflow::Flag("input_layer", &tensorflow::flags->input_layer),
          tensorflow::Flag("input_layer_shape",
                           &tensorflow::flags->input_layer_shape),
          tensorflow::Flag("input_layer_type",
                           &tensorflow::flags->input_layer_type),
          tensorflow::Flag("output_layer", &tensorflow::flags->output_layer),
          tensorflow::Flag("num_runs", &tensorflow::flags->num_runs),
          tensorflow::Flag("run_delay", &tensorflow::flags->run_delay),
          tensorflow::Flag("num_threads", &tensorflow::flags->num_threads),
      });

  if (!parse_result) {
    LOG(ERROR) << "Error parsing command-line flags.";
    return -1;
  }

  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1];
    return -1;
  }

  LOG(INFO) << "Graph: [" << tensorflow::flags->graph << "]";
  LOG(INFO) << "Input layer: [" << tensorflow::flags->input_layer << "]";
  LOG(INFO) << "Input shape: [" << tensorflow::flags->input_layer_shape << "]";
  LOG(INFO) << "Input type: [" << tensorflow::flags->input_layer_type << "]";
  LOG(INFO) << "Output layer: [" << tensorflow::flags->output_layer << "]";
  LOG(INFO) << "Num runs: [" << tensorflow::flags->num_runs << "]";
  LOG(INFO) << "Inter-run delay (seconds): [" << tensorflow::flags->run_delay
            << "]";
  LOG(INFO) << "Num threads: [" << tensorflow::flags->num_threads << "]";

  if (!tensorflow::InitializeBenchmark()) {
    return -1;
  }

  // Convert the run_delay string into a timespec.
  const double sleep_seconds =
      std::strtod(tensorflow::flags->run_delay.c_str(), nullptr);
  timespec req;
  req.tv_sec = static_cast<time_t>(sleep_seconds);
  req.tv_nsec = (sleep_seconds - req.tv_sec) * 1000000000;

  LOG(INFO) << "Running benchmark";
  for (int i = 0; i < tensorflow::flags->num_runs; ++i) {
    if (!tensorflow::RunBenchmark()) {
      LOG(INFO) << "Failed on run " << i;
      return -1;
    }

    // If requested, sleep between runs for an arbitrary amount of time.
    // This can be helpful to determine the effect of mobile processor
    // scaling and thermal throttling.
    if (sleep_seconds > 0.0) {
      nanosleep(&req, nullptr);
    }
  }

  tensorflow::g_stats->PrintStepStats();
  return 0;
}
