/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/benchmark/benchmark_model.h"

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
#include "tensorflow/core/util/reporter.h"
#include "tensorflow/core/util/stat_summarizer.h"

namespace tensorflow {
namespace benchmark_model {

Status InitializeSession(int num_threads, const string& graph,
                         std::unique_ptr<Session>* session,
                         std::unique_ptr<StatSummarizer>* stats) {
  LOG(INFO) << "Loading TensorFlow.";

  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& config = options.config;
  if (num_threads > 0) {
    config.set_intra_op_parallelism_threads(num_threads);
  }
  LOG(INFO) << "Got config, " << config.device_count_size() << " devices";

  session->reset(tensorflow::NewSession(options));
  tensorflow::GraphDef tensorflow_graph;
  Status s = ReadBinaryProto(Env::Default(), graph, &tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
    return s;
  }

  stats->reset(new tensorflow::StatSummarizer(tensorflow_graph));

  s = (*session)->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Session: " << s;
    return s;
  }

  // Clear the proto to save memory space.
  tensorflow_graph.Clear();
  return Status::OK();
}

Status RunBenchmark(const std::vector<InputLayerInfo>& inputs,
                    const std::vector<string>& outputs, Session* session,
                    StatSummarizer* stats) {
  std::vector<std::pair<string, tensorflow::Tensor> > input_tensors;
  for (const InputLayerInfo& input : inputs) {
    Tensor input_tensor(input.data_type, input.shape);
    switch (input.data_type) {
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
      case DT_UINT8: {
        auto int_tensor = input_tensor.flat<uint8>();
        int_tensor = int_tensor.constant(0.0);
        break;
      }
      default:
        LOG(FATAL) << "Unsupported input type: " << input.data_type;
    }
    input_tensors.push_back({input.name, input_tensor});
  }

  std::vector<tensorflow::Tensor> output_tensors;

  tensorflow::Status s;

  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  RunMetadata run_metadata;

  s = session->Run(run_options, input_tensors, outputs, {}, &output_tensors,
                   &run_metadata);

  if (!s.ok()) {
    LOG(ERROR) << "Error during inference: " << s;
  }

  assert(run_metadata.has_step_stats());

  const StepStats& step_stats = run_metadata.step_stats();

  stats->ProcessStepStats(step_stats);

  return s;
}

Status TimeMultipleRuns(double sleep_seconds, int num_runs,
                        const std::vector<InputLayerInfo>& inputs,
                        const std::vector<string>& outputs, Session* session,
                        StatSummarizer* stats) {
  // Convert the run_delay string into a timespec.
  timespec req;
  req.tv_sec = static_cast<time_t>(sleep_seconds);
  req.tv_nsec = (sleep_seconds - req.tv_sec) * 1000000000;

  LOG(INFO) << "Running benchmark";
  for (int i = 0; i < num_runs; ++i) {
    Status run_status = RunBenchmark(inputs, outputs, session, stats);
    if (!run_status.ok()) {
      LOG(INFO) << "Failed on run " << i;
      return run_status;
    }

    // If requested, sleep between runs for an arbitrary amount of time.
    // This can be helpful to determine the effect of mobile processor
    // scaling and thermal throttling.
    if (sleep_seconds > 0.0) {
      nanosleep(&req, nullptr);
    }
  }

  return Status::OK();
}

int Main(int argc, char** argv) {
  string graph = "/data/local/tmp/tensorflow_inception_graph.pb";
  string input_layer_string = "input:0";
  string input_layer_shape_string = "1,224,224,3";
  string input_layer_type_string = "float";
  string output_layer_string = "output:0";
  int num_runs = 50;
  string run_delay = "-1.0";
  int num_threads = -1;
  string benchmark_name = "";
  string output_prefix = "";
  bool show_sizes = false;

  std::vector<Flag> flag_list = {
      Flag("graph", &graph, "graph file name"),
      Flag("input_layer", &input_layer_string, "input layer names"),
      Flag("input_layer_shape", &input_layer_shape_string, "input layer shape"),
      Flag("input_layer_type", &input_layer_type_string, "input layer type"),
      Flag("output_layer", &output_layer_string, "output layer name"),
      Flag("num_runs", &num_runs, "number of runs"),
      Flag("run_delay", &run_delay, "delay between runs in seconds"),
      Flag("num_threads", &num_threads, "number of threads"),
      Flag("benchmark_name", &benchmark_name, "benchmark name"),
      Flag("output_prefix", &output_prefix, "benchmark output prefix"),
      Flag("show_sizes", &show_sizes, "whether to show sizes"),
  };
  string usage = Flags::Usage(argv[0], flag_list);
  const bool parse_result = Flags::Parse(&argc, argv, flag_list);

  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  std::vector<string> input_layers = str_util::Split(input_layer_string, ',');
  std::vector<string> input_layer_shapes =
      str_util::Split(input_layer_shape_string, ':');
  std::vector<string> input_layer_types =
      str_util::Split(input_layer_type_string, ',');
  std::vector<string> output_layers = str_util::Split(output_layer_string, ',');
  if ((input_layers.size() != input_layer_shapes.size()) ||
      (input_layers.size() != input_layer_types.size())) {
    LOG(ERROR) << "There must be the same number of items in --input_layer,"
               << " --input_layer_shape, and --input_layer_type, for example"
               << " --input_layer=input1,input2 --input_layer_type=float,float "
               << " --input_layer_shape=1,224,224,4:1,20";
    LOG(ERROR) << "--input_layer=" << input_layer_string << " ("
               << input_layers.size() << " items)";
    LOG(ERROR) << "--input_layer_type=" << input_layer_type_string << " ("
               << input_layer_types.size() << " items)";
    LOG(ERROR) << "--input_layer_shape=" << input_layer_shape_string << " ("
               << input_layer_shapes.size() << " items)";
    return -1;
  }
  const size_t inputs_count = input_layers.size();

  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  LOG(INFO) << "Graph: [" << graph << "]";
  LOG(INFO) << "Input layers: [" << input_layer_string << "]";
  LOG(INFO) << "Input shapes: [" << input_layer_shape_string << "]";
  LOG(INFO) << "Input types: [" << input_layer_type_string << "]";
  LOG(INFO) << "Output layers: [" << output_layer_string << "]";
  LOG(INFO) << "Num runs: [" << num_runs << "]";
  LOG(INFO) << "Inter-run delay (seconds): [" << run_delay << "]";
  LOG(INFO) << "Num threads: [" << num_threads << "]";
  LOG(INFO) << "Benchmark name: [" << benchmark_name << "]";
  LOG(INFO) << "Output prefix: [" << output_prefix << "]";
  LOG(INFO) << "Show sizes: [" << show_sizes << "]";

  std::unique_ptr<Session> session;
  std::unique_ptr<StatSummarizer> stats;
  Status initialize_status =
      InitializeSession(num_threads, graph, &session, &stats);
  if (!initialize_status.ok()) {
    return -1;
  }

  const double sleep_seconds = std::strtod(run_delay.c_str(), nullptr);

  std::vector<InputLayerInfo> inputs;
  for (int n = 0; n < inputs_count; ++n) {
    InputLayerInfo input;
    CHECK(DataTypeFromString(input_layer_types[n], &input.data_type))
        << input_layer_types[n] << " was an invalid type";
    std::vector<int32> sizes;
    CHECK(str_util::SplitAndParseAsInts(input_layer_shapes[n], ',', &sizes))
        << "Incorrect size string specified: " << input_layer_shapes[n];
    for (int i = 0; i < sizes.size(); ++i) {
      input.shape.AddDim(sizes[i]);
    }
    input.name = input_layers[n];
    inputs.push_back(input);
  }

  const int64 start_time = Env::Default()->NowMicros();
  Status time_status =
      TimeMultipleRuns(sleep_seconds, num_runs, inputs, output_layers,
                       session.get(), stats.get());
  const int64 end_time = Env::Default()->NowMicros();
  const double wall_time = (end_time - start_time) / 1000000.0;

  if (!time_status.ok()) {
    LOG(ERROR) << "Timing failed with " << time_status;
    return -1;
  }

  stats->PrintStepStats();

  if (show_sizes) {
    stats->PrintOutputs();
  }

  if (!benchmark_name.empty() && !output_prefix.empty()) {
    // Compute the total number of values per input.
    int64 total_size = inputs[0].shape.num_elements();

    // Throughput in MB/s
    const double throughput = DataTypeSize(inputs[0].data_type) * total_size *
                              num_runs / static_cast<double>(wall_time) /
                              (1024 * 1024);

    // Report the stats.
    TestReporter reporter(output_prefix, benchmark_name);
    reporter.Initialize();
    reporter.Benchmark(num_runs, -1.0, wall_time, throughput);
    reporter.Close();
  }

  return 0;
}

}  // namespace benchmark_model
}  // namespace tensorflow
