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

Status RunBenchmark(DataType input_data_type, TensorShape input_shape,
                    const string& input_layer, const string output_layer,
                    Session* session, StatSummarizer* stats) {
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
      LOG(FATAL) << "Unsupported input type: " << input_data_type;
  }

  std::vector<std::pair<string, tensorflow::Tensor> > input_tensors(
      {{input_layer, input_tensor}});

  std::vector<tensorflow::Tensor> output_tensors;
  std::vector<string> output_names({output_layer});

  tensorflow::Status s;

  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  RunMetadata run_metadata;

  s = session->Run(run_options, input_tensors, output_names, {},
                   &output_tensors, &run_metadata);

  if (!s.ok()) {
    LOG(ERROR) << "Error during inference: " << s;
  }

  assert(run_metadata.has_step_stats());

  const StepStats& step_stats = run_metadata.step_stats();

  stats->ProcessStepStats(step_stats);

  return s;
}

Status TimeMultipleRuns(double sleep_seconds, int num_runs,
                        DataType input_data_type, TensorShape input_shape,
                        const string& input_layer, const string output_layer,
                        Session* session, StatSummarizer* stats) {
  // Convert the run_delay string into a timespec.
  timespec req;
  req.tv_sec = static_cast<time_t>(sleep_seconds);
  req.tv_nsec = (sleep_seconds - req.tv_sec) * 1000000000;

  LOG(INFO) << "Running benchmark";
  for (int i = 0; i < num_runs; ++i) {
    Status run_status = RunBenchmark(input_data_type, input_shape, input_layer,
                                     output_layer, session, stats);
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
  string input_layer = "input:0";
  string input_layer_shape = "1,224,224,3";
  string input_layer_type = "float";
  string output_layer = "output:0";
  int num_runs = 50;
  string run_delay = "-1.0";
  int num_threads = -1;
  string benchmark_name = "";
  string output_prefix = "";

  const bool parse_result = ParseFlags(
      &argc, argv, {
                       Flag("graph", &graph),                          //
                       Flag("input_layer", &input_layer),              //
                       Flag("input_layer_shape", &input_layer_shape),  //
                       Flag("input_layer_type", &input_layer_type),    //
                       Flag("output_layer", &output_layer),            //
                       Flag("num_runs", &num_runs),                    //
                       Flag("run_delay", &run_delay),                  //
                       Flag("num_threads", &num_threads),              //
                       Flag("benchmark_name", &benchmark_name),        //
                       Flag("output_prefix", &output_prefix),          //
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

  LOG(INFO) << "Graph: [" << graph << "]";
  LOG(INFO) << "Input layer: [" << input_layer << "]";
  LOG(INFO) << "Input shape: [" << input_layer_shape << "]";
  LOG(INFO) << "Input type: [" << input_layer_type << "]";
  LOG(INFO) << "Output layer: [" << output_layer << "]";
  LOG(INFO) << "Num runs: [" << num_runs << "]";
  LOG(INFO) << "Inter-run delay (seconds): [" << run_delay << "]";
  LOG(INFO) << "Num threads: [" << num_threads << "]";
  LOG(INFO) << "Benchmark name: [" << benchmark_name << "]";
  LOG(INFO) << "Output prefix: [" << output_prefix << "]";

  std::unique_ptr<Session> session;
  std::unique_ptr<StatSummarizer> stats;
  Status initialize_status =
      InitializeSession(num_threads, graph, &session, &stats);
  if (!initialize_status.ok()) {
    return -1;
  }

  const double sleep_seconds = std::strtod(run_delay.c_str(), nullptr);
  DataType input_data_type;
  CHECK(DataTypeFromString(input_layer_type, &input_data_type))
      << input_layer_type << " was an invalid type";
  std::vector<int32> sizes;
  CHECK(str_util::SplitAndParseAsInts(input_layer_shape, ',', &sizes))
      << "Incorrect size string specified: " << input_layer_shape;
  TensorShape input_shape;
  for (int i = 0; i < sizes.size(); ++i) {
    input_shape.AddDim(sizes[i]);
  }

  const int64 start_time = Env::Default()->NowMicros();
  Status time_status =
      TimeMultipleRuns(sleep_seconds, num_runs, input_data_type, input_shape,
                       input_layer, output_layer, session.get(), stats.get());
  const int64 end_time = Env::Default()->NowMicros();
  const double wall_time = (end_time - start_time) / 1000000.0;

  if (!time_status.ok()) {
    LOG(ERROR) << "Timing failed with " << time_status;
    return -1;
  }

  stats->PrintStepStats();

  if (!benchmark_name.empty() && !output_prefix.empty()) {
    // Compute the total number of values per input.
    int64 total_size = 1;
    for (int32 size : sizes) {
      total_size *= size;
    }

    // Throughput in MB/s
    const double throughput = DataTypeSize(input_data_type) * total_size *
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
