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
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/reporter.h"
#include "tensorflow/core/util/stat_summarizer.h"

namespace tensorflow {
namespace benchmark_model {

Status InitializeSession(int num_threads, const string& graph,
                         std::unique_ptr<Session>* session,
                         std::unique_ptr<GraphDef>* graph_def) {
  LOG(INFO) << "Loading TensorFlow.";

  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& config = options.config;
  if (num_threads > 0) {
    config.set_intra_op_parallelism_threads(num_threads);
  }
  LOG(INFO) << "Got config, " << config.device_count_size() << " devices";

  session->reset(tensorflow::NewSession(options));
  graph_def->reset(new GraphDef());
  tensorflow::GraphDef tensorflow_graph;
  Status s = ReadBinaryProto(Env::Default(), graph, graph_def->get());
  if (!s.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
    return s;
  }

  s = (*session)->Create(*(graph_def->get()));
  if (!s.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Session: " << s;
    return s;
  }

  return Status::OK();
}

template <class T>
void InitializeTensor(const std::vector<float>& initialization_values,
                      Tensor* input_tensor) {
  auto type_tensor = input_tensor->flat<T>();
  type_tensor = type_tensor.constant(0);
  if (!initialization_values.empty()) {
    for (int i = 0; i < initialization_values.size(); ++i) {
      type_tensor(i) = static_cast<T>(initialization_values[i]);
    }
  }
}

void CreateTensorsFromInputInfo(
    const std::vector<InputLayerInfo>& inputs,
    std::vector<std::pair<string, tensorflow::Tensor> >* input_tensors) {
  for (const InputLayerInfo& input : inputs) {
    Tensor input_tensor(input.data_type, input.shape);
    switch (input.data_type) {
      case DT_INT32: {
        InitializeTensor<int32>(input.initialization_values, &input_tensor);
        break;
      }
      case DT_FLOAT: {
        InitializeTensor<float>(input.initialization_values, &input_tensor);
        break;
      }
      case DT_QUINT8: {
        InitializeTensor<quint8>(input.initialization_values, &input_tensor);
        break;
      }
      case DT_UINT8: {
        InitializeTensor<uint8>(input.initialization_values, &input_tensor);
        break;
      }
      case DT_BOOL: {
        InitializeTensor<bool>(input.initialization_values, &input_tensor);
        break;
      }
      case DT_STRING: {
        if (!input.initialization_values.empty()) {
          LOG(FATAL) << "Initialization values are not supported for strings";
        }
        auto type_tensor = input_tensor.flat<string>();
        type_tensor = type_tensor.constant("");
        break;
      }
      default:
        LOG(FATAL) << "Unsupported input type: "
                   << DataTypeString(input.data_type);
    }
    input_tensors->push_back({input.name, input_tensor});
  }
}

Status GetOutputShapes(const std::vector<InputLayerInfo>& inputs,
                       const std::set<string>& wanted_shapes, Session* session,
                       std::unordered_map<string, TensorShape>* node_shapes) {
  std::vector<std::pair<string, tensorflow::Tensor> > input_tensors;
  CreateTensorsFromInputInfo(inputs, &input_tensors);
  std::vector<tensorflow::Tensor> output_tensors;
  std::vector<string> output_tensor_names(wanted_shapes.begin(),
                                          wanted_shapes.end());
  TF_RETURN_IF_ERROR(
      session->Run(input_tensors, output_tensor_names, {}, &output_tensors));
  CHECK_EQ(output_tensors.size(), output_tensor_names.size());
  for (int i = 0; i < output_tensor_names.size(); ++i) {
    const string& wanted_shape_name = output_tensor_names[i];
    const TensorShape& found_shape = output_tensors[i].shape();
    (*node_shapes)[wanted_shape_name] = found_shape;
  }
  return Status::OK();
}

Status CalculateFlops(const GraphDef& graph,
                      const std::vector<InputLayerInfo>& inputs,
                      Session* session, int64* total_flops,
                      std::unordered_map<string, int64>* flops_by_op) {
  std::unordered_set<string> floppable_ops = {
      "Conv2D", "MatMul", "QuantizedConv2D", "QuantizedMatMul"};

  std::set<string> wanted_shapes;
  for (const NodeDef& node : graph.node()) {
    if (floppable_ops.count(node.op())) {
      for (const string& input : node.input()) {
        wanted_shapes.insert(input);
      }
      wanted_shapes.insert(node.name());
    }
  }
  std::unordered_map<string, TensorShape> found_shapes;
  TF_RETURN_IF_ERROR(
      GetOutputShapes(inputs, wanted_shapes, session, &found_shapes));

  *total_flops = 0;
  for (const NodeDef& node : graph.node()) {
    if (floppable_ops.count(node.op())) {
      int64 current_flops = 0;
      // This is a very crude approximation to FLOPs that only looks at a few
      // op types that commonly form the bulk of the computation for many
      // models. It's included here because getting even an approximate value
      // for FLOPs is still very useful for estimating utilization, versus a
      // device's theoretical maximum FLOPs/second.
      if ((node.op() == "Conv2D") || (node.op() == "QuantizedConv2D")) {
        const TensorShape& filter_shape = found_shapes[node.input(1)];
        const TensorShape& output_shape = found_shapes[node.name()];
        int64 filter_height = filter_shape.dim_size(0);
        int64 filter_width = filter_shape.dim_size(1);
        int64 filter_in_depth = filter_shape.dim_size(2);
        int64 output_count = output_shape.num_elements();
        current_flops =
            output_count * filter_in_depth * filter_height * filter_width * 2;
      } else if ((node.op() == "MatMul") || (node.op() == "QuantizedMatMul")) {
        const bool transpose_a = node.attr().at("transpose_a").b();
        const TensorShape& a_shape = found_shapes[node.input(0)];
        const TensorShape& output_shape = found_shapes[node.name()];
        int64 k;
        if (transpose_a) {
          k = a_shape.dim_size(0);
        } else {
          k = a_shape.dim_size(1);
        }
        int64 output_count = output_shape.num_elements();
        current_flops = k * output_count * 2;
      }
      (*flops_by_op)[node.op()] += current_flops;
      *total_flops += current_flops;
    }
  }
  return Status::OK();
}

Status RunBenchmark(const std::vector<InputLayerInfo>& inputs,
                    const std::vector<string>& outputs, Session* session,
                    StatSummarizer* stats, int64* inference_time_us) {
  std::vector<std::pair<string, tensorflow::Tensor> > input_tensors;
  CreateTensorsFromInputInfo(inputs, &input_tensors);

  std::vector<tensorflow::Tensor> output_tensors;

  tensorflow::Status s;

  RunOptions run_options;
  if (stats != nullptr) {
    run_options.set_trace_level(RunOptions::FULL_TRACE);
  }

  RunMetadata run_metadata;
  const int64 start_time = Env::Default()->NowMicros();
  s = session->Run(run_options, input_tensors, outputs, {}, &output_tensors,
                   &run_metadata);
  const int64 end_time = Env::Default()->NowMicros();
  *inference_time_us = end_time - start_time;

  if (!s.ok()) {
    LOG(ERROR) << "Error during inference: " << s;
    return s;
  }

  if (stats != nullptr) {
    assert(run_metadata.has_step_stats());
    const StepStats& step_stats = run_metadata.step_stats();
    stats->ProcessStepStats(step_stats);
  }

  return s;
}

Status TimeMultipleRuns(double sleep_seconds, int num_runs,
                        const std::vector<InputLayerInfo>& inputs,
                        const std::vector<string>& outputs, Session* session,
                        StatSummarizer* stats, int64* total_time_us) {
  // Convert the run_delay string into a timespec.
  timespec req;
  req.tv_sec = static_cast<time_t>(sleep_seconds);
  req.tv_nsec = (sleep_seconds - req.tv_sec) * 1000000000;

  *total_time_us = 0;

  LOG(INFO) << "Running benchmark for " << num_runs << " iterations "
            << (stats != nullptr ? "with" : "without")
            << " detailed stat logging:";

  Stat<int64> stat;
  for (int i = 0; i < num_runs; ++i) {
    int64 time;
    Status run_status = RunBenchmark(inputs, outputs, session, stats, &time);
    stat.UpdateStat(time);
    *total_time_us += time;
    if (!run_status.ok()) {
      LOG(INFO) << "Failed on run " << i;
      return run_status;
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
  std::stringstream stream;
  stat.OutputToStream(&stream);
  LOG(INFO) << stream.str() << std::endl;

  return Status::OK();
}

int Main(int argc, char** argv) {
  string graph = "/data/local/tmp/tensorflow_inception_graph.pb";
  string input_layer_string = "input:0";
  string input_layer_shape_string = "1,224,224,3";
  string input_layer_type_string = "float";
  string input_layer_values_string = "";
  string output_layer_string = "output:0";
  int num_runs = 50;
  string run_delay = "-1.0";
  int num_threads = -1;
  string benchmark_name = "";
  string output_prefix = "";
  bool show_sizes = false;
  bool show_run_order = true;
  int run_order_limit = 0;
  bool show_time = true;
  int time_limit = 10;
  bool show_memory = true;
  int memory_limit = 10;
  bool show_type = true;
  bool show_summary = true;
  bool show_flops = false;
  int warmup_runs = 2;

  std::vector<Flag> flag_list = {
      Flag("graph", &graph, "graph file name"),
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
      Flag("show_sizes", &show_sizes, "whether to show sizes"),
      Flag("show_run_order", &show_run_order,
           "whether to list stats by run order"),
      Flag("run_order_limit", &run_order_limit,
           "how many items to show by run order"),
      Flag("show_time", &show_time, "whether to list stats by time taken"),
      Flag("time_limit", &time_limit, "how many items to show by time taken"),
      Flag("show_memory", &show_memory, "whether to list stats by memory used"),
      Flag("memory_limit", &memory_limit,
           "how many items to show by memory used"),
      Flag("show_type", &show_time, "whether to list stats by op type"),
      Flag("show_summary", &show_time,
           "whether to show a summary of the stats"),
      Flag("show_flops", &show_flops, "whether to estimate the model's FLOPs"),
      Flag("warmup_runs", &warmup_runs, "how many runs to initialize model"),
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
  std::vector<string> input_layer_values =
      str_util::Split(input_layer_values_string, ':');
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
  LOG(INFO) << "Warmup runs: [" << warmup_runs << "]";

  std::unique_ptr<Session> session;
  std::unique_ptr<StatSummarizer> stats;
  std::unique_ptr<GraphDef> graph_def;
  Status initialize_status =
      InitializeSession(num_threads, graph, &session, &graph_def);
  if (!initialize_status.ok()) {
    return -1;
  }

  StatSummarizerOptions stats_options;
  stats_options.show_run_order = show_run_order;
  stats_options.run_order_limit = run_order_limit;
  stats_options.show_time = show_time;
  stats_options.time_limit = time_limit;
  stats_options.show_memory = show_memory;
  stats_options.memory_limit = memory_limit;
  stats_options.show_type = show_type;
  stats_options.show_summary = show_summary;
  stats.reset(new tensorflow::StatSummarizer(stats_options));

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
    if (n < input_layer_values.size()) {
      CHECK(str_util::SplitAndParseAsFloats(input_layer_values[n], ',',
                                            &input.initialization_values))
          << "Incorrect initialization values string specified: "
          << input_layer_values[n];
    }
    inputs.push_back(input);
  }

  // If requested, run through the graph first to preinitialize everything
  // before the benchmarking runs.
  int64 warmup_time_us = 0;
  if (warmup_runs > 0) {
    Status warmup_time_status =
        TimeMultipleRuns(sleep_seconds, warmup_runs, inputs, output_layers,
                         session.get(), nullptr, &warmup_time_us);
    if (!warmup_time_status.ok()) {
      LOG(ERROR) << "Timing failed with " << warmup_time_status;
      return -1;
    }
  }

  // Capture overall inference time without stat logging overhead. This is the
  // timing data that can be compared to other libaries.
  int64 no_stat_time_us = 0;
  Status no_stat_time_status =
      TimeMultipleRuns(sleep_seconds, num_runs, inputs, output_layers,
                       session.get(), nullptr, &no_stat_time_us);
  const double no_stat_wall_time = no_stat_time_us / 1000000.0;
  if (!no_stat_time_status.ok()) {
    LOG(ERROR) << "Timing failed with " << no_stat_time_status;
    return -1;
  }

  // Run again to gather detailed log stats to get a better idea of where
  // relative time is going within the graph.
  int64 stat_time_us = 0;
  Status stat_time_status =
      TimeMultipleRuns(sleep_seconds, num_runs, inputs, output_layers,
                       session.get(), stats.get(), &stat_time_us);
  if (!stat_time_status.ok()) {
    LOG(ERROR) << "Timing failed with " << stat_time_status;
    return -1;
  }

  LOG(INFO) << "Average inference timings in us: "
            << "Warmup: "
            << (warmup_runs > 0 ? warmup_time_us / warmup_runs : 0) << ", "
            << "no stats: " << no_stat_time_us / num_runs << ", "
            << "with stats: " << stat_time_us / num_runs;

  stats->PrintStepStats();

  if (show_sizes) {
    stats->PrintOutputs();
  }

  if (show_flops) {
    int64 total_flops;
    std::unordered_map<string, int64> flops_by_op;
    Status flop_status = CalculateFlops(*graph_def, inputs, session.get(),
                                        &total_flops, &flops_by_op);
    if (!flop_status.ok()) {
      LOG(ERROR) << "FLOPs calculation failed with " << flop_status;
      return -1;
    }
    string pretty_flops;
    if (total_flops < 1000) {
      pretty_flops = strings::StrCat(total_flops, " FLOPs");
    } else if (total_flops < (1000 * 1000)) {
      const float rounded_flops = (total_flops / 1000.0f);
      pretty_flops = strings::StrCat(rounded_flops, "k FLOPs");
    } else if (total_flops < (1000 * 1000 * 1000)) {
      const float rounded_flops = round(total_flops / 1000.0f) / 1000.0f;
      pretty_flops = strings::StrCat(rounded_flops, " million FLOPs");
    } else {
      const float rounded_flops =
          round(total_flops / (1000.0f * 1000.0f)) / 1000.0f;
      pretty_flops = strings::StrCat(rounded_flops, " billion FLOPs");
    }
    LOG(INFO) << "FLOPs estimate: " << strings::HumanReadableNum(total_flops);
    const double mean_run_time = no_stat_wall_time / num_runs;
    LOG(INFO) << "FLOPs/second: "
              << strings::HumanReadableNum(
                     static_cast<int64>(total_flops / mean_run_time));
  }

  if (!benchmark_name.empty() && !output_prefix.empty()) {
    // Compute the total number of values per input.
    int64 total_size = inputs[0].shape.num_elements();

    // Throughput in MB/s
    const double throughput =
        DataTypeSize(inputs[0].data_type) * total_size * num_runs /
        static_cast<double>(no_stat_wall_time) / (1024 * 1024);

    // Report the stats.
    TestReporter reporter(output_prefix, benchmark_name);
    TF_QCHECK_OK(reporter.Initialize());
    TF_QCHECK_OK(
        reporter.Benchmark(num_runs, -1.0, no_stat_wall_time, throughput));
    TF_QCHECK_OK(reporter.Close());

    std::map<string, int64> node_type_map_count;
    std::map<string, int64> node_type_map_time;
    std::map<string, int64> node_type_map_memory;

    int64 accumulated_us;
    stats->ComputeStatsByType(&node_type_map_count, &node_type_map_time,
                              &node_type_map_memory, &accumulated_us);
    for (const auto& time : node_type_map_time) {
      std::stringstream stream;
      stream << benchmark_name << "_" << time.first;
      TestReporter node_reporter(output_prefix, stream.str());

      LOG(INFO) << "Outputting: [" << time.first << "]";

      TF_QCHECK_OK(node_reporter.Initialize());
      TF_QCHECK_OK(node_reporter.Benchmark(
          num_runs, -1.0, (time.second * num_runs) / 1000000.0f, -1.0));
      TF_QCHECK_OK(node_reporter.Close());
    }
  }

  return 0;
}

}  // namespace benchmark_model
}  // namespace tensorflow
