/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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

// Usage: capture_tpu_profile --service_addr="localhost:8466" --logdir=/tmp/log
//
// Initiates a TPU profiling on the TPUProfiler service at service_addr,
// receives and dumps the profile data to a tensorboard log directory.

#include "tensorflow/contrib/tpu/profiler/version.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"
#include "tensorflow/core/util/command_line_flags.h"

int main(int argc, char** argv) {
  tensorflow::string FLAGS_service_addr;
  tensorflow::string FLAGS_logdir;
  tensorflow::string FLAGS_workers_list;
  int FLAGS_duration_ms = 0;
  int FLAGS_num_tracing_attempts = 3;
  bool FLAGS_include_dataset_ops = true;
  int FLAGS_monitoring_level = 0;
  bool FLAGS_timestamp = false;
  int FLAGS_num_queries = 100;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("service_addr", &FLAGS_service_addr,
                       "Address of TPU profiler service e.g. localhost:8466"),
      tensorflow::Flag("workers_list", &FLAGS_workers_list,
                       "The list of worker TPUs that we are about to profile "
                       "in the current session."),
      tensorflow::Flag("logdir", &FLAGS_logdir,
                       "Path of TensorBoard log directory e.g. /tmp/tb_log, "
                       "gs://tb_bucket"),
      tensorflow::Flag(
          "duration_ms", &FLAGS_duration_ms,
          "Duration of tracing or monitoring in ms. Default is 2000ms for "
          "tracing and 1000ms for monitoring."),
      tensorflow::Flag("num_tracing_attempts", &FLAGS_num_tracing_attempts,
                       "Automatically retry N times when no trace event "
                       "is collected. Default is 3."),
      tensorflow::Flag("include_dataset_ops", &FLAGS_include_dataset_ops,
                       "Set to false to profile longer TPU device traces."),
      tensorflow::Flag("monitoring_level", &FLAGS_monitoring_level,
                       "Choose a monitoring level between 1 and 2 to monitor "
                       "your TPU job continuously. Level 2 is more verbose "
                       "than level 1 and shows more metrics."),
      tensorflow::Flag("timestamp", &FLAGS_timestamp,
                       "Set to true to display timestamp in monitoring "
                       "results."),
      tensorflow::Flag("num_queries", &FLAGS_num_queries,
                       "This script will run monitoring for num_queries before "
                       "it stops.")};

  std::cout << "Welcome to the Cloud TPU Profiler v" << TPU_PROFILER_VERSION
            << std::endl;

  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok || FLAGS_service_addr.empty() ||
      (FLAGS_logdir.empty() && FLAGS_monitoring_level == 0)) {
    // Fail if flags are not parsed correctly or service_addr not provided.
    // Also, fail if neither logdir is provided (required for tracing) nor
    // monitoring level is provided (required for monitoring).
    std::cout << usage.c_str() << std::endl;
    return 2;
  }
  if (FLAGS_monitoring_level < 0 || FLAGS_monitoring_level > 2) {
    // Invalid monitoring level.
    std::cout << usage.c_str() << std::endl;
    return 2;
  }
  tensorflow::Status status;
  status =
      tensorflow::profiler::client::ValidateHostPortPair(FLAGS_service_addr);
  if (!status.ok()) {
    std::cout << status.error_message() << std::endl;
    std::cout << usage.c_str() << std::endl;
    return 2;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  // Sets the minimum duration_ms, tracing attempts and num queries.
  int duration_ms = std::max(FLAGS_duration_ms, 0);
  if (duration_ms == 0) {
    // If profiling duration was not set by user or set to a negative value, we
    // set it to default values of 2000ms for tracing and 1000ms for monitoring.
    duration_ms = FLAGS_monitoring_level == 0 ? 2000 : 1000;
  }
  int num_tracing_attempts = std::max(FLAGS_num_tracing_attempts, 1);
  int num_queries = std::max(FLAGS_num_queries, 1);

  if (FLAGS_monitoring_level != 0) {
    std::cout << "Since monitoring level is provided, profile "
              << FLAGS_service_addr << " for " << duration_ms
              << "ms and show metrics for " << num_queries << " time(s)."
              << std::endl;
    tensorflow::profiler::client::StartMonitoring(
        FLAGS_service_addr, duration_ms, FLAGS_monitoring_level,
        FLAGS_timestamp, num_queries);
  } else {
    status = tensorflow::profiler::client::StartTracing(
        FLAGS_service_addr, FLAGS_logdir, FLAGS_workers_list,
        FLAGS_include_dataset_ops, duration_ms, num_tracing_attempts);
    if (!status.ok() && status.code() != tensorflow::error::Code::UNAVAILABLE) {
      std::cout << status.error_message() << std::endl;
      std::cout << usage.c_str() << std::endl;
      return 2;
    }
  }
  return 0;
}
