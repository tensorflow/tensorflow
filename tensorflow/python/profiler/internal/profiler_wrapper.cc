/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/convert/op_stats_to_overview_page.h"
#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_memory_profile.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = ::pybind11;

namespace {

tensorflow::Status ValidateHostPortPair(const std::string& host_port) {
  tensorflow::uint32 port;
  std::vector<absl::string_view> parts = absl::StrSplit(host_port, ':');
  // Must be host:port, port must be a number, host must not contain a '/',
  // host also must not be empty.
  if (parts.size() != 2 || !absl::SimpleAtoi(parts[1], &port) ||
      absl::StrContains(parts[0], "/") || parts[0].empty()) {
    return tensorflow::errors::InvalidArgument(
        "Could not interpret \"", host_port, "\" as a host-port pair.");
  }
  return tensorflow::Status::OK();
}

// Takes profiler options in a py::dict and returns a ProfileOptions.
// This must be called under GIL because it reads Python objects. Reading Python
// objects require GIL because the objects can be mutated by other Python
// threads. In addition, Python objects are reference counted; reading py::dict
// will increase its reference count.
tensorflow::ProfileOptions GetOptionsLocked(const py::dict& opts) {
  tensorflow::ProfileOptions options =
      tensorflow::ProfilerSession::DefaultOptions();
  for (const auto& kw : opts) {
    std::string key = py::cast<std::string>(kw.first);
    if (key == "host_tracer_level") {
      options.set_host_tracer_level(py::cast<int>(kw.second));
      VLOG(1) << "host_tracer_level set to " << options.host_tracer_level();
    } else if (key == "device_tracer_level") {
      options.set_device_tracer_level(py::cast<int>(kw.second));
      VLOG(1) << "device_tracer_level set to " << options.device_tracer_level();
    } else if (key == "python_tracer_level") {
      options.set_python_tracer_level(py::cast<int>(kw.second));
      VLOG(1) << "python_tracer_level set to " << options.python_tracer_level();
    }
  }
  return options;
}

class ProfilerSessionWrapper {
 public:
  void Start(const char* logdir, const py::dict& options) {
    session_ = tensorflow::ProfilerSession::Create(GetOptionsLocked(options));
    logdir_ = logdir;
    tensorflow::MaybeRaiseRegisteredFromStatus(session_->Status());
  }

  py::bytes Stop() {
    tensorflow::string content;
    if (session_ != nullptr) {
      tensorflow::profiler::XSpace xspace;
      tensorflow::Status status = session_->CollectData(&xspace);
      session_.reset();
      tensorflow::profiler::ConvertXSpaceToTraceEventsString(xspace, &content);
      tensorflow::MaybeRaiseRegisteredFromStatus(status);
    }
    // The content is not valid UTF-8, so it must be converted to bytes.
    return py::bytes(content);
  }

  void ExportToTensorBoard() {
    if (!session_ || logdir_.empty()) return;
    tensorflow::profiler::XSpace xspace;
    tensorflow::Status status;
    status = session_->CollectData(&xspace);
    session_.reset();
    status = tensorflow::profiler::ExportToTensorBoard(xspace, logdir_);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
  }

 private:
  std::unique_ptr<tensorflow::ProfilerSession> session_;
  tensorflow::string logdir_;
};

// Converts a pybind list of XSpace paths to a cpp vector.
std::vector<std::string> GetXSpacePaths(const py::list& python_paths) {
  std::vector<std::string> cpp_paths;
  for (py::handle obj : python_paths) {
    cpp_paths.push_back(std::string(py::cast<py::str>(obj)));
  }
  return cpp_paths;
}

}  // namespace

PYBIND11_MODULE(_pywrap_profiler, m) {
  py::class_<ProfilerSessionWrapper> profiler_session_class(m,
                                                            "ProfilerSession");
  profiler_session_class.def(py::init<>())
      .def("start", &ProfilerSessionWrapper::Start)
      .def("stop", &ProfilerSessionWrapper::Stop)
      .def("export_to_tb", &ProfilerSessionWrapper::ExportToTensorBoard);

  m.def("start_server", [](int port) {
    auto profiler_server =
        absl::make_unique<tensorflow::profiler::ProfilerServer>();
    profiler_server->StartProfilerServer(port);
    // Intentionally release profiler server. Should transfer ownership to
    // caller instead.
    profiler_server.release();
  });

  m.def("trace",
        [](const char* service_addr, const char* logdir,
           const char* worker_list, bool include_dataset_ops, int duration_ms,
           int num_tracing_attempts, py::dict options) {
          // Normalize py::dict into a well defined proto.
          tensorflow::ProfileOptions opts = GetOptionsLocked(options);

          tensorflow::Status status = ValidateHostPortPair(service_addr);
          tensorflow::MaybeRaiseRegisteredFromStatus(status);
          opts.set_include_dataset_ops(include_dataset_ops);
          {
            // Release the lock to keep the lock scope to a minimum, and allow
            // other threads to proceed.
            py::gil_scoped_release release;
            status = tensorflow::profiler::Trace(service_addr, logdir,
                                                 worker_list, duration_ms,
                                                 num_tracing_attempts, opts);
          }
          tensorflow::MaybeRaiseRegisteredFromStatus(status);
        });

  m.def("monitor", [](const char* service_addr, int duration_ms,
                      int monitoring_level, bool display_timestamp) {
    tensorflow::Status status = ValidateHostPortPair(service_addr);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    tensorflow::string content;
    {
      // Release the lock to keep the lock scope to a minimum, and allow
      // other threads to proceed.
      py::gil_scoped_release release;
      status = tensorflow::profiler::Monitor(service_addr, duration_ms,
                                             monitoring_level,
                                             display_timestamp, &content);
    }
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    return content;
  });

  m.def("xspace_to_trace_events", [](const py::list& xspace_path_list) {
    std::vector<std::string> xspace_paths = GetXSpacePaths(xspace_path_list);
    if (xspace_paths.size() != 1) {
      LOG(WARNING) << "Trace events tool expects only 1 XSpace path but gets "
                   << xspace_paths.size();
      return py::make_tuple(py::bytes(), py::bool_(false));
    }
    tensorflow::profiler::XSpace xspace;
    tensorflow::Status status = tensorflow::ReadBinaryProto(
        tensorflow::Env::Default(), xspace_paths[0], &xspace);
    if (!status.ok()) {
      LOG(WARNING) << "Could not read XSpace for trace events: "
                   << xspace_paths[0];
      return py::make_tuple(py::bytes(), py::bool_(false));
    }
    tensorflow::string content;
    tensorflow::profiler::ConvertXSpaceToTraceEventsString(xspace, &content);
    return py::make_tuple(py::bytes(content), py::bool_(true));
  });

  m.def("xspace_to_overview_page", [](const py::list& xspace_path_list) {
    std::vector<std::string> xspace_paths = GetXSpacePaths(xspace_path_list);
    tensorflow::profiler::OpStatsOptions options;
    options.generate_kernel_stats_db = true;
    options.generate_op_metrics_db = true;
    options.generate_step_db = true;
    tensorflow::profiler::OpStats combined_op_stats;
    tensorflow::Status status = ConvertMultiXSpacesToCombinedOpStats(
        xspace_paths, options, &combined_op_stats);
    if (!status.ok()) {
      LOG(WARNING) << "Could not generate OpStats for overview page. Error: "
                   << status.error_message();
      return py::make_tuple(py::bytes(), py::bool_(false));
    }
    // TODO(profiler): xspace should tell whether this is sampling mode.
    tensorflow::profiler::OverviewPage overview_page =
        tensorflow::profiler::ConvertOpStatsToOverviewPage(combined_op_stats);
    return py::make_tuple(py::bytes(overview_page.SerializeAsString()),
                          py::bool_(true));
  });

  m.def("xspace_to_input_pipeline", [](const py::list& xspace_path_list) {
    std::vector<std::string> xspace_paths = GetXSpacePaths(xspace_path_list);
    tensorflow::profiler::OpStatsOptions options;
    options.generate_op_metrics_db = true;
    options.generate_step_db = true;
    tensorflow::profiler::OpStats combined_op_stats;
    tensorflow::Status status = ConvertMultiXSpacesToCombinedOpStats(
        xspace_paths, options, &combined_op_stats);
    if (!status.ok()) {
      LOG(WARNING) << "Could not generate OpStats for input pipeline. Error: "
                   << status.error_message();
      return py::make_tuple(py::bytes(), py::bool_(false));
    }
    tensorflow::profiler::InputPipelineAnalysisResult input_pipeline =
        tensorflow::profiler::ConvertOpStatsToInputPipelineAnalysis(
            combined_op_stats);
    return py::make_tuple(py::bytes(input_pipeline.SerializeAsString()),
                          py::bool_(true));
  });

  m.def("xspace_to_tf_stats", [](const py::list& xspace_path_list) {
    std::vector<std::string> xspace_paths = GetXSpacePaths(xspace_path_list);
    tensorflow::profiler::OpStatsOptions options;
    options.generate_op_metrics_db = true;
    options.generate_kernel_stats_db = true;
    tensorflow::profiler::OpStats combined_op_stats;
    tensorflow::Status status = ConvertMultiXSpacesToCombinedOpStats(
        xspace_paths, options, &combined_op_stats);
    if (!status.ok()) {
      LOG(WARNING) << "Could not generate OpStats for tensorflow stats. Error: "
                   << status.error_message();
      return py::make_tuple(py::bytes(), py::bool_(false));
    }
    tensorflow::profiler::TfStatsDatabase tf_stats_db =
        tensorflow::profiler::ConvertOpStatsToTfStats(combined_op_stats);
    return py::make_tuple(py::bytes(tf_stats_db.SerializeAsString()),
                          py::bool_(true));
  });

  m.def("xspace_to_kernel_stats", [](const py::list& xspace_path_list) {
    std::vector<std::string> xspace_paths = GetXSpacePaths(xspace_path_list);
    tensorflow::profiler::OpStatsOptions options;
    options.generate_kernel_stats_db = true;
    tensorflow::profiler::OpStats combined_op_stats;
    tensorflow::Status status = ConvertMultiXSpacesToCombinedOpStats(
        xspace_paths, options, &combined_op_stats);
    if (!status.ok()) {
      LOG(WARNING) << "Could not generate OpStats for kernel stats. Error: "
                   << status.error_message();
      return py::make_tuple(py::bytes(), py::bool_(false));
    }
    return py::make_tuple(
        py::bytes(combined_op_stats.kernel_stats_db().SerializeAsString()),
        py::bool_(true));
  });

  m.def("xspace_to_memory_profile", [](const py::list& xspace_path_list) {
    std::vector<std::string> xspace_paths = GetXSpacePaths(xspace_path_list);
    if (xspace_paths.size() != 1) {
      LOG(WARNING) << "Memory profile tool expects only 1 XSpace path but gets "
                   << xspace_paths.size();
      return py::make_tuple(py::bytes(), py::bool_(false));
    }
    tensorflow::profiler::XSpace xspace;
    tensorflow::Status status = tensorflow::ReadBinaryProto(
        tensorflow::Env::Default(), xspace_paths[0], &xspace);
    if (!status.ok()) {
      LOG(WARNING) << "Could not read XSpace for memory profile: "
                   << xspace_paths[0];
      return py::make_tuple(py::bytes(), py::bool_(false));
    }
    std::string json_output;
    tensorflow::profiler::ConvertXSpaceToMemoryProfileJson(xspace,
                                                           &json_output);
    return py::make_tuple(py::bytes(json_output), py::bool_(true));
  });
};
