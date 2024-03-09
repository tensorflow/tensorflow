/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/profiler/internal/profiler_pywrap_impl.h"

#include <string>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/variant.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/xplane_to_tools_data.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include "tsl/profiler/convert/xplane_to_trace_events.h"
#include "tsl/profiler/rpc/client/capture_profile.h"
#include "tsl/profiler/utils/session_manager.h"

namespace tensorflow {
namespace profiler {
namespace pywrap {

using tsl::profiler::GetRemoteSessionManagerOptionsLocked;
using tsl::profiler::ValidateHostPortPair;

tensorflow::Status Trace(
    const char* service_addr, const char* logdir, const char* worker_list,
    bool include_dataset_ops, int duration_ms, int num_tracing_attempts,
    const absl::flat_hash_map<std::string, std::variant<int, std::string>>&
        options) {
  return tsl::profiler::CaptureRemoteTrace(service_addr, logdir, worker_list,
                                           include_dataset_ops, duration_ms,
                                           num_tracing_attempts, options);
}

tensorflow::Status Monitor(const char* service_addr, int duration_ms,
                           int monitoring_level, bool display_timestamp,
                           tensorflow::string* result) {
  TF_RETURN_IF_ERROR(ValidateHostPortPair(service_addr));
  {
    TF_RETURN_IF_ERROR(tsl::profiler::Monitor(service_addr, duration_ms,
                                              monitoring_level,
                                              display_timestamp, result));
  }
  return absl::OkStatus();
}

tensorflow::Status ProfilerSessionWrapper::Start(
    const char* logdir,
    const absl::flat_hash_map<std::string, std::variant<int, std::string>>&
        options) {
  auto opts = GetRemoteSessionManagerOptionsLocked(logdir, options);
  session_ = tensorflow::ProfilerSession::Create(opts.profiler_options());
  logdir_ = logdir;
  return session_->Status();
}

tensorflow::Status ProfilerSessionWrapper::Stop(tensorflow::string* result) {
  if (session_ != nullptr) {
    tensorflow::profiler::XSpace xspace;
    tensorflow::Status status = session_->CollectData(&xspace);
    session_.reset();
    tsl::profiler::ConvertXSpaceToTraceEventsString(xspace, result);
    TF_RETURN_IF_ERROR(status);
  }
  return absl::OkStatus();
}

tensorflow::Status ProfilerSessionWrapper::ExportToTensorBoard() {
  if (!session_ || logdir_.empty()) {
    return absl::OkStatus();
  }
  tensorflow::profiler::XSpace xspace;
  tensorflow::Status status;
  status = session_->CollectData(&xspace);
  session_.reset();
  status = tsl::profiler::ExportToTensorBoard(xspace, logdir_);
  return status;
}

}  // namespace pywrap
}  // namespace profiler
}  // namespace tensorflow
