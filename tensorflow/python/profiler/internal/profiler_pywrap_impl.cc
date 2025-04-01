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
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/convert/xplane_to_trace_events.h"
#include "xla/tsl/profiler/rpc/client/capture_profile.h"
#include "xla/tsl/profiler/utils/session_manager.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/xplane_to_tools_data.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace pywrap {

using tsl::profiler::GetRemoteSessionManagerOptionsLocked;
using tsl::profiler::ValidateHostPortPair;

absl::Status ProfilerSessionWrapper::Start(
    const char* logdir,
    const absl::flat_hash_map<std::string, std::variant<int, std::string>>&
        options) {
  auto opts = GetRemoteSessionManagerOptionsLocked(logdir, options);
  session_ = tsl::ProfilerSession::Create(opts.profiler_options());
  logdir_ = logdir;
  return session_->Status();
}

absl::Status ProfilerSessionWrapper::Stop(tensorflow::string* result) {
  if (session_ != nullptr) {
    tensorflow::profiler::XSpace xspace;
    absl::Status status = session_->CollectData(&xspace);
    session_.reset();
    tsl::profiler::ConvertXSpaceToTraceEventsString(xspace, result);
    TF_RETURN_IF_ERROR(status);
  }
  return absl::OkStatus();
}

absl::Status ProfilerSessionWrapper::ExportToTensorBoard() {
  if (!session_ || logdir_.empty()) {
    return absl::OkStatus();
  }
  tensorflow::profiler::XSpace xspace;
  absl::Status status;
  status = session_->CollectData(&xspace);
  session_.reset();
  status = tsl::profiler::ExportToTensorBoard(xspace, logdir_);
  return status;
}

}  // namespace pywrap
}  // namespace profiler
}  // namespace tensorflow
