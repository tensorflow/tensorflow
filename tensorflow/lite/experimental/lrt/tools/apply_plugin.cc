// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/lrt/tools/apply_plugin.h"

#include <memory>
#include <ostream>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"

namespace lrt::tools {

#define _ENSURE_CONFIG(expr)        \
  if (!(expr)) {                    \
    return kLrtStatusToolBadConfig; \
  }

using ::lrt::tools::ApplyPluginRun;

namespace {

class ApplyPluginContext {
 public:
  using Ptr = std::unique_ptr<ApplyPluginContext>;
  using ResultT = LrtResult<ApplyPluginContext>;

  explicit ApplyPluginContext(ApplyPluginRun::Ptr run) : run_(std::move(run)) {}

  ApplyPluginRun::Cmd Cmd() const { return run_->cmd; }

  absl::Span<const absl::string_view> LibSearchPaths() const {
    return absl::MakeConstSpan(run_->lib_search_paths.data(),
                               run_->lib_search_paths.size());
  }

  std::ostream& Dump() {
    if (!run_->dump_out.has_value()) {
      return null_stream_;
    }
    return *run_->dump_out;
  }

  const ApplyPluginRun& Run() const { return *run_; }

  void DumpPrelude();

 private:
  ApplyPluginRun::Ptr run_;
  std::ostream null_stream_ = std::ostream(nullptr);
};

void ApplyPluginContext::DumpPrelude() {
  static constexpr absl::string_view kCmdTpl = "ApplyPlugin: %s\n";
  switch (Run().cmd) {
    case ApplyPluginRun::Cmd::INFO:
      Dump() << absl::StreamFormat(kCmdTpl, "INFO");
      break;
    case ApplyPluginRun::Cmd::NOOP:
      Dump() << absl::StreamFormat(kCmdTpl, "NOOP");
      break;
    case ApplyPluginRun::Cmd::PARTITION:
      Dump() << absl::StreamFormat(kCmdTpl, "PARTITION");
      break;
    case ApplyPluginRun::Cmd::COMPILE:
      Dump() << absl::StreamFormat(kCmdTpl, "COMPILE");
      break;
    case ApplyPluginRun::Cmd::APPLY:
      Dump() << absl::StreamFormat(kCmdTpl, "APPLY");
      break;
  }
}

//
// INFO Command
//

LrtStatus ValidateInfoRun(const ApplyPluginRun& run) {
  _ENSURE_CONFIG(!run.lib_search_paths.empty());
  _ENSURE_CONFIG(run.dump_out.has_value());
  return kLrtStatusOk;
}

LrtStatus Info(ApplyPluginContext* context) {
  // TODO
  return kLrtStatusErrorUnsupported;
}

//
// NOOP Command
//

LrtStatus ValidateNoopRun(const ApplyPluginRun& run) {
  _ENSURE_CONFIG(run.model.has_value());
  _ENSURE_CONFIG(run.outs.size() == 1);
  return kLrtStatusOk;
}

LrtStatus Noop(ApplyPluginContext* context) {
  // TODO
  return kLrtStatusErrorUnsupported;
}

//
// PARTITION Command
//

LrtStatus ValidatePartitionRun(const ApplyPluginRun& run) {
  _ENSURE_CONFIG(!run.lib_search_paths.empty());
  _ENSURE_CONFIG(run.model.has_value());
  _ENSURE_CONFIG(run.soc_manufacturer.has_value());
  _ENSURE_CONFIG(!run.outs.empty());
  return kLrtStatusOk;
}

LrtStatus Partition(ApplyPluginContext* context) {
  // TODO
  return kLrtStatusErrorUnsupported;
}

//
// COMPILE Command
//

LrtStatus ValidateCompileRun(const ApplyPluginRun& run) {
  _ENSURE_CONFIG(!run.lib_search_paths.empty());
  _ENSURE_CONFIG(run.model.has_value());
  _ENSURE_CONFIG(run.soc_manufacturer.has_value());
  _ENSURE_CONFIG(run.outs.size() == run.soc_models.size());
  // TODO: implement multi target compilation.
  LRT_ENSURE_SUPPORTED(run.soc_models.size() == 1,
                       "Multi target compilation not implemented.");
  // TODO: implement append serialization.
  LRT_ENSURE_SUPPORTED(
      run.serialization == ApplyPluginRun::Serialization::METADATA,
      "Only metadata serialization currently supported.");
  return kLrtStatusOk;
}

LrtStatus Compile(ApplyPluginContext* context) {
  // TODO
  return kLrtStatusErrorUnsupported;
}

//
// APPLY Command
//

LrtStatus ValidateApplyRun(const ApplyPluginRun& run) {
  _ENSURE_CONFIG(!run.lib_search_paths.empty());
  _ENSURE_CONFIG(run.model.has_value());
  _ENSURE_CONFIG(run.soc_manufacturer.has_value());
  _ENSURE_CONFIG(run.outs.size() == run.soc_models.size());
  // TODO: implement multi target compilation.
  LRT_ENSURE_SUPPORTED(run.soc_models.size() == 1,
                       "Multi target compilation not implemented.");
  // TODO: implement append serialization.
  LRT_ENSURE_SUPPORTED(
      run.serialization == ApplyPluginRun::Serialization::METADATA,
      "Only metadata serialization currently supported.");
  return kLrtStatusOk;
}

LrtStatus Apply(ApplyPluginContext* context) {
  // TODO
  return kLrtStatusErrorUnsupported;
}

}  // namespace

LrtStatus ApplyPlugin(ApplyPluginRun::Ptr run) {
  ApplyPluginContext context(std::move(run));
  context.DumpPrelude();

  switch (context.Cmd()) {
    case ApplyPluginRun::Cmd::INFO:
      LRT_RETURN_STATUS_IF_NOT_OK(ValidateInfoRun(context.Run()));
      return Info(&context);

    case ApplyPluginRun::Cmd::PARTITION:
      LRT_RETURN_STATUS_IF_NOT_OK(ValidatePartitionRun(context.Run()));
      return Partition(&context);

    case ApplyPluginRun::Cmd::COMPILE:
      LRT_RETURN_STATUS_IF_NOT_OK(ValidateCompileRun(context.Run()));
      return Compile(&context);

    case ApplyPluginRun::Cmd::APPLY:
      LRT_RETURN_STATUS_IF_NOT_OK(ValidateApplyRun(context.Run()));
      return Apply(&context);

    case ApplyPluginRun::Cmd::NOOP:
      LRT_RETURN_STATUS_IF_NOT_OK(ValidateNoopRun(context.Run()));
      return Noop(&context);

    default:
      return kLrtStatusErrorInvalidArgument;
  }

  return kLrtStatusOk;
}

}  // namespace lrt::tools
