/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace profiler {
namespace runtime {
namespace {
class TraceCollector : public RunMetadataListener {
 public:
  explicit TraceCollector(EagerContext* const eager_context);

  void BeforeClearRunMetadata() override;

  Status CollectData(RunMetadata* run_metadata);

 private:
  RunMetadata run_metadata_;
  EagerContext* const context_;
};

class EagerProfiler : public ProfilerInterface {
 public:
  explicit EagerProfiler(EagerContext* const eager_context);

  Status Start() override;

  Status Stop() override;

  Status CollectData(RunMetadata* run_metadata) override;

  EagerContext* const context_;
  TraceCollector collector_;
};

TraceCollector::TraceCollector(EagerContext* const eager_context)
    : context_(eager_context) {}

void TraceCollector::BeforeClearRunMetadata() {
  run_metadata_.MergeFrom(*context_->RunMetadataProto());
}

Status TraceCollector::CollectData(RunMetadata* run_metadata) {
  run_metadata->MergeFrom(run_metadata_);
  return Status::OK();
}

Status EagerProfiler::Start() {
  if (context_ == nullptr) {
    return Status(tensorflow::error::Code::FAILED_PRECONDITION,
                  "No eager context attached.");
  }
  return context_->RegisterRunMetadataListener(&collector_);
}

Status EagerProfiler::Stop() {
  collector_.BeforeClearRunMetadata();
  context_->ClearRunMetadataListener();
  return Status::OK();
}

Status EagerProfiler::CollectData(RunMetadata* run_metadata) {
  return collector_.CollectData(run_metadata);
}

EagerProfiler::EagerProfiler(EagerContext* const eager_context)
    : context_(eager_context), collector_(eager_context) {}
}  // namespace

std::unique_ptr<ProfilerInterface> CreateEagerProfiler(
    const ProfilerContext* context) {
  if (!context || !context->eager_context) {
    return nullptr;
  }
  return absl::make_unique<EagerProfiler>(context->eager_context);
}

auto register_eager_profiler_factory = [] {
  bool enable;
  TF_CHECK_OK(
      ReadBoolFromEnvVar("TF_ENABLE_EAGER_RUNTIME_PROFILER", true, &enable));
  if (enable) {
    RegisterProfilerFactory(&CreateEagerProfiler);
  }
  return 0;
}();

}  // namespace runtime
}  // namespace profiler
}  // namespace tensorflow
