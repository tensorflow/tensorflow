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
#include "tensorflow/core/profiler/internal/runtime/eager_profiler.h"

namespace tensorflow {
namespace profiler {
namespace runtime {

TraceCollector::TraceCollector(EagerContext* const eager_context)
    : context_(eager_context) {}

void TraceCollector::BeforeClearRunMetadata() {
  run_metadata_.MergeFrom(*context_->RunMetadataProto());
}

Status TraceCollector::CollectData(RunMetadata* run_metadata) {
  run_metadata->MergeFrom(run_metadata_);
  return Status::OK();
}

/* static */ std::unique_ptr<ProfilerInterface> EagerProfiler::Create(
    EagerContext* const eager_context) {
  return absl::WrapUnique(new EagerProfiler(eager_context));
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

}  // namespace runtime
}  // namespace profiler
}  // namespace tensorflow
