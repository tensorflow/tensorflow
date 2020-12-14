/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"

namespace tensorflow {
namespace profiler {
namespace {

// Tpu implementation of ProfilerInterface.
//
// Thread-safety: This class is go/thread-compatible.
class TpuTracer : public ProfilerInterface {
 public:
  explicit TpuTracer();
  ~TpuTracer() override;

  Status Start() override;

  Status Stop() override;

  // Unsupported.
  Status CollectData(RunMetadata* run_metadata) override;

  Status CollectData(XSpace* space) override;

 private:
  TpuProfiler* tpu_profiler_;
};

TpuTracer::TpuTracer() {
  tpu_profiler_ = tpu::OpsApiFn()->TpuProfiler_CreateFn();
}

TpuTracer::~TpuTracer() { tpu::OpsApiFn()->TpuProfiler_FreeFn(tpu_profiler_); }

Status TpuTracer::Start() {
  StatusHelper status;
  tpu::OpsApiFn()->TpuProfiler_StartFn(tpu_profiler_, status.c_status);
  if (!status.ok()) {
    VLOG(1) << "Run Start failed.";
    return status.status();
  }
  return Status::OK();
}

Status TpuTracer::Stop() {
  StatusHelper status;
  tpu::OpsApiFn()->TpuProfiler_StopFn(tpu_profiler_, status.c_status);
  if (!status.ok()) {
    VLOG(1) << "Run Stop failed.";
    return status.status();
  }
  return Status::OK();
}

Status TpuTracer::CollectData(RunMetadata* run_metadata) {
  // Unsupported
  return Status::OK();
}

Status TpuTracer::CollectData(XSpace* space) {
  StatusHelper status;
  tpu::OpsApiFn()->TpuProfiler_CollectDataFn(tpu_profiler_, status.c_status,
                                             space);
  if (!status.ok()) {
    VLOG(1) << "Run CollectData failed.";
    return status.status();
  }
  return Status::OK();
}

}  // namespace

// Not in anonymous namespace for testing purposes.
std::unique_ptr<ProfilerInterface> CreateTpuTracer(
    const ProfileOptions& options) {
  if (options.device_type() != ProfileOptions::TPU &&
      options.device_type() != ProfileOptions::UNSPECIFIED) {
    return nullptr;
  }
  return absl::make_unique<TpuTracer>();
}

auto register_host_tracer_factory = [] {
  RegisterProfilerFactory(&CreateTpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace tensorflow
