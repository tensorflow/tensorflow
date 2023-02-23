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
#include "tensorflow/compiler/xla/stream_executor/tpu/status_helper.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_api.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/types.h"
#include "tensorflow/tsl/profiler/lib/profiler_factory.h"
#include "tensorflow/tsl/profiler/lib/profiler_interface.h"
#include "tensorflow/tsl/profiler/protobuf/profiler_options.pb.h"
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"
#include "tensorflow/tsl/profiler/utils/xplane_schema.h"

namespace xla {
namespace profiler {
namespace {

using tensorflow::ProfileOptions;
using tensorflow::profiler::XPlane;
using tensorflow::profiler::XSpace;
using tsl::OkStatus;  // TENSORFLOW_STATUS_OK
using tsl::Status;    // TENSORFLOW_STATUS_OK
using tsl::profiler::ProfilerInterface;

// Tpu implementation of ProfilerInterface.
//
// Thread-safety: This class is go/thread-compatible.
class TpuTracer : public ProfilerInterface {
 public:
  explicit TpuTracer();
  ~TpuTracer() override;

  Status Start() override;

  Status Stop() override;

  Status CollectData(XSpace* space) override;

 private:
  TpuProfiler* tpu_profiler_;
};

TpuTracer::TpuTracer() {
  StatusHelper status;
  stream_executor::tpu::OpsApiFn()->TpuProfiler_CreateFn(&tpu_profiler_,
                                                         status.c_status);
  if (!status.ok()) {
    LOG(ERROR) << status.status().error_message();
  }
}

TpuTracer::~TpuTracer() {
  stream_executor::tpu::OpsApiFn()->TpuProfiler_DestroyFn(tpu_profiler_);
}

Status TpuTracer::Start() {
  StatusHelper status;
  stream_executor::tpu::OpsApiFn()->TpuProfiler_StartFn(tpu_profiler_,
                                                        status.c_status);
  if (!status.ok()) {
    LOG(ERROR) << "TPU tracer failed to start.";
    return status.status();
  }
  return OkStatus();
}

Status TpuTracer::Stop() {
  StatusHelper status;
  stream_executor::tpu::OpsApiFn()->TpuProfiler_StopFn(tpu_profiler_,
                                                       status.c_status);
  if (!status.ok()) {
    LOG(ERROR) << "TPU tracer failed to stop.";
    return status.status();
  }
  return OkStatus();
}

Status TpuTracer::CollectData(XSpace* space) {
  StatusHelper status;
  // Get size of buffer required for TPU driver to serialize XSpace into.
  size_t size_in_bytes;
  stream_executor::tpu::OpsApiFn()->TpuProfiler_CollectDataFn(
      tpu_profiler_, status.c_status,
      /*buffer=*/nullptr, &size_in_bytes);
  // Prepare an appropriately sized buffer.
  if (size_in_bytes > 0) {
    std::vector<uint8_t> buffer(size_in_bytes);
    stream_executor::tpu::OpsApiFn()->TpuProfiler_CollectDataFn(
        tpu_profiler_, status.c_status, buffer.data(), &size_in_bytes);
    // Deserialize XSpace from the buffer and return it.
    XSpace tpu_space;
    tpu_space.ParseFromArray(buffer.data(), buffer.size());
    for (XPlane& tpu_plane : *tpu_space.mutable_planes()) {
      XPlane* plane = space->add_planes();
      plane->Swap(&tpu_plane);
    }
  }
  if (!status.ok()) {
    LOG(ERROR) << "TPU tracer failed to collect data.";
    return status.status();
  }
  return OkStatus();
}

}  // namespace

// Not in anonymous namespace for testing purposes.
std::unique_ptr<ProfilerInterface> CreateTpuTracer(
    const ProfileOptions& options) {
  if (options.device_type() != ProfileOptions::TPU &&
      options.device_type() != ProfileOptions::UNSPECIFIED) {
    return nullptr;
  }
  // Don't attempt to create a TpuTracer if the TPU C API isn't initialized.
  if (stream_executor::tpu::OpsApiFn()->TpuProfiler_CreateFn == nullptr) {
    return nullptr;
  }
  return std::make_unique<TpuTracer>();
}

auto register_tpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateTpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace xla
