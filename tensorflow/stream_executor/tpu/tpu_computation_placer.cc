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

#include "tensorflow/stream_executor/tpu/tpu_computation_placer.h"

#include "tensorflow/stream_executor/tpu/tpu_platform.h"

template <typename T>
using StatusOr = TpuComputationPlacer::StatusOr<T>;

TpuComputationPlacer::TpuComputationPlacer() {
  placer_ = TpuComputationPlacer_New();
}

TpuComputationPlacer::~TpuComputationPlacer() {
  TpuComputationPlacer_Free(placer_);
}

StatusOr<int> TpuComputationPlacer::DeviceId(int replica, int computation,
                                             int replica_count,
                                             int computation_count) {
  LOG(FATAL) << "Unimplemented.";
}

StatusOr<xla::DeviceAssignment> TpuComputationPlacer::AssignDevices(
    int replica_count, int computation_count) {
  LOG(FATAL) << "Unimplemented.";
}

static std::unique_ptr<xla::ComputationPlacer> CreateTpuComputationPlacer() {
  return std::make_unique<TpuComputationPlacer>();
}

static bool InitModule() {
  xla::ComputationPlacer::RegisterComputationPlacer(
      tensorflow::TpuPlatform::kId, CreateTpuComputationPlacer);
  return true;
}
static bool module_initialized = InitModule();
