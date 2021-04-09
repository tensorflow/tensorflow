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

#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_xfeed.h"
#include "tensorflow/core/platform/env.h"

namespace xla {

static const char kCpuPlatformName[] = "cpu";

TfrtCpuDevice::TfrtCpuDevice(int id, bool asynchronous)
    : id_(id),
      max_inflight_computations_semaphore_(/*capacity=*/asynchronous ? 32 : 1) {
}

absl::string_view TfrtCpuDevice::device_kind() const {
  return kCpuPlatformName;
}

std::string TfrtCpuDevice::DebugString() const {
  return absl::StrCat("TFRT_CPU_", id());
}

Status TfrtCpuDevice::TransferToInfeed(const LiteralSlice& literal) {
  return TransferLiteralToInfeedOnCpu(local_hardware_id(), literal);
}

Status TfrtCpuDevice::TransferFromOutfeed(MutableBorrowingLiteral literal) {
  return TransferLiteralFromOutfeedOnCpu(local_hardware_id(), literal);
}

}  // namespace xla
