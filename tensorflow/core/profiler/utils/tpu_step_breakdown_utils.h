/* Copyright 2024 The OpenXLA Authors.

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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_TPU_STEP_BREAKDOWN_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_TPU_STEP_BREAKDOWN_UTILS_H_

#include <cstdint>

#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"

namespace tensorflow {
namespace profiler {

// Total duration of infeed from host or SparseCoreV0 to TensorCore.
inline uint64_t InfeedDurationPs(const TpuStepBreakdown& tpu) {
  return tpu.infeed_duration_ps() + tpu.wait_for_scv0_duration_ps() +
         tpu.scv0_infeed_transform_ps();
}

// Total duration of outfeed from TensorCore to host or SparseCoreV0.
inline uint64_t OutfeedDurationPs(const TpuStepBreakdown& tpu) {
  return tpu.host_outfeed_ps() + tpu.scv0_outfeed_ps();
}

// Total duration of infeed from host to SparseCoreV0.
inline uint64_t ScV0InfeedDurationPs(const TpuStepBreakdown& tpu) {
  return tpu.wait_for_scv0_duration_ps() * tpu.scv0_infeed_percent() / 100.0;
}

// Total duration of SparseCoreV0 compute.
inline uint64_t ScV0ComputeDurationPs(const TpuStepBreakdown& tpu) {
  return tpu.wait_for_scv0_duration_ps() - ScV0InfeedDurationPs(tpu);
}

// Total duration of infeed from host to TensorCore or SparseCoreV0.
inline uint64_t TcPlusScV0InfeedDurationPs(const TpuStepBreakdown& tpu) {
  return tpu.infeed_duration_ps() + ScV0InfeedDurationPs(tpu);
}

// Total duration of send and recv ops.
inline uint64_t SendRecvDurationPs(const TpuStepBreakdown& tpu) {
  return tpu.send_duration_ps() + tpu.recv_duration_ps();
}

// Total duration of host send and host recv ops.
inline uint64_t HostSendRecvDurationPs(const TpuStepBreakdown& tpu) {
  return tpu.host_send_duration_ps() + tpu.host_recv_duration_ps();
}

// Total duration TensorCore spends waiting for host.
inline uint64_t WaitForHostDurationPs(const TpuStepBreakdown& tpu) {
  return tpu.infeed_duration_ps() + tpu.host_outfeed_ps() +
         HostSendRecvDurationPs(tpu) + tpu.tc_idle_ps();
}

// Total duration TensorCore spends waiting for host or SparseCoreV0.
inline uint64_t WaitForHostOrScV0DurationPs(const TpuStepBreakdown& tpu) {
  return WaitForHostDurationPs(tpu) + tpu.wait_for_scv0_duration_ps();
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_TPU_STEP_BREAKDOWN_UTILS_H_
