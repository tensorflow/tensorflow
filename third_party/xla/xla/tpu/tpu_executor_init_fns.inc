// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#ifndef TENSORFLOW_COMPILER_XLA_TPU_TPU_EXECUTOR_INIT_FNS_INC_
#define TENSORFLOW_COMPILER_XLA_TPU_TPU_EXECUTOR_INIT_FNS_INC_

#include "xla/tpu/tpu_api.h"
#include "xla/tpu/tpu_executor_api.h"

namespace {

// NOLINTNEXTLINE(google-readability-function-size)
absl::Status DoSetExecutorStructFn(  // NOLINT(google-readability-function-size)
    void* library_handle) {          // TENSORFLOW_STATUS_OK,
                                     // NOLINT(google-readability-function-size)
  auto set_executor_struct = [&]() -> absl::Status {
    static TfTpu_ExecutorApiFn executor_fn;

    TFTPU_SET_FN(&executor_fn, TpuStatus_New);
    TFTPU_SET_FN(&executor_fn, TpuStatus_Create);
    TFTPU_SET_FN(&executor_fn, TpuStatus_Set);
    TFTPU_SET_FN(&executor_fn, TpuStatus_Free);
    TFTPU_SET_FN(&executor_fn, TpuStatus_Message);
    TFTPU_SET_FN(&executor_fn, TpuStatus_Code);
    TFTPU_SET_FN(&executor_fn, TpuStatus_Ok);

    TFTPU_SET_FN(&executor_fn, TpuTransferManager_GetInfeedLayout);

    TFTPU_SET_FN(&executor_fn, TpuTopology_LogicalDevicesPerHost);
    TFTPU_SET_FN(&executor_fn, TpuTopology_LogicalDevicesPerChip);
    TFTPU_SET_FN(&executor_fn, TpuTopology_HostCount);
    TFTPU_SET_FN(&executor_fn, TpuTopology_ChipsPerHost);
    TFTPU_SET_FN(&executor_fn, TpuTopology_ChipBounds_X);
    TFTPU_SET_FN(&executor_fn, TpuTopology_ChipBounds_Y);
    TFTPU_SET_FN(&executor_fn, TpuTopology_ChipBounds_Z);
    TFTPU_SET_FN(&executor_fn, TpuTopology_HasChip);
    TFTPU_SET_FN(&executor_fn, TpuTopology_CoreForId);
    TFTPU_SET_FN(&executor_fn, TpuTopology_Core);
    TFTPU_SET_FN(&executor_fn, TpuTopology_NumCores);
    TFTPU_SET_FN(&executor_fn, TpuTopology_Cores);
    TFTPU_SET_FN(&executor_fn, TpuTopology_IdForHost);
    TFTPU_SET_FN(&executor_fn, TpuTopology_Version);

    TFTPU_SET_FN(&executor_fn, TpuCoreLocation_ChipCoordinates);
    TFTPU_SET_FN(&executor_fn, TpuCoreLocation_HostCoordinates);
    TFTPU_SET_FN(&executor_fn, TpuCoreLocation_Index);
    TFTPU_SET_FN(&executor_fn, TpuCoreLocation_Id);

    TFTPU_SET_FN(&executor_fn, TpuHostLocation_Id);
    TFTPU_SET_FN(&executor_fn, TpuHostLocation_NumCores);
    TFTPU_SET_FN(&executor_fn, TpuHostLocation_Cores);

    ::stream_executor::tpu::SetExecutorApiFn(&executor_fn);
    return absl::OkStatus();
  };

  // Capture the status of setting up the executor functions. If any TFTPU_SET_FN
  // fails, the lambda will return an error status, which is then stored.
  ::stream_executor::tpu::GetTpuExecutorInitStatus() = set_executor_struct();
  return ::stream_executor::tpu::GetTpuExecutorInitStatus();
}

// To avoid template symbol redefinition and link-time resolution mismatches
// across shared library boundaries (which occur with templates like
// absl::AnyInvocable under specific compilers), we use absolute
// GetTpuExecutorInitOnceFlag() getters that return raw once-flags pointers.
absl::Status SetExecutorStructFn(void* library_handle) {
  absl::call_once(*::stream_executor::tpu::GetTpuExecutorInitOnceFlag(),
                  [library_handle]() {
                    DoSetExecutorStructFn(library_handle).IgnoreError();
                  });
  return ::stream_executor::tpu::GetTpuExecutorInitStatus();
}

}  // namespace

#endif  // TENSORFLOW_COMPILER_XLA_TPU_TPU_EXECUTOR_INIT_FNS_INC_
