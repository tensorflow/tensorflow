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

#ifndef TENSORFLOW_COMPILER_XLA_TPU_TPU_LIBRARY_INIT_FNS_INC_
#define TENSORFLOW_COMPILER_XLA_TPU_TPU_LIBRARY_INIT_FNS_INC_

#include "third_party/absl/status/status.h"
#include "xla/tpu/tpu_api.h"
#include "xla/tpu/tpu_executor_init_fns.inc"
#include "xla/tpu/tpu_ops_c_api.h"
#include "xla/tsl/platform/errors.h"

namespace {

absl::Status DoSetTpuOpsStructFns(void* library_handle) {
  static TfTpu_OpsApiFn ops_api_fn;
  TFTPU_SET_FN(&ops_api_fn, SetGlobalTPUArrayOp_DoWork);
  TFTPU_SET_FN(&ops_api_fn, TpuConfigurationApi_FreeCharArray);

  TFTPU_SET_FN(&ops_api_fn, TpuMeshState_Create);
  TFTPU_SET_FN(&ops_api_fn, TpuMeshState_Free);
  TFTPU_SET_FN(&ops_api_fn, TpuMeshState_MeshCommonState);

  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngineState_Create);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngineState_Free);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngineState_GetState);

  TFTPU_SET_FN(&ops_api_fn, TpuTopology_AvailableCoreCount);
  TFTPU_SET_FN(&ops_api_fn, TpuTopology_AvailableCoresPerChip);
  TFTPU_SET_FN(&ops_api_fn,
               TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice);
  TFTPU_SET_FN(&ops_api_fn, TpuUtil_GetTopologyPtr);
  TFTPU_SET_FN(&ops_api_fn, TpuUtil_GetXlaPadSizeFromTpuTopology);

  TFTPU_SET_FN(&ops_api_fn, TfTpuOrdinalSelector_Create);
  TFTPU_SET_FN(&ops_api_fn, TfTpuOrdinalSelector_Destroy);
  TFTPU_SET_FN(&ops_api_fn, TfTpuOrdinalSelector_GetOrdinal);
  TFTPU_SET_FN(&ops_api_fn, TfTpuOrdinalSelector_DequeueFromCoreSelector);
  TFTPU_SET_FN(&ops_api_fn, TfTpu_GetTpuPartitionedCallParams);

  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_ExecutePartitioner);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_ConfigureMemory);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_CollateMemory);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_ConfigureHost);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_ConnectHosts);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_Finalize);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_IsInitialized);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_WriteParameters);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_ReadParameters);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingTensorBatchFixedState_Create);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingTensorBatchFixedState_Destroy);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_EnqueueTensorBatch);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_RecvActivationsComputation);
  TFTPU_SET_FN(&ops_api_fn,
               TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation);
  TFTPU_SET_FN(&ops_api_fn,
               TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_DedupDataSizeComputation);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngine_DedupDataTupleMaskComputation);

  TFTPU_SET_FN(&ops_api_fn, SparseCore_GetMaxIdsAndUniques);

  ::stream_executor::tpu::SetOpsApiFn(&ops_api_fn);
  return absl::OkStatus();
}

// To avoid template symbol redefinition and link-time resolution mismatches
// across shared library boundaries (which occur with templates like
// absl::AnyInvocable under specific compilers), we use absolute
// GetTpuOpsStructFnsOnceFlag() getters that return raw once-flags pointers.
absl::Status SetTpuOpsStructFns(void* library_handle) {  // TENSORFLOW_STATUS_OK
  absl::call_once(*::stream_executor::tpu::GetTpuOpsStructFnsOnceFlag(),
                  [library_handle]() {
                    ::stream_executor::tpu::GetTpuOpsStructFnsInitStatus() =
                        DoSetTpuOpsStructFns(library_handle);
                  });
  return ::stream_executor::tpu::GetTpuOpsStructFnsInitStatus();
}

static absl::Status DoInitializeBaseApi(void* library_handle) {
  static TfTpu_BaseFn base_fn;
  TFTPU_SET_FN(&base_fn, TfTpu_Initialize);
  ::stream_executor::tpu::SetBaseApiFn(&base_fn);
  ::stream_executor::tpu::GetTpuBaseApiInitStatus() =
      absl::OkStatus();  // TENSORFLOW_STATUS_OK
  return absl::OkStatus();
}

[[maybe_unused]] absl::Status InitializeTpuStructFns(  // TENSORFLOW_STATUS_OK
    void* library_handle) {
  absl::call_once(*::stream_executor::tpu::GetTpuBaseApiInitOnceFlag(),
                  [library_handle]() {
                    DoInitializeBaseApi(library_handle).IgnoreError();
                  });
  TF_RETURN_IF_ERROR(::stream_executor::tpu::GetTpuBaseApiInitStatus());

  TF_RETURN_IF_ERROR(SetTpuOpsStructFns(library_handle));
  TF_RETURN_IF_ERROR(SetExecutorStructFn(library_handle));

  return absl::OkStatus();
}

}  // namespace

#endif  // TENSORFLOW_COMPILER_XLA_TPU_TPU_LIBRARY_INIT_FNS_INC_
