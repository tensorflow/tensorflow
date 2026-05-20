#ifndef TENSORFLOW_COMPILER_XLA_TPU_TPU_LIBRARY_INIT_FNS_INC_
#define TENSORFLOW_COMPILER_XLA_TPU_TPU_LIBRARY_INIT_FNS_INC_

#include "third_party/absl/base/call_once.h"
#include "third_party/absl/status/status.h"
#include "xla/tpu/tpu_api.h"
#include "xla/tpu/tpu_executor_init_fns.inc"
#include "xla/tpu/tpu_ops_c_api.h"
#include "xla/tsl/platform/errors.h"

namespace {

absl::Status DoSetTpuOpsStructFns(void* library_handle) {
  static TfTpu_OpsApiFn ops_api_fn;
  TFTPU_SET_FN(&ops_api_fn, ConfigureDistributedTpuOp_DoWork);
  TFTPU_SET_FN(&ops_api_fn, WaitForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(&ops_api_fn, InitializeHostForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(&ops_api_fn, SetGlobalTPUArrayOp_DoWork);
  TFTPU_SET_FN(&ops_api_fn, DisconnectDistributedTpuChipsOp_DoWork);
  TFTPU_SET_FN(&ops_api_fn, TpuConfigurationApi_FreeCharArray);
  TFTPU_SET_FN(&ops_api_fn, TpuConfigurationApi_FreeInt32Array);
  TFTPU_SET_FN(&ops_api_fn, TpuConfigurationApi_HasTPUPodState);
  TFTPU_SET_FN(&ops_api_fn, TpuConfigurationApi_TpusPerHost);
  TFTPU_SET_FN(&ops_api_fn, TpuConfigurationApi_TpuMemoryLimit);
  TFTPU_SET_FN(&ops_api_fn,
               TpuConfigurationApi_RemoteCompilationCacheSizeInBytes);
  TFTPU_SET_FN(&ops_api_fn,
               TpuConfigurationApi_CompilationCacheServerAddressFromConfig);
  TFTPU_SET_FN(&ops_api_fn, TpuConfigurationApi_GetServerAddressAndPort);

  TFTPU_SET_FN(&ops_api_fn, TpuMeshState_Create);
  TFTPU_SET_FN(&ops_api_fn, TpuMeshState_Free);
  TFTPU_SET_FN(&ops_api_fn, TpuMeshState_MeshCommonState);

  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngineState_Create);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngineState_Free);
  TFTPU_SET_FN(&ops_api_fn, TpuEmbeddingEngineState_GetState);

  TFTPU_SET_FN(&ops_api_fn, TpuCompile_CompileAndBuild);

  TFTPU_SET_FN(&ops_api_fn, TpuExecutable_LoadProgramAndEnqueueToStream);

  TFTPU_SET_FN(&ops_api_fn, HardwareLayout_HostShapeToDeviceShape);
  TFTPU_SET_FN(&ops_api_fn, HardwareLayout_ShapeSize);
  TFTPU_SET_FN(&ops_api_fn, HardwareLayout_ShapeSizeCompact);
  TFTPU_SET_FN(&ops_api_fn, HardwareLayout_ShapeSizeCompactRaw);

  TFTPU_SET_FN(&ops_api_fn, TpuExecute_RuntimeInputToPaddedData);

  TFTPU_SET_FN(&ops_api_fn, TpuExecute_GetTpuEmbeddingMemoryAllocations);
  TFTPU_SET_FN(&ops_api_fn, TpuExecute_FreeTpuEmbeddingMemoryAllocations);

  TFTPU_SET_FN(&ops_api_fn, TpuProgram_New);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_Free);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_NewArray);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_FreeArray);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_UnloadAndDestroy);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_GetProgramSize);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_LogProgramMemorySummary);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_GetExecutableInfo);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_GetHostTransferInfo);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_GetHloMetadata);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_GetMayModifyVariables);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_HasSharding);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_GetTpuProgram);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_SerializeTpuExecutable);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_SerializeCompilerMetadata);
  TFTPU_SET_FN(&ops_api_fn,
               TpuProgram_DeserializeFromGetTpuProgramResponseProto);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_GetFingerprint);
  TFTPU_SET_FN(&ops_api_fn, TpuProgram_DestroyFingerprint);

  TFTPU_SET_FN(&ops_api_fn, TpuNodeContext_Create);
  TFTPU_SET_FN(&ops_api_fn, TpuNodeContext_Free);
  TFTPU_SET_FN(&ops_api_fn, TpuNodeContext_Initialize);
  TFTPU_SET_FN(&ops_api_fn, TpuNodeContext_CloseTpuHost);
  TFTPU_SET_FN(&ops_api_fn, TpuNodeContext_CompactionSupported);

  TFTPU_SET_FN(&ops_api_fn, TpuTopology_AvailableCoreCount);
  TFTPU_SET_FN(&ops_api_fn, TpuTopology_AvailableCoresPerChip);
  TFTPU_SET_FN(&ops_api_fn,
               TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice);
  TFTPU_SET_FN(&ops_api_fn, TpuNetUtil_RecycleUnusedPort);
  TFTPU_SET_FN(&ops_api_fn, TpuCompile_IsTpuCompilationEnabled);
  TFTPU_SET_FN(&ops_api_fn, TpuCompile_ShouldTpuCompileOpIgnoreCancellation);
  TFTPU_SET_FN(&ops_api_fn, TpuCompile_CreateCompilationCacheKey);
  TFTPU_SET_FN(&ops_api_fn, TpuCompile_DestroyCompilationCacheKey);
  TFTPU_SET_FN(&ops_api_fn, TpuCompile_CreateGuaranteedConstFingerprint);
  TFTPU_SET_FN(&ops_api_fn, TpuUtil_GetTopologyPtr);
  TFTPU_SET_FN(&ops_api_fn, TpuUtil_GetXlaPadSizeFromTpuTopology);

  TFTPU_SET_FN(&ops_api_fn, TfTpu_InitializeTpuModelServer);

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
  ::stream_executor::tpu::GetTpuOpsStructFnsInitStatus() = absl::OkStatus();
  return absl::OkStatus();
}

// To avoid template symbol redefinition and link-time resolution mismatches
// across shared library boundaries (which occur with templates like
// absl::AnyInvocable under specific compilers), we use absolute
// GetTpuOpsStructFnsOnceFlag() getters that return raw once-flags pointers.
absl::Status SetTpuOpsStructFns(void* library_handle) {  // TENSORFLOW_STATUS_OK
  absl::call_once(*::stream_executor::tpu::GetTpuOpsStructFnsOnceFlag(),
                  [library_handle]() {
                    DoSetTpuOpsStructFns(library_handle).IgnoreError();
                  });
  return ::stream_executor::tpu::GetTpuOpsStructFnsInitStatus();
}

static absl::Status DoInitializeBaseApi(void* library_handle) {
  static TfTpu_BaseFn base_fn;
  TFTPU_SET_FN(&base_fn, TfTpu_Initialize);
  ::stream_executor::tpu::SetBaseApiFn(&base_fn);
  ::stream_executor::tpu::GetTpuBaseApiInitStatus() = absl::OkStatus();  // TENSORFLOW_STATUS_OK
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
