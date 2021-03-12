#if defined(PLATFORM_GOOGLE)
#include "third_party/tensorflow/core/tpu/tpu_executor_init_fns.inc"
#else
#include "tensorflow/core/tpu/tpu_executor_init_fns.inc"
#endif

namespace {

tensorflow::Status SetTpuOpsStructFns(void* library_handle) {
  // Constant cast so that we can initialize the functions. The functions are
  // mutable here because this is the only place where they are initialized.
  auto* ops_api_fn = const_cast<TfTpu_OpsApiFn*>(tensorflow::tpu::OpsApiFn());

  TFTPU_SET_FN(ops_api_fn, ConfigureDistributedTpuOp_DoWork);
  TFTPU_SET_FN(ops_api_fn, WaitForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(ops_api_fn, InitializeHostForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(ops_api_fn, SetGlobalTPUArrayOp_DoWork);
  TFTPU_SET_FN(ops_api_fn, DisconnectDistributedTpuChipsOp_DoWork);
  TFTPU_SET_FN(ops_api_fn, TpuConfigurationApi_FreeCharArray);
  TFTPU_SET_FN(ops_api_fn, TpuConfigurationApi_FreeInt32Array);
  TFTPU_SET_FN(ops_api_fn, TpuConfigurationApi_HasTPUPodState);
  TFTPU_SET_FN(ops_api_fn, TpuConfigurationApi_TpusPerHost);
  TFTPU_SET_FN(ops_api_fn, TpuConfigurationApi_TpuMemoryLimit);
  TFTPU_SET_FN(ops_api_fn,
               TpuConfigurationApi_RemoteCompilationCacheSizeInBytes);
  TFTPU_SET_FN(ops_api_fn,
               TpuConfigurationApi_CompilationCacheServerAddressFromConfig);
  TFTPU_SET_FN(ops_api_fn, TpuConfigurationApi_GetServerAddressAndPort);

  TFTPU_SET_FN(ops_api_fn, TpuMeshState_Create);
  TFTPU_SET_FN(ops_api_fn, TpuMeshState_Free);
  TFTPU_SET_FN(ops_api_fn, TpuMeshState_MeshCommonState);

  TFTPU_SET_FN(ops_api_fn, TpuCompile_CompileAndBuild);
  TFTPU_SET_FN(ops_api_fn, TpuCompile_XrtCompileAndBuild);

  TFTPU_SET_FN(ops_api_fn, TpuExecutable_LoadProgramAndEnqueueToStream);
  TFTPU_SET_FN(ops_api_fn, HardwareLayout_HostShapeToDeviceShape);
  TFTPU_SET_FN(ops_api_fn, HardwareLayout_ShapeSize);
  TFTPU_SET_FN(ops_api_fn, HardwareLayout_ShapeSizeCompact);
  TFTPU_SET_FN(ops_api_fn, HardwareLayout_ShapeSizeCompactRaw);
  TFTPU_SET_FN(ops_api_fn, TpuExecute_RuntimeInputToPaddedData);

  TFTPU_SET_FN(ops_api_fn, TpuProgram_New);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_Free);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_NewArray);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_FreeArray);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_UnloadAndDestroy);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetProgramSize);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_LogProgramMemorySummary);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetExecutableInfo);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetHostTransferInfo);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetHloMetadata);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetMayModifyVariables);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_HasSharding);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetTpuProgram);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_SerializeTpuExecutable);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_SerializeCompilerMetadata);
  TFTPU_SET_FN(ops_api_fn,
               TpuProgram_DeserializeFromGetTpuProgramResponseProto);

  TFTPU_SET_FN(ops_api_fn, TpuNodeContext_Create);
  TFTPU_SET_FN(ops_api_fn, TpuNodeContext_Free);
  TFTPU_SET_FN(ops_api_fn, TpuNodeContext_Initialize);
  TFTPU_SET_FN(ops_api_fn, TpuNodeContext_StopChipHeartbeats);
  TFTPU_SET_FN(ops_api_fn, TpuNodeContext_CloseTpuHost);
  TFTPU_SET_FN(ops_api_fn, TpuNodeContext_CompactionSupported);

  TFTPU_SET_FN(ops_api_fn, TpuTopology_AvailableCoreCount);
  TFTPU_SET_FN(ops_api_fn, TpuNetUtil_RecycleUnusedPort);
  TFTPU_SET_FN(ops_api_fn, TpuCompile_IsTpuCompilationEnabled);
  TFTPU_SET_FN(ops_api_fn, TpuCompile_ShouldTpuCompileOpIgnoreCancellation);
  TFTPU_SET_FN(ops_api_fn, TpuCompile_CreateCompilationCacheKey);
  TFTPU_SET_FN(ops_api_fn, TpuCompile_DestroyCompilationCacheKey);
  TFTPU_SET_FN(ops_api_fn, TpuCompile_CreateGuaranteedConstFingerprint);

  TFTPU_SET_FN(ops_api_fn, TpuProfiler_Create);
  TFTPU_SET_FN(ops_api_fn, TpuProfiler_Destroy);
  TFTPU_SET_FN(ops_api_fn, TpuProfiler_Start);
  TFTPU_SET_FN(ops_api_fn, TpuProfiler_Stop);
  TFTPU_SET_FN(ops_api_fn, TpuProfiler_CollectData);

  TFTPU_SET_FN(ops_api_fn, TfTpu_InitializeTpuModelServer);

  TFTPU_SET_FN(ops_api_fn, TfTpuOrdinalSelector_Create);
  TFTPU_SET_FN(ops_api_fn, TfTpuOrdinalSelector_Destroy);
  TFTPU_SET_FN(ops_api_fn, TfTpuOrdinalSelector_GetOrdinal);
  TFTPU_SET_FN(ops_api_fn, TfTpuOrdinalSelector_DequeueFromCoreSelector);
  TFTPU_SET_FN(ops_api_fn, TfTpu_GetTpuPartitionedCallParams);

  return tensorflow::Status::OK();
}

tensorflow::Status InitializeTpuStructFns(void* library_handle) {
  TF_RETURN_IF_ERROR(SetTpuOpsStructFns(library_handle));
  TF_RETURN_IF_ERROR(SetExecutorStructFn(library_handle));

  return tensorflow::Status::OK();
}

}  // namespace
