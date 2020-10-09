#if defined(PLATFORM_GOOGLE)
#include "third_party/tensorflow/core/tpu/tpu_executor_init_fns.inc"
#else
#include "tensorflow/core/tpu/tpu_executor_init_fns.inc"
#endif

namespace {

tensorflow::Status SetTpuConfigStructFns(void* library_handle) {
  auto* config_fn = tensorflow::tpu::ConfigApiFn();

  TFTPU_SET_FN(config_fn, ConfigureDistributedTpuOp_DoWork);
  TFTPU_SET_FN(config_fn, WaitForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(config_fn, InitializeHostForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(config_fn, SetGlobalTPUArrayOp_DoWork);
  TFTPU_SET_FN(config_fn, DisconnectDistributedTpuChipsOp_DoWork);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_FreeCharArray);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_FreeInt32Array);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_HasTPUPodState);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_TpusPerHost);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_TpuMemoryLimit);
  TFTPU_SET_FN(config_fn,
               TpuConfigurationApi_RemoteCompilationCacheSizeInBytes);
  TFTPU_SET_FN(config_fn,
               TpuConfigurationApi_CompilationCacheServerAddressFromConfig);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_GetServerAddressAndPort);

  return tensorflow::Status::OK();
}

tensorflow::Status SetTpuMeshStateStructFns(void* library_handle) {
  auto* mesh_state_fn = tensorflow::tpu::MeshStateApiFn();

  TFTPU_SET_FN(mesh_state_fn, TpuMeshState_Create);
  TFTPU_SET_FN(mesh_state_fn, TpuMeshState_Free);
  TFTPU_SET_FN(mesh_state_fn, TpuMeshState_MeshCommonState);

  return tensorflow::Status::OK();
}

tensorflow::Status SetCompileStructFn(void* library_handle) {
  auto* compile_fn = tensorflow::tpu::CompileApiFn();

  TFTPU_SET_FN(compile_fn, TpuCompile_CompileAndBuild);

  return tensorflow::Status::OK();
}

tensorflow::Status SetExecuteStructFn(void* library_handle) {
  auto* execute_fn = tensorflow::tpu::ExecuteApiFn();

  TFTPU_SET_FN(execute_fn, TpuExecutable_LoadProgramAndEnqueueToStream);
  TFTPU_SET_FN(execute_fn, HardwareLayout_HostShapeToDeviceShape);
  TFTPU_SET_FN(execute_fn, HardwareLayout_ShapeSize);
  TFTPU_SET_FN(execute_fn, HardwareLayout_ShapeSizeCompact);
  TFTPU_SET_FN(execute_fn, HardwareLayout_ShapeSizeCompactRaw);
  TFTPU_SET_FN(execute_fn, TpuExecute_RuntimeInputToPaddedData);

  return tensorflow::Status::OK();
}

tensorflow::Status SetTpuProgramStructFn(void* library_handle) {
  auto* tpu_program_fn = tensorflow::tpu::TpuProgramApiFn();

  TFTPU_SET_FN(tpu_program_fn, TpuProgram_New);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_Free);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_NewArray);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_FreeArray);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_UnloadAndDestroy);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_GetProgramSize);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_LogProgramMemorySummary);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_GetExecutableInfo);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_GetHostTransferInfo);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_GetHloMetadata);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_GetMayModifyVariables);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_HasSharding);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_GetTpuProgram);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_SerializeTpuExecutable);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_SerializeCompilerMetadata);
  TFTPU_SET_FN(tpu_program_fn, TpuProgram_SerializeHostComputeMetadata);
  TFTPU_SET_FN(tpu_program_fn,
               TpuProgram_DeserializeFromGetTpuProgramResponseProto);

  return tensorflow::Status::OK();
}

tensorflow::Status SetTpuNodeContextStructFns(void* library_handle) {
  auto* node_context_fn = tensorflow::tpu::NodeContextApiFn();

  TFTPU_SET_FN(node_context_fn, TpuNodeContext_Create);
  TFTPU_SET_FN(node_context_fn, TpuNodeContext_Free);
  TFTPU_SET_FN(node_context_fn, TpuNodeContext_Initialize);
  TFTPU_SET_FN(node_context_fn, TpuNodeContext_StopChipHeartbeats);
  TFTPU_SET_FN(node_context_fn, TpuNodeContext_CloseTpuHost);

  return tensorflow::Status::OK();
}

tensorflow::Status SetTpuUtilStructFns(void* library_handle) {
  auto* util_fn = tensorflow::tpu::UtilApiFn();

  TFTPU_SET_FN(util_fn, TpuTopology_AvailableCoreCount);
  TFTPU_SET_FN(util_fn, TpuNetUtil_RecycleUnusedPort);
  TFTPU_SET_FN(util_fn, TpuCompile_IsTpuCompilationEnabled);
  TFTPU_SET_FN(util_fn, TpuCompile_ShouldTpuCompileOpIgnoreCancellation);
  TFTPU_SET_FN(util_fn, TpuCompile_CreateCompilationCacheKey);
  TFTPU_SET_FN(util_fn, TpuCompile_DestroyCompilationCacheKey);
  TFTPU_SET_FN(util_fn, TpuCompile_CreateGuaranteedConstFingerprint);

  return tensorflow::Status::OK();
}

tensorflow::Status InitializeTpuStructFns(void* library_handle) {
  TF_RETURN_IF_ERROR(SetTpuConfigStructFns(library_handle));
  TF_RETURN_IF_ERROR(SetTpuMeshStateStructFns(library_handle));
  TF_RETURN_IF_ERROR(SetCompileStructFn(library_handle));
  TF_RETURN_IF_ERROR(SetExecuteStructFn(library_handle));
  TF_RETURN_IF_ERROR(SetTpuProgramStructFn(library_handle));
  TF_RETURN_IF_ERROR(SetExecutorStructFn(library_handle));
  TF_RETURN_IF_ERROR(SetTpuNodeContextStructFns(library_handle));
  TF_RETURN_IF_ERROR(SetTpuUtilStructFns(library_handle));

  return tensorflow::Status::OK();
}

}  // namespace
