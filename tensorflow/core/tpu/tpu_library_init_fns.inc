namespace {

tensorflow::Status SetTpuConfigStructFns(void* library_handle) {
  auto* config_fn = tensorflow::tpu::ConfigApiFn();

  TFTPU_SET_FN(config_fn, ConfigureDistributedTpuOp_DoWork);
  TFTPU_SET_FN(config_fn, WaitForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(config_fn, ShutdownDistributedTpuOp_DoWork);
  TFTPU_SET_FN(config_fn, InitializeHostForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(config_fn, SetGlobalTPUArrayOp_DoWork);
  TFTPU_SET_FN(config_fn, DisconnectDistributedTpuChipsOp_DoWork);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_FreeCharArray);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_FreeInt32Array);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_HasTPUPodState);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_TpusPerHost);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_TpuMemoryLimit);

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

  TFTPU_SET_FN(compile_fn, TpuTopology_AvailableCoreCount);
  TFTPU_SET_FN(compile_fn, TpuCompile_CreateCompilationCacheKey);
  TFTPU_SET_FN(compile_fn, TpuCompile_CreateGuaranteedConstFingerprint);
  TFTPU_SET_FN(compile_fn, TpuCompile_CompileAheadOfTime);
  TFTPU_SET_FN(compile_fn, TpuCompile_BuildXLADeviceAssignment);

  return tensorflow::Status::OK();
}

tensorflow::Status SetExecutorStructFn(void* library_handle) {
  auto* executor_fn = tensorflow::tpu::ExecutorApiFn();

  TFTPU_SET_FN(executor_fn, TpuPlatform_New);
  TFTPU_SET_FN(executor_fn, TpuPlatform_Free);
  TFTPU_SET_FN(executor_fn, TpuPlatform_Initialize);
  TFTPU_SET_FN(executor_fn, TpuPlatform_Initialized);
  TFTPU_SET_FN(executor_fn, TpuPlatform_GetExecutor);
  TFTPU_SET_FN(executor_fn, TpuPlatform_Id);
  TFTPU_SET_FN(executor_fn, TpuPlatform_VisibleDeviceCount);
  TFTPU_SET_FN(executor_fn, TpuPlatform_TpuMemoryLimit);
  TFTPU_SET_FN(executor_fn, TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopy);

  TFTPU_SET_FN(executor_fn, TpuExecutor_Init);
  TFTPU_SET_FN(executor_fn, TpuExecutor_Free);
  TFTPU_SET_FN(executor_fn, TpuExecutor_PlatformDeviceCount);
  TFTPU_SET_FN(executor_fn, TpuExecutor_Allocate);
  TFTPU_SET_FN(executor_fn, TpuExecutor_Deallocate);
  TFTPU_SET_FN(executor_fn, TpuExecutor_GetAllocatorStats);
  TFTPU_SET_FN(executor_fn, TpuExecutor_DeviceMemoryUsage);
  TFTPU_SET_FN(executor_fn, TpuExecutor_AllocateStream);
  TFTPU_SET_FN(executor_fn, TpuExecutor_DeallocateStream);
  TFTPU_SET_FN(executor_fn, TpuExecutor_CreateStreamDependency);
  TFTPU_SET_FN(executor_fn, TpuExecutor_GetStatus);
  TFTPU_SET_FN(executor_fn, TpuExecutor_AllocateEvent);
  TFTPU_SET_FN(executor_fn, TpuExecutor_DeallocateEvent);
  TFTPU_SET_FN(executor_fn, TpuExecutor_PollForEventStatus);
  TFTPU_SET_FN(executor_fn, TpuExecutor_RecordEvent);
  TFTPU_SET_FN(executor_fn, TpuExecutor_WaitForEvent);
  TFTPU_SET_FN(executor_fn, TpuExecutor_AllocateTimer);
  TFTPU_SET_FN(executor_fn, TpuExecutor_DeallocateTimer);
  TFTPU_SET_FN(executor_fn, TpuExecutor_StartTimer);
  TFTPU_SET_FN(executor_fn, TpuExecutor_StopTimer);
  TFTPU_SET_FN(executor_fn, TpuExecutor_SynchronousMemcpyToHost);
  TFTPU_SET_FN(executor_fn, TpuExecutor_SynchronousMemcpyFromHost);
  TFTPU_SET_FN(executor_fn, TpuExecutor_MemcpyToHost);
  TFTPU_SET_FN(executor_fn, TpuExecutor_MemcpyFromHost);
  TFTPU_SET_FN(executor_fn, TpuExecutor_EnqueueInfeed);
  TFTPU_SET_FN(executor_fn, TpuExecutor_DequeueOutfeed);
  TFTPU_SET_FN(executor_fn, TpuExecutor_WaitForInfeedReady);
  TFTPU_SET_FN(executor_fn, TpuExecutor_WaitForOutfeedReady);
  TFTPU_SET_FN(executor_fn, TpuExecutor_BlockHostUntilDone);
  TFTPU_SET_FN(executor_fn, TpuExecutor_BlockUntilDoneOrFailed);
  TFTPU_SET_FN(executor_fn, TpuExecutor_SyncAndForgetFailedStreams);
  TFTPU_SET_FN(executor_fn, TpuExecutor_SynchronizeAllActivity);

  TFTPU_SET_FN(executor_fn, TpuStream_New);
  TFTPU_SET_FN(executor_fn, TpuStream_Free);
  TFTPU_SET_FN(executor_fn, TpuStream_Stream);
  TFTPU_SET_FN(executor_fn, TpuStream_Status);
  TFTPU_SET_FN(executor_fn, TpuStream_IsSameSharedMemoryLocation);
  TFTPU_SET_FN(executor_fn, TpuStream_TpuEnqueueOnDeviceSendRecvLocal);

  TFTPU_SET_FN(executor_fn, TpuEvent_New);
  TFTPU_SET_FN(executor_fn, TpuEvent_Free);

  TFTPU_SET_FN(executor_fn, TpuTimer_New);
  TFTPU_SET_FN(executor_fn, TpuTimer_Free);
  TFTPU_SET_FN(executor_fn, TpuTimer_Nanoseconds);
  TFTPU_SET_FN(executor_fn, TpuTimer_Microseconds);

  TFTPU_SET_FN(executor_fn, TpuStatus_New);
  TFTPU_SET_FN(executor_fn, TpuStatus_Create);
  TFTPU_SET_FN(executor_fn, TpuStatus_Free);
  TFTPU_SET_FN(executor_fn, TpuStatus_Message);
  TFTPU_SET_FN(executor_fn, TpuStatus_Code);
  TFTPU_SET_FN(executor_fn, TpuStatus_Ok);

  TFTPU_SET_FN(executor_fn, TpuStreamExecutorConfig_Default);
  TFTPU_SET_FN(executor_fn, TpuStreamExecutorConfig_SetOrdinal);
  TFTPU_SET_FN(executor_fn, TpuStreamExecutorConfig_Free);

  TFTPU_SET_FN(executor_fn, TpuDeviceDescription_New);
  TFTPU_SET_FN(executor_fn, TpuDeviceDescription_Free);

  TFTPU_SET_FN(executor_fn, TpuExecutor_CreateDeviceDescription);
  TFTPU_SET_FN(executor_fn, TpuExecutor_NewDeviceOptions);
  TFTPU_SET_FN(executor_fn, TpuExecutor_FreeDeviceOptions);
  TFTPU_SET_FN(executor_fn, TpuExecutor_HostCallback);

  TFTPU_SET_FN(executor_fn, TpuTransferManager_New);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_Free);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_PlatformId);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_HostShapeToDeviceShape);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_TransferLiteralToDeviceAsync);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_TransferLiteralFromDevice);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_GetByteSizeRequirement);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_WriteSingleTupleIndexTable);

  TFTPU_SET_FN(executor_fn, TpuComputationPlacer_New);
  TFTPU_SET_FN(executor_fn, TpuComputationPlacer_Free);

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

  TFTPU_SET_FN(util_fn, TpuCompile_IsTpuCompilationEnabled);
  TFTPU_SET_FN(util_fn, TpuCompile_ShouldTpuCompileOpIgnoreCancellation);
  TFTPU_SET_FN(util_fn, TpuCompile_ToTpuShapeRepresentation);

  return tensorflow::Status::OK();
}

tensorflow::Status InitializeTpuStructFns(void* library_handle) {
  TF_RETURN_IF_ERROR(SetTpuConfigStructFns(library_handle));
  TF_RETURN_IF_ERROR(SetTpuMeshStateStructFns(library_handle));
  TF_RETURN_IF_ERROR(SetCompileStructFn(library_handle));
  TF_RETURN_IF_ERROR(SetExecutorStructFn(library_handle));
  TF_RETURN_IF_ERROR(SetTpuNodeContextStructFns(library_handle));
  TF_RETURN_IF_ERROR(SetTpuUtilStructFns(library_handle));

  return tensorflow::Status::OK();
}

}  // namespace