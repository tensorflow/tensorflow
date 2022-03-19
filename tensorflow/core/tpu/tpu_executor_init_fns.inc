namespace {

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
  TFTPU_SET_FN(executor_fn, TpuPlatform_GetTopologyPtr);
  TFTPU_SET_FN(executor_fn, TpuPlatform_GetHostLocation);
  TFTPU_SET_FN(executor_fn, TpuPlatform_GetRuntimeVersion);

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
  TFTPU_SET_FN(executor_fn, TpuExecutor_GetCoreLocation);
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
  TFTPU_SET_FN(executor_fn, TpuExecutor_UnloadAllPrograms);
  TFTPU_SET_FN(executor_fn, TpuExecutor_EnqueueCompactionOnStreamForHbm);

  TFTPU_SET_FN(executor_fn, TpuStream_New);
  TFTPU_SET_FN(executor_fn, TpuStream_Free);
  TFTPU_SET_FN(executor_fn, TpuStream_Stream);
  TFTPU_SET_FN(executor_fn, TpuStream_Status);
  TFTPU_SET_FN(executor_fn, TpuStream_IsSameSharedMemoryLocation);
  TFTPU_SET_FN(executor_fn, TpuStream_EnqueueTransferHostToDevice);
  TFTPU_SET_FN(executor_fn, TpuStream_EnqueueTransferDeviceToHost);
  TFTPU_SET_FN(executor_fn, TpuStream_TpuEnqueueOnDeviceSendRecvLocal);

  TFTPU_SET_FN(executor_fn, TpuEvent_New);
  TFTPU_SET_FN(executor_fn, TpuEvent_Free);

  TFTPU_SET_FN(executor_fn, TpuTimer_New);
  TFTPU_SET_FN(executor_fn, TpuTimer_Free);
  TFTPU_SET_FN(executor_fn, TpuTimer_Nanoseconds);
  TFTPU_SET_FN(executor_fn, TpuTimer_Microseconds);

  TFTPU_SET_FN(executor_fn, TpuStatus_New);
  TFTPU_SET_FN(executor_fn, TpuStatus_Create);
  TFTPU_SET_FN(executor_fn, TpuStatus_Set);
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
  TFTPU_SET_FN(executor_fn, TpuTransferManager_ChooseCompactLayoutForShape);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_CanShapedBufferBeAccessedNow);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_CanBufferBeAccessedNow);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_WriteSingleTupleIndexTable);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_GetInfeedLayout);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_LinearizeToBuffers);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_FreeBuffers);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_TransferLiteralToInfeed);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_TransferBuffersToInfeed);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_TransferLiteralFromOutfeed);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_ResetDevices);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_ReadDynamicShapes);

  TFTPU_SET_FN(executor_fn, TpuComputationPlacer_New);
  TFTPU_SET_FN(executor_fn, TpuComputationPlacer_Free);
  TFTPU_SET_FN(executor_fn, TpuComputationPlacer_AssignDevices);
  TFTPU_SET_FN(executor_fn, TpuComputationPlacer_AssignLocalDevices);

  TFTPU_SET_FN(executor_fn, TpuTopology_LogicalDevicesPerHost);
  TFTPU_SET_FN(executor_fn, TpuTopology_LogicalDevicesPerChip);
  TFTPU_SET_FN(executor_fn, TpuTopology_HostCount);
  TFTPU_SET_FN(executor_fn, TpuTopology_ChipsPerHost);
  TFTPU_SET_FN(executor_fn, TpuTopology_ChipBounds_X);
  TFTPU_SET_FN(executor_fn, TpuTopology_ChipBounds_Y);
  TFTPU_SET_FN(executor_fn, TpuTopology_ChipBounds_Z);
  TFTPU_SET_FN(executor_fn, TpuTopology_HasChip);
  TFTPU_SET_FN(executor_fn, TpuTopology_CoreForId);
  TFTPU_SET_FN(executor_fn, TpuTopology_Core);
  TFTPU_SET_FN(executor_fn, TpuTopology_NumCores);
  TFTPU_SET_FN(executor_fn, TpuTopology_Cores);
  TFTPU_SET_FN(executor_fn, TpuTopology_IdForHost);
  TFTPU_SET_FN(executor_fn, TpuTopology_Version);

  TFTPU_SET_FN(executor_fn, TpuCoreLocation_ChipCoordinates);
  TFTPU_SET_FN(executor_fn, TpuCoreLocation_HostCoordinates);
  TFTPU_SET_FN(executor_fn, TpuCoreLocation_Index);
  TFTPU_SET_FN(executor_fn, TpuCoreLocation_Id);

  TFTPU_SET_FN(executor_fn, TpuHostLocation_Id);
  TFTPU_SET_FN(executor_fn, TpuHostLocation_NumCores);
  TFTPU_SET_FN(executor_fn, TpuHostLocation_Cores);

  TFTPU_SET_FN(executor_fn, TpuCompiler_New);
  TFTPU_SET_FN(executor_fn, TpuCompiler_Free);

  TFTPU_SET_FN(executor_fn, TpuCompiler_RunHloPasses);
  TFTPU_SET_FN(executor_fn, TpuCompiler_RunBackend);
  TFTPU_SET_FN(executor_fn, TpuCompiler_Compile);
  TFTPU_SET_FN(executor_fn, TpuCompiler_ShapeSize);
  TFTPU_SET_FN(executor_fn, TpuExecutable_ExecuteAsyncOnStream);
  TFTPU_SET_FN(executor_fn, TpuExecutable_FreeXlaShapeIndexArray);
  TFTPU_SET_FN(executor_fn, TpuExecutable_FreeMaybeOwningDeviceMemoryArray);
  TFTPU_SET_FN(executor_fn, TpuExecutable_Fingerprint);
  TFTPU_SET_FN(executor_fn, TpuExecutable_Serialize);
  TFTPU_SET_FN(executor_fn, TpuExecutableSerialize_GetByteSize);
  TFTPU_SET_FN(executor_fn, TpuExecutableSerialize_WriteToArray);
  TFTPU_SET_FN(executor_fn, TpuExecutableSerialize_FreeHandle);
  TFTPU_SET_FN(executor_fn, TpuExecutable_Deserialize);
  TFTPU_SET_FN(executor_fn, TpuExecutable_HloModule);
  TFTPU_SET_FN(executor_fn, TpuExecutable_Free);

  TFTPU_SET_FN(executor_fn, XlaShapeToTpuShapeRepresentation);
  TFTPU_SET_FN(executor_fn, XlaShapeToTpuPaddedShape);

  TFTPU_SET_FN(executor_fn, TpuAsyncCollectiveOffloadHelper_Init);
  TFTPU_SET_FN(executor_fn, TpuAsyncCollectiveOffloadHelper_Shutdown);

  return tensorflow::Status::OK();
}

}  // namespace
