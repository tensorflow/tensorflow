# PJRT C API changelog

## 0.98

* Added `PJRT_Buffer_Bitcast`.

## 0.97

* Add `multi_slice_config` field to `PJRT_ExecuteOptions`.

## 0.96

* Added `PJRT_LoadedExecutable_AddressableDeviceLogicalIds`.

## 0.95

* Added `PJRT_Client_Load`.

## 0.94

* Added types S1 and U1.

## 0.93

* Added `PJRT_Layouts_PJRT_Executable_GetParameterLayouts`. This extension API
  complements `PJRT_Layouts_PJRT_Executable_GetOutputLayouts`.

## 0.92

* Added `PJRT_Device_GetAttributes`.

## 0.91

* Added `PJRT_Extension_Type::PJRT_Extension_Type_Shardings`. This extension
  defines `PJRT_Executable_ParameterShardings` and
  `PJRT_Executable_OutputShardings`. They are currently optional API; if the
  plugin does not support shardings, they can be `nullptr`, or set the
  `shardings` and `sharding_sizes` fields of
  `PJRT_Executable_ParameterShardings_Args` and
  `PJRT_Executable_OutputShardings_Args` to `nullptr`.

## 0.90

* Added `total_size_in_bytes` field to `PJRT_Executable_GetCompiledMemoryStats_Args`.

## 0.89

* Add `PJRT_Event_Create` and `PJRT_Event_Set`.

## 0.88

* Add `PJRT_Buffer_DonateWithControlDependency`.

## 0.87

* Add `PJRT_Executable_GetCompileOptions`.

## 0.86

* Add `PJRT_Device_CreateAsyncTrackingEvent`.

* Add `PJRT_AsyncTrackingEvent_Destroy`.

## 0.85

* Add `PJRT_Device_PoisonExecution`.

## 0.84

* Add `PJRT_Buffer_CopyRawToHostFuture`.

## 0.83

* Add `PJRT_AsyncHostToDeviceTransferManager_TransferLiteral`.

## 0.82

* Add `PJRT_Client_CreateErrorBuffer`.

## 0.81

* Added `PJRT_Layouts_PJRT_Executable_GetOutputLayouts`.

## 0.80

* Added `PJRT_Extension_Type::PJRT_Extension_Type_HostAllocator`.

## 0.79

* Added `PJRT_LoadedExecutable_GetDeviceAssignment.`

## 0.78

* Add incarnations to `PJRT_ExecuteOptions`.

## 0.77

* Added buffer aliasing support to the PJRT C API

## 0.76

* Added `call_location` to `PJRT_ExecuteOptions`

## 0.75

* Added `PJRT_TopologyDescription_Deserialize`.

## 0.74

* Added `PJRT_Extension_Type::PJRT_Extension_Type_CrossHostTransfers`.

## 0.73

* Added `PJRT_Client_UpdateGlobalProcessInfo`.

## 0.72

* Added `peak_memory_in_bytes` to `CompiledMemoryStats`.

## 0.71

*   Added `overridden_serialized_compile_options` and
    `overridden_serialized_compile_options_size` fields to
    `PJRT_Executable_DeserializeAndLoad_Args`.

## 0.70

* Sharding ops may appear directly in the payload (previously they were
  serialized in `custom_calls`).

## 0.69

*   Implemented PjRtClient::CreateUninitializedBuffer

## 0.68

* Changed the type of ``topology`` in
  ``PJRT_TopologyDescription_PlatformName_Args`` and
  ``PJRT_TopologyDescription_GetDeviceDescriptions_Args`` from
  ``PJRT_TopologyDescription*`` to ``const PJRT_TopologyDescription*``.

## 0.67
* Added ``PJRT_Client_DmaMap`` and ``PJRT_Client_DmaUnmap``.

## 0.66
* Added ``memory`` field of type ``PJRT_Memory *`` in ``PJRT_Client_CreateViewOfDeviceBuffer_Args``.
  The new field should be preferred over ``device``, which is now deprecated.

## 0.65
* Added ``PJRT_Triton_Extension``.

## 0.64
* Added ``context`` field of type ``PJRT_ExecuteContext *`` in ``PJRT_ExecuteOptions``.

## 0.63
*   Added types F4E2M1FN and F8E8M0FNU.

## 0.62
* Added more member functions for ``PJRT_AsyncHostToDeviceTransferManager``.

## 0.61
* Added ``PJRT_KeyValueTryGet`` to the KV store interface,
  which is non-blocking and immediately returns an error if the
  key is not found.

## 0.60
* Added ``PJRT_Client_CreateBuffersForAsyncHostToDevice`` and ``PJRT_AsyncHostToDeviceTransferManager_TransferRawDataToSubBuffer``.

## 0.59
* Added ``PJRT_MemoryDescriptions_Extension``.

## 0.57
* Rearranged fields in the PJRT_Api
* Update outdated struct sizes from previous changes to
  ``PJRT_Client_TopologyDescription`` and ``PJRT_Buffer_CopyRawToHost``.

## 0.56 (Nov 11, 2024)
* Added ``PJRT_Buffer_CopyRawToHost``

## 0.55
* Added types F8E4M3 and F8E3M4.

## 0.54
* Deprecated PJRT_Buffer_GetMemoryLayout.

## 0.53
* Added ``PJRT_FFI_Extension` extension to support passing user data to FFI
  handlers on compatible PJRT backends.

## 0.52
* Added ``PJRT_ExecuteContext`` struct corresponding to ``xla::ExecuteContext``.

## 0.51
* Added ``PJRT_Extension_Type::PJRT_Extension_Type_Layouts``.

## 0.50 (Apr 26, 2024)
* Added a new type ``PJRT_Buffer_Type_TOKEN`` to ``PJRT_Buffer_Type``.

## 0.49 (Apr 19, 2024)
* Added ``PJRT_Extension_Type::PJRT_Extension_Type_Stream``.

## 0.48 (Apr 10, 2024)
* Added ``PjRtCApiMemorySpace::kind_id`` for uniquely identifying memory space kinds.
* Renamed memory space kind to ``PjRtCApiMemorySpace::memory_space_kind`` to
  ``PjRtCApiMemorySpace::kind``.
* Added new host buffer semantics enum
  ``PJRT_HostBufferSemantics_kMutableZeroCopy``

## 0.47 (Mar 29, 2024)
* Added ``PJRT_Extension_Type::PJRT_Extension_Type_Custom_Partitioner``.
* Renamed host buffer semantics enum from ``PJRT_HostBufferSemantics_kZeroCopy``
  to ``PJRT_HostBufferSemantics_kImmutableZeroCopy``.

## 0.46 (Feb 29, 2024)
* Update outdated struct sizes from previous changes to
  ``PJRT_Device_AddressableMemories_Args`` and ``PJRT_ExecuteOptions``.

## 0.45 (Feb 27, 2024)
* Breaking changes
  * Added struct_size field to beginning of PJRT_Extension_Base. This is so
    forwards and backwards compatibility logic can be implemented with extension
    structs.

## 0.44 (Feb 26, 2024)
* Changed all ``void*`` extension fields to have type ``PJRT_Extension_Base*``

## 0.43 (Feb 24, 2024)
* Added some new fields to PJRT_Executable_GetCompiledMemoryStats

## 0.42 (Feb 13, 2024)
* Renamed all ``priv`` fields to ``extension_start``

## 0.41 (Feb 13, 2024)
* Renamed PJRT_Structure_Base to PJRT_Extension_Base
* Renamed PJRT_Structure_Type to PJRT_Extension_Type (and similarly for enum fields)

## 0.40 (Nov 27, 2023)
* Added PJRT_Executable_GetCompiledMemoryStats.

## 0.39 (Nov 16, 2023)
* Add non_donatable_input_indices and num_non_donatable_input_indices to
PJRT_ExecuteOptions.

## 0.38 (Oct 30, 2023)
* Use `enum` to define STRUCT_SIZE constants in a header file.

## 0.37 (Oct 27, 2023)
* Added const to a bunch of lists and value types.

## 0.36 (Oct 24, 2023)
* Added PJRT_Client_TopologyDescription

## 0.35 (Oct 20, 2023)
* Added PJRT_Executable_Fingerprint method
* Deprecated PJRT_LoadedExecutable_Fingerprint

## 0.34 (Oct 9, 2023)
* Added PJRT_Extension_Type::PJRT_Extension_Type_Profiler.

## 0.33 (Oct 3, 2023)
* Added PJRT_Client_CreateViewOfDeviceBuffer.

## 0.32 (Sep 26, 2023)
* Added PJRT_Buffer_CopyToMemory.

## 0.31 (Sep 22, 2023)
* Added PJRT_Extension_Base.
* Added PJRT_Extension_Type.
* Renamed PJRT_Api.extension_start to PJRT_Api.extension_start.

## 0.30 (Sep 14, 2023)
* Added PJRT_NamedValue_Type::PJRT_NamedValue_kBool.

## 0.29 (Sep 6, 2023)
* Added PJRT_Executable_OutputElementTypes.
* Added PJRT_Executable_OutputDimensions.
