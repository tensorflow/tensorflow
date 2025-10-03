# PJRT C API Client

This directory contains the "top sandwich" of the PJRT C API. This is a
thick client wrapper that can be used to interface with plugins presenting
the PJRT C API.

The primary purpose of the code in this folder is to provide a C++
object-oriented interface (`PjRtClient`, `PjRtDevice`, etc.) on top of the
purely C-style PJRT C API. This allows existing XLA/JAX code that uses the
`PjRt` C++ interface to interact with PJRT providers implemented as C API
plugins.

The key classes implemented in this folder, wrapping the PJRT C API,
include:

*   `PjRtCApiClient`: Implements the `PjRtClient` interface, serving as the
    main entry point for interacting with a PJRT C API plugin. It wraps the
    `PJRT_Client` structure.
*   `PjRtCApiDeviceDescription`: Implements `PjRtDeviceDescription`,
    wrapping the `PJRT_DeviceDescription` structure to provide details about
    a device.
*   `PjRtCApiDevice`: Implements `PjRtDevice`, representing a single device
    managed by the `PjRtCApiClient`. It wraps the `PJRT_Device` structure.
*   `PjRtCApiMemorySpace`: Implements `PjRtMemorySpace`, wrapping the
    `PJRT_Memory` structure to represent a memory space on a device.
*   `PjRtCApiExecutable`: Implements `PjRtExecutable`, representing a
    compiled but not yet loaded executable. It wraps the `PJRT_Executable`
    structure.
*   `PjRtCApiLoadedExecutable`: Implements `PjRtLoadedExecutable`,
    representing an executable loaded onto devices and ready for execution.
    It wraps the `PJRT_LoadedExecutable` structure.
*   `PjRtCApiBuffer`: Implements `PjRtBuffer`, representing a device buffer.
    It wraps the `PJRT_Buffer` structure. Internally, it uses `PJRT_Event`
    (wrapped by `pjrt::ConvertCEventToCppFuture`) to manage buffer
    readiness.
*   `PjRtCApiTopologyDescription`: Implements `PjRtTopologyDescription`,
    representing the device topology. It wraps the `PJRT_TopologyDescription`
    structure.
*   `PjRtCApiCompiler`: Implements `PjRtCompiler`, providing Ahead-of-Time
    (AOT) compilation capabilities by wrapping C API functions like
    `PJRT_Compile` and `PJRT_TopologyDescription_Deserialize`.
*   `PjRtCApiAsyncHostToDeviceTransferManager`: Implements
    `PjRtClient::AsyncHostToDeviceTransferManager`, wrapping the
    `PJRT_AsyncHostToDeviceTransferManager` structure for asynchronous
    host-to-device transfers.
*   `PjRtCApiExternalReference`: A helper class used by `PjRtCApiBuffer` to
    manage external reference counts on `PJRT_Buffer` objects.
*   `CApiCopyToDeviceStream`: Implements `CopyToDeviceStream`, wrapping the
    `PJRT_CopyToDeviceStream` structure used in receive callbacks for
    cross-host transfers.

These `PjRtCApi*` and `CApi*` classes translate calls from the standard
XLA/JAX C++ `PjRt` interfaces into the corresponding C API function calls,
effectively acting as an adapter layer.
