# IFRT Proxy Protocol Versions

## Version 1

*   Added date: 2023-12-20.
*   Changes:
    *   Initial version.

## Version 2

*   Added date: 2024-05-31.
*   Changes:
    *   Added support for `Client::GetReadyFuture()`.

## Version 3

*   Added date: 2024-06-17.
*   Changes:
    *   Added native support for `Client::CopyArrays()`.

## Version 4

*   Added date: 2024-06-18.
*   Changes:
    *   Changed the serialization of client and device attributes to use `xla.ifrt.AttributeMapProto` instead of `map<string, xla.ifrt.proto.Variant>`.

## Version 5

*   Added date: 2024-09-20.
*   Changes:
    *   Batch array deletions and destruction on client before sending to server.

## Version 6

*   Added date: 2024-09-30.
*   Changes:
    *   Added `ExecuteOptions::fill_status`.

## Version 7

*   Added date: 2024-10-01.
*   Changes:
    *   Added support for `Client::GetAllDevices()`.

## Version 8

*   Added date: 2024-10-11.
*   Changes:
    *   Added support for `SingleDeviceShardSemantics` in Array assembly and disassembly operations.

## Version 9

*   Added date: 2024-10-31.
*   Changes:
    *   Added support for string Arrays (i.e., arrays with dtype `DType::kString`).

## Version 10

*   Added date: 2024-11-08.
*   Changes:
    *   MakeArrayFromHostBuffer uses client-generated array handles and sends data asynchronously.

## Version kClientHandlesOptimization2

*   Added date: 2024-11-19
*   Changes:
    *   Introduces a set of performance optimizations where the client generates array handles.

## Version kClientHandlesExecutableOptimization

*   Added date: 2024-11-26
*   Changes:
    *   Client generates array handles for execute requests.

## Version kAssembleArrayFromSingleDeviceArraysWithDType

*   Added date: 2025-02-11
*   Changes:
    *   Added support for `Client::AssembleArrayFromSingleDeviceArrays` that
    takes `DType`.

## Version kMakeArraysFromHostBufferShards

*   Added date: 2025-03-12
*   Changes:
    *   Added support for `Client::MakeArraysFromHostBufferShards()`.
