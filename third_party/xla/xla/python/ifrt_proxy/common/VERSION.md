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
