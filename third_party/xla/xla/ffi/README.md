# XLA FFI

This is the next generation of XLA custom calls with rich type-safe APIs.

https://en.wikipedia.org/wiki/Foreign_function_interface

```
A foreign function interface (FFI) is a mechanism by which a program written in
one programming language can call routines or make use of services written or
compiled in another one. An FFI is often used in contexts where calls are made
into binary dynamic-link library.
```

XLA FFI is a mechanism by which an XLA program can call functions compiled with
another programming language using a stable C API (which guarantees ABI
compatibility between XLA and external functions). XLA FFI also provides a C++
header-only library that hides all the details of underlying C API from the
user.

**WARNING:** Under construction. We already have a rich type-safe custom call
mechanism for XLA runtime. However, it doesn't provide a stable C API. XLA FFI
aims to replicate the usability of XLA runtime's custom calls with a stable
C API.