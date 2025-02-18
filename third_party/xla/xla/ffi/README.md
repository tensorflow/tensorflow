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

## XLA FFI C API (ABI)

XLA FFI uses C to define API between XLA itself and FFI handlers (custom call
implementations), which acts as an
[ABI](https://en.wikipedia.org/wiki/Application_binary_interface) contract
between XLA and FFI handlers. C structs define the memory layout of all the
data passed across the XLA FFI boundary. It is notoriously hard to guarantee
ABI stability and compatibility for C++ APIs, that's why we fall back on plain
C for XLA FFI.

See `xla/ffi/api/c_api.h` for implementation details. XLA FFI API is inspired by
PJRT C APIs and provides a similar level of backward compatibility.

**WARNING:** XLA FFI in under construction and currently does not provide any
backward compatibility guarantees. Once we reach a point when we are reasonably
confident that we got all APIs right, we will define `XLA_FFI_API_MAJOR` and
`XLA_FFI_API_MINOR` API versions and will start providing API and ABI
backward compatibility.

## XLA FFI External vs Internal APIs

There are two flavors of XLA FFI APIs, and you should use only one of them:

* **External** (`xla/ffi/api/ffi.h`) - for clients that build shared libraries
   with custom call implementation, which are later loaded at run time and
   linked dynamically. These shared libraries can be built with an older XLA
   version, or a different C++ compiler. External FFI API depends only on C API
   that defines memory layout and standard C++ headers. We do not pass any
   pointers to C++ structs, which could be dangerous and lead to undefined
   behavior.

* **Internal** (`xla/ffi/ffi.h`) - for clients that build statically linked
  binaries, and compile everything from the same XLA commit with the same
  compiler toolchain. Internal FFI API uses ABSL and XLA headers to define XLA
  FFI data types. This would be unsafe to do with dynamic linking.

Depending on how exactly your build is structured you should choose one of the
options. If you build XLA + your custom calls into a statically linked artifact,
you might prefer "internal" version as it gives you access to more XLA
internals, for example you can get access to the underlying `se::StreamExecutor`
instance.

If you want to be decoupled from XLA and build and ship your custom calls
library separately, then you should choose "external" XLA FFI API version, and
simply copy XLA FFI headers to your project.

#### Examples:

* `xla::ffi::Error` vs `absl::Status` - it is unsafe to pass `absl::Status`
  across the dynamic library boundary, and external XLA FFI defines its own
  `Error` type which has a very similar API to `absl::Status`. We make sure
  that the enum class that defines error codes has identical underlying value
  as its `absl::StatusCode` counterpart, so we can safely pass it as an
  integral type.

* `xla::ffi::Span` vs `absl::Span` - for the same reason as `xla::ffi::Error`,
  we define our own `Span` type, which has a very similar API to `absl::Span`.
  XLA FFI will migrate to `std::span` when XLA will move to C++ 20.

* `xla::ffi::DataType` vs `xla::PrimitiveType` - similar to `Error`, `DataType`
  is a mirror of the enum defined in XLA.

