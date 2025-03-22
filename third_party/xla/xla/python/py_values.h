/* Copyright 2020 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Helpers for converting Python values into buffers.

#ifndef XLA_PYTHON_PY_VALUES_H_
#define XLA_PYTHON_PY_VALUES_H_

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/nb_numpy.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"

namespace xla {

struct DevicePutResult {
  DevicePutResult(tsl::RCReference<ifrt::Array> ifrt_array, bool weak_type)
      : ifrt_array_or_host_buffer(std::move(ifrt_array)),
        weak_type(weak_type),
        // host_buffer_semantics is not meaningful when
        // `ifrt_array_or_host_buffer` is an IFRT Array.
        host_buffer_semantics(
            ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall) {}

  DevicePutResult(ifrt::Client::HostBuffer ifrt_host_buffer, bool weak_type,
                  ifrt::Client::HostBufferSemantics host_buffer_semantics)
      : ifrt_array_or_host_buffer(std::move(ifrt_host_buffer)),
        weak_type(weak_type),
        host_buffer_semantics(host_buffer_semantics) {}

  // Disallow copy since copying `DevicePutResult` without holding GIL may be
  // dangerous due to `owning_pybuffer`.
  DevicePutResult(const DevicePutResult&) = delete;
  DevicePutResult& operator=(const DevicePutResult&) = delete;
  DevicePutResult(DevicePutResult&&) noexcept = default;
  DevicePutResult& operator=(DevicePutResult&&) noexcept = default;

  ifrt::DType ifrt_dtype() const;
  const ifrt::Shape& ifrt_shape() const;

  // Points to the on-device array or on-host buffer.
  std::variant<tsl::RCReference<ifrt::Array>, ifrt::Client::HostBuffer>
      ifrt_array_or_host_buffer;
  bool weak_type;
  ifrt::Client::HostBufferSemantics host_buffer_semantics;
};

// Copies a buffer-like object to be on device.
//
// If `arg` is not convertible to a `PjRtBuffer` from C++, an error will be
// returned; float0s are not supported yet.
// If the value is known to be a PyBuffer object, py_buffer can be passed as
// an optimization to avoid a Python->C++ cast.
//
// This function performs Python work inline but postpones C++ work until the
// returned function is called. The returned function must be called after
// releasing GIL. Useful for batching GIL release when there are many device_put
// to execute.
//
// May throw exceptions from nanobind in addition to failing via an error
// absl::Status. (We could catch these if needed, but there seems little point.)
struct DevicePutOptions {
  bool squash_64bit_types = false;
  bool allow_zero_copy = true;
};
using DevicePutResultFn =
    absl::AnyInvocable<absl::StatusOr<DevicePutResult>() &&>;
absl::StatusOr<DevicePutResultFn> DevicePut(nanobind::handle arg,
                                            ifrt::Client* client,
                                            ifrt::Device* to_device,
                                            const DevicePutOptions& options,
                                            ifrt::MemoryKind to_memory_kind);

// Runs `device_put_result_fn` for a single addressable shard, and constructs an
// IFRT Array from the per-shard array or host buffer. The return value is a
// constructed IFRT Array.
//
// This function requires less array metadata and has lower overhead than
// `MakeIfrtArrayFromBatchedDevicePut` as it is specialized for a single
// addressable shard.
//
// `sharding` determines the sharding of the returned IFRT Array.
//
// Requires GIL. Will release GIL while calling `device_put_result_fn` and
// constructing the IFRT Array.
absl::StatusOr<tsl::RCReference<ifrt::Array>> MakeIfrtArrayFromDevicePut(
    ifrt::Client* ifrt_client, nanobind::handle sharding,
    DevicePutResultFn device_put_result_fn);

// Runs `device_put_result_fns` for each addressable shard, and constructs an
// IFRT Array from these per-shard arrays or host buffers. The return value is a
// constructed IFRT Array.
//
// `shape` and `sharding` determine the shape and sharding of the returned
// IFRT Array.
//
// Requires GIL. Will release GIL while calling `device_put_result_fns` and
// constructing the IFRT Array.
absl::StatusOr<tsl::RCReference<ifrt::Array>> MakeIfrtArrayFromBatchedDevicePut(
    ifrt::Client* ifrt_client, nb_dtype dtype, absl::Span<const int64_t> shape,
    nanobind::handle sharding,
    absl::Span<DevicePutResultFn> device_put_result_fns);

// Returns `true` if `arg` is a JAX float0 array.
bool IsFloat0(xla::nb_numpy_ndarray arg);

// Describes the abstract shape and dtype of an argument.
struct PyArgSignature {
  PyArgSignature(PrimitiveType dtype, absl::Span<const int64_t> shape,
                 bool weak_type)
      : dtype(dtype), shape(shape.begin(), shape.end()), weak_type(weak_type) {}
  // This is the XLA dtype of the object.
  const PrimitiveType dtype;
  const absl::InlinedVector<int64_t, 4> shape;
  // JAX arguments can be of weak type, if and only if they are Python scalars
  // or `DeviceArray` values such that `aval.weak_type` is true.
  const bool weak_type;
  bool operator==(const PyArgSignature& other) const {
    return std::tie(dtype, weak_type, shape) ==
           std::tie(other.dtype, other.weak_type, other.shape);
  }
  bool operator!=(const PyArgSignature& other) const {
    return !(*this == other);
  }
  std::string DebugString() const;
};

// Returns the PyArgSignature associated with an argument. Returns an error if
// the argument is not supported.
absl::StatusOr<PyArgSignature> PyArgSignatureOfValue(nanobind::handle arg,
                                                     bool jax_enable_x64);

template <typename H>
H AbslHashValue(H h, const xla::PyArgSignature& s) {
  h = H::combine(std::move(h), s.dtype);
  h = H::combine_contiguous(std::move(h), s.shape.data(), s.shape.size());
  return h;
}
}  // namespace xla

#endif  // XLA_PYTHON_PY_VALUES_H_
