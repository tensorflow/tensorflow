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

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/nb_numpy.h"
#include "xla/xla_data.pb.h"
#include "tsl/concurrency/ref_count.h"

namespace xla {

struct DevicePutResult {
  explicit DevicePutResult(
      tsl::RCReference<ifrt::Array> ifrt_array, bool weak_type,
      nanobind::object owning_pybuffer = nanobind::object())
      : ifrt_array(std::move(ifrt_array)),
        weak_type(weak_type),
        owning_pybuffer(owning_pybuffer) {}

  // Disallow copy since copying `DevicePutResult` without holding GIL may be
  // dangerous due to `owning_pybuffer`.
  DevicePutResult(const DevicePutResult&) = delete;
  DevicePutResult& operator=(const DevicePutResult&) = delete;
  DevicePutResult(DevicePutResult&&) = default;
  DevicePutResult& operator=(DevicePutResult&&) = default;

  // Points to the on-device array. Not owned.
  tsl::RCReference<ifrt::Array> ifrt_array;
  bool weak_type;

  nanobind::object owning_pybuffer;
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
// Status. (We could catch these if needed, but there seems little point.)
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
