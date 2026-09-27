/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_FFI_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_FFI_H_

// Platform-independent pieces of the CUB/hipCUB radix sort FFI handlers shared
// by the CUDA and ROCm backends: buffer verification, segment offsets layout
// and the FFI binding signatures. The platform-specific parts (kernel
// dispatch, stream type and host-to-device copy) stay in the backends.

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Verifies that the keys output buffer matches the keys input buffer.
inline absl::Status VerifyCubSortKeysBuffers(
    ffi::AnyBuffer keys, ffi::Result<ffi::AnyBuffer> keys_out) {
  return ffi::Verify("keys output", *keys_out, ffi::match::Buffer().Like(keys));
}

// Verifies that the values input has the shape of the keys input, and that the
// keys and values output buffers match their corresponding inputs.
inline absl::Status VerifyCubSortPairsBuffers(
    ffi::AnyBuffer keys, ffi::AnyBuffer values,
    ffi::Result<ffi::AnyBuffer> keys_out,
    ffi::Result<ffi::AnyBuffer> values_out) {
  ABSL_RETURN_IF_ERROR(ffi::Verify("values input", values,
                              ffi::match::Buffer().WithShapeOf(keys)));
  ABSL_RETURN_IF_ERROR(
      ffi::Verify("keys output", *keys_out, ffi::match::Buffer().Like(keys)));
  return ffi::Verify("values output", *values_out,
                     ffi::match::Buffer().Like(values));
}

// Size in bytes of the segment offsets appended to the end of the scratch
// buffer for batched segmented sort. N pairs of [start_offset, end_offset)
// require (N+1) storage.
inline int64_t GetCubSortOffsetsSize(int64_t batch_size) {
  return (batch_size + 1) * sizeof(int32_t);
}

// Returns host segment offsets [0, segment_size, 2*segment_size, ...] with
// `batch_size + 1` entries, to be copied to the end of the scratch buffer.
inline std::vector<int32_t> MakeCubSortSegmentOffsets(int64_t batch_size,
                                                      int64_t segment_size) {
  std::vector<int32_t> offsets(batch_size + 1);
  for (int32_t i = 0; i <= batch_size; ++i) {
    offsets[i] = i * segment_size;
  }
  return offsets;
}

//===----------------------------------------------------------------------===//
// FFI bindings.
//===----------------------------------------------------------------------===//

// CubSortKeys HLO custom call layout:
//   operands: [keys_in]
//   results:  [keys_out, scratch]
template <typename Binding>
auto BindCubSortKeysArgs(Binding binding) {
  return std::move(binding)
      .template Arg<ffi::AnyBuffer>()          // d_keys_in
      .template Ret<ffi::AnyBuffer>()          // d_keys_out
      .template Ret<ffi::BufferR1<xla::U8>>()  // d_temp_storage
      .template Attr<bool>("descending")
      .template Attr<int64_t>("batch_size");
}

inline auto BindCubSortKeysInstantiate() {
  return BindCubSortKeysArgs(ffi::Ffi::BindInstantiate());
}

template <typename StreamT>
auto BindCubSortKeysExecute() {
  return BindCubSortKeysArgs(ffi::Ffi::Bind())
      .template Ctx<ffi::PlatformStream<StreamT>>();
}

// CubSortPairs HLO custom call layout:
//   operands: [keys_in, values_in]
//   results:  [keys_out, values_out, scratch]
template <typename Binding>
auto BindCubSortPairsArgs(Binding binding) {
  return std::move(binding)
      .template Arg<ffi::AnyBuffer>()          // d_keys_in
      .template Arg<ffi::AnyBuffer>()          // d_values_in
      .template Ret<ffi::AnyBuffer>()          // d_keys_out
      .template Ret<ffi::AnyBuffer>()          // d_values_out
      .template Ret<ffi::BufferR1<xla::U8>>()  // d_temp_storage
      .template Attr<bool>("descending")
      .template Attr<int64_t>("batch_size");
}

inline auto BindCubSortPairsInstantiate() {
  return BindCubSortPairsArgs(ffi::Ffi::BindInstantiate());
}

template <typename StreamT>
auto BindCubSortPairsExecute() {
  return BindCubSortPairsArgs(ffi::Ffi::Bind())
      .template Ctx<ffi::PlatformStream<StreamT>>();
}

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_FFI_H_
