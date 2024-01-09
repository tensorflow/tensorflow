/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_RUNTIME_MEMREF_VIEW_H_
#define XLA_RUNTIME_MEMREF_VIEW_H_

#include <cstdint>

#include "absl/types/span.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace runtime {

// A view into the memref argument. Corresponds to the MemrefDesc, however it
// doesn't own the sizes/strides vectors, and cheap to pass around. Memrefs with
// non-identity layouts can be decoded only as a StridedMemrefView.
struct StridedMemrefView {
  PrimitiveType dtype;
  void* data;
  absl::Span<const int64_t> sizes;
  absl::Span<const int64_t> strides;
};

// A view into the memref argument with an identity (row major) layout.
struct MemrefView {
  PrimitiveType dtype;
  void* data;
  absl::Span<const int64_t> sizes;
};

// A flat view into memref argument with an identity (row major) layout. If the
// memref shape and strides are not required for the custom call, it's cheaper
// to pass the flat view.
struct FlatMemrefView {
  PrimitiveType dtype;
  void* data;
  int64_t size_in_bytes;
};

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_MEMREF_VIEW_H_
