/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_CLIENT_IMPL_UTIL_H_
#define XLA_PYTHON_IFRT_CLIENT_IMPL_UTIL_H_

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// Portable adapter for `MakeArraysFromHostBufferShards`. It breaks downs
// requests into `MakeArrayFromHostBuffer` calls followed by
// `AssembleArrayFromSingleDeviceArrays`.
//
// TODO(hyeontaek): Remove this adapter once all major IFRT implementations
// natively support `MakeArraysFromHostBufferShards`.
absl::StatusOr<std::vector<tsl::RCReference<Array>>>
ClientMakeArraysFromHostBufferShards(
    Client* client,
    absl::Span<Client::MakeArraysFromHostBufferShardsSpec> specs,
    Client::HostBufferSemantics semantics,
    tsl::RCReference<UserContext> user_context);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_CLIENT_IMPL_UTIL_H_
