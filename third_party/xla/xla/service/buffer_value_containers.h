/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_BUFFER_VALUE_CONTAINERS_H_
#define XLA_SERVICE_BUFFER_VALUE_CONTAINERS_H_

#include "absl/container/flat_hash_set.h"
#include "xla/service/buffer_value.h"
#include "xla/service/logical_buffer.h"
#include "tsl/lib/gtl/compactptrset.h"

namespace xla {

// Define various containers of BufferValues, and utilities to convert from
// containers of LogicalBuffers to containers of BufferValues.

using BufferValueCompactPointerSet =
    tsl::gtl::CompactPointerSet<const BufferValue*>;
template <class LogicalBufferContainerT>
BufferValueCompactPointerSet ToBufferValueCompactPointerSet(
    const LogicalBufferContainerT& logical_buffer_container) {
  BufferValueCompactPointerSet output;
  for (const LogicalBuffer* buffer : logical_buffer_container) {
    output.insert(buffer);
  }
  return output;
}

using BufferValueFlatSet = absl::flat_hash_set<const BufferValue*>;
template <class LogicalBufferContainerT>
BufferValueFlatSet ToBufferValueFlatSet(
    const LogicalBufferContainerT& logical_buffer_container) {
  BufferValueFlatSet output;
  output.reserve(logical_buffer_container.size());
  for (const LogicalBuffer* buffer : logical_buffer_container) {
    output.insert(buffer);
  }
  return output;
}

}  // namespace xla

#endif  // XLA_SERVICE_BUFFER_VALUE_CONTAINERS_H_
