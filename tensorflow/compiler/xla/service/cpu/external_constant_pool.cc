/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/external_constant_pool.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace xla {
namespace cpu {
void ExternalConstantPool::Insert(string name, const LiteralSlice& literal,
                                  int64 alignment) {
  CHECK(!ShapeUtil::IsTuple(literal.shape()));
  CHECK(alignment > 0 && IsPowerOfTwo(static_cast<uint64>(alignment)));
  CHECK(entries_.find(name) == entries_.end());

  const int64 literal_size = ShapeUtil::ByteSizeOf(literal.shape());
  void* raw_pointer = tensorflow::port::AlignedMalloc(
      literal_size, std::max<size_t>(alignment, sizeof(void*)));
  CHECK(raw_pointer != nullptr) << "failed to allocate " << literal_size
                                << " bytes with alignment of " << alignment;

  std::memcpy(raw_pointer, literal.untyped_data(), literal_size);
  entries_.emplace(std::move(name), static_cast<uint8*>(raw_pointer));
}

const uint8* ExternalConstantPool::Find(const string& name) {
  auto it = entries_.find(name);
  return it == entries_.end() ? nullptr : it->second.get();
}
}  // namespace cpu
}  // namespace xla
