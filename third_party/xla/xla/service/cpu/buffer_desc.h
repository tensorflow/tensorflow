/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_BUFFER_DESC_H_
#define XLA_SERVICE_CPU_BUFFER_DESC_H_

#include <cstddef>

namespace xla {
namespace cpu {

// BufferDesc for passing raw `buffer` (i.e. void ptr + size) arguments.
class BufferDesc {
 public:
  BufferDesc(void* data, size_t size) : data_(data), size_(size) {}
  void* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  void* data_;
  size_t size_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_BUFFER_DESC_H_
