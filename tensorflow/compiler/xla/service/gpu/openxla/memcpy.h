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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_OPENXLA_MEMCPY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_OPENXLA_MEMCPY_H_

#include "third_party/iree/runtime/src/iree/hal/api.h"  // IWYU pragma: keep
#include "third_party/iree/runtime/src/iree/vm/api.h"   // IWYU pragma: keep
#include "tensorflow/compiler/xla/service/gpu/openxla/vm.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla::gpu {

//===-----------------------------------------------------------------------===/
// XLA:GPU memcpy API
//===-----------------------------------------------------------------------===/

Status DispatchMemcpyD2D(const vm::ExecutionContext& ctx,
                         iree_hal_allocator_t* device_allocator,
                         iree_hal_buffer_view_t* dst,
                         iree_hal_buffer_view_t* src);

//===-----------------------------------------------------------------------===/
// XLA:GPU custom module memcpy API
//===-----------------------------------------------------------------------===/

namespace vm {

class MemcpyAPI {
 public:
  explicit MemcpyAPI(iree_hal_allocator_t* device_allocator);

  iree::Status MemcpyD2D(iree::vm::ref<ExecutionContext> ctx,
                         iree::vm::ref<iree_hal_buffer_view_t> dst,
                         iree::vm::ref<iree_hal_buffer_view_t> src);

 private:
  iree_hal_allocator_t* device_allocator_;
};

}  // namespace vm
}  // namespace xla::gpu

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_OPENXLA_MEMCPY_H_
