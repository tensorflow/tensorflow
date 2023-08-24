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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_MODULE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_MODULE_H_

#include "third_party/iree/runtime/src/iree/hal/api.h"  // IWYU pragma: keep
#include "third_party/iree/runtime/src/iree/vm/api.h"   // IWYU pragma: keep

namespace xla::gpu {

// Creates XLA:GPU custom module implementing StreamExecutor integration.
iree_status_t CreateXlaGpuModule(iree_vm_instance_t* instance,
                                 iree_allocator_t host_allocator,
                                 iree_hal_allocator_t* device_allocator,
                                 iree_vm_module_t** out_module);

// Register XLA:GPU custom module types with the IREE VM.
iree_status_t RegisterXlaGpuTypes(iree_vm_instance_t* instance);

}  // namespace xla::gpu

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_MODULE_H_
