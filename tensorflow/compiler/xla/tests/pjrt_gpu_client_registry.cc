/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "tensorflow/compiler/xla/tests/pjrt_client_registry.h"

namespace xla {
namespace {

// Register a GPU PjRt client for tests.
const bool kUnused =
    (RegisterPjRtClientTestFactory([]() {
       xla::GpuAllocatorConfig gpu_config;
       gpu_config.kind = xla::GpuAllocatorConfig::Kind::kDefault;
       gpu_config.preallocate = true;
       gpu_config.memory_fraction = 0.08;
       return GetStreamExecutorGpuClient(/*asynchronous=*/true, gpu_config,
                                         /*distributed_client=*/nullptr,
                                         /*node_id=*/0);
     }),
     true);

}  // namespace
}  // namespace xla
