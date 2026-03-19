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

#ifndef XLA_BACKENDS_GPU_PROFILER_KERNEL_NAME_TRACER_FACTORY_H_
#define XLA_BACKENDS_GPU_PROFILER_KERNEL_NAME_TRACER_FACTORY_H_

#include <memory>

#include "xla/backends/gpu/profiler/kernel_name_tracer.h"

namespace xla::gpu {

// This trait identifies the factory function that creates a platform-specific
// KernelNameTracer in the platform object registry.
// Don't use it directly. Use KernelNameTracer::Create instead.
struct KernelNameTracerFactory {
  using Type = std::unique_ptr<KernelNameTracer> (*)();
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_PROFILER_KERNEL_NAME_TRACER_FACTORY_H_
