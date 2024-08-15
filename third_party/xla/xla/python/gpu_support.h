/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_GPU_SUPPORT_H_
#define XLA_PYTHON_GPU_SUPPORT_H_

#include "nanobind/nanobind.h"

namespace xla {

// Registers `make_gpu_client` and `GpuAllocatorConfig` in the parameter
// nanobind module.
void RegisterGpuClientAndDefineGpuAllocatorConfig(nanobind::module_& m_nb);

}  // namespace xla

#endif  // XLA_PYTHON_GPU_SUPPORT_H_
