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

#include <Python.h>

#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "xla/python/gpu_support.h"
#include "xla/python/logging.h"
#include "xla/python/py_client.h"  // IWYU pragma: keep

namespace xla {
namespace {

namespace nb = nanobind;

}  // namespace

NB_MODULE(xla_gpu_extension, m_nb) {
  // Initialize ABSL logging because code within XLA uses it.
#ifndef PLATFORM_GOOGLE
  InitializeAbslLogging();
#endif  // PLATFORM_GOOGLE

  // We seem to get a fair number of leak warnings from nanobind. It's unclear
  // whether these are false positives or not.
  nb::set_leak_warnings(false);

  RegisterGpuClientAndDefineGpuAllocatorConfig(m_nb);
}

}  // namespace xla
