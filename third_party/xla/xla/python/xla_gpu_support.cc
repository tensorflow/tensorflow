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

#include "absl/strings/string_view.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "xla/pjrt/status_casters.h"
#include "xla/pjrt/triton.h"
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

  nb::class_<triton::CompilationResult>(m_nb, "TritonCompilationResult")
      .def_ro("asm", &triton::CompilationResult::asm_text)
      .def_ro("smem_bytes", &triton::CompilationResult::smem_bytes)
      .def_ro("cluster_dim_x", &triton::CompilationResult::cluster_dim_x)
      .def_ro("cluster_dim_y", &triton::CompilationResult::cluster_dim_y)
      .def_ro("cluster_dim_z", &triton::CompilationResult::cluster_dim_z);

  m_nb.def("compile_triton_to_asm",
           [](nb::bytes module, nb::str arch_name, int num_warps, int num_ctas,
              int num_stages) {
             return xla::ValueOrThrow(xla::triton::Compile(
                 absl::string_view(static_cast<const char*>(module.data()),
                                   module.size()),
                 arch_name.c_str(), num_warps, num_ctas, num_stages));
           });
}

}  // namespace xla
