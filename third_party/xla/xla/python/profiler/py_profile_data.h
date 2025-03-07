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

#ifndef XLA_PYTHON_PROFILER_PY_PROFILE_DATA_H_
#define XLA_PYTHON_PROFILER_PY_PROFILE_DATA_H_

#include <nanobind/nanobind.h>

namespace xla {

void BuildProfileDataSubmodule(nanobind::module_& parent_module);

}  // namespace xla

#endif  // XLA_PYTHON_PROFILER_PY_PROFILE_DATA_H_
