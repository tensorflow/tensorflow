/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PMAP_LIB_H_
#define XLA_PYTHON_PMAP_LIB_H_

#include <optional>
#include <utility>
#include <vector>

// placeholder for index annotation headers
#include "third_party/nanobind/include/nanobind/nanobind.h"

// TODO(jblespiau): The current implementation moves the Python logic to C++,
// as a preliminary step to executing the `pmap` execution path from C++.
// It implements the current Python behavior (thus, it may not be optimal, and
// we will be able to modify it later).

namespace jax {

void BuildPmapSubmodule(nanobind::module_& m);

}  // namespace jax

#endif  // XLA_PYTHON_PMAP_LIB_H_
