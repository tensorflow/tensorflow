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

#ifndef XLA_PYTHON_PROFILER_BUILD_PROFILE_DATA_NB_MODULE_H_
#define XLA_PYTHON_PROFILER_BUILD_PROFILE_DATA_NB_MODULE_H_

#include <nanobind/make_iterator.h>  // For automatic conversion of std::iterator to Python iterable.
#include <nanobind/stl/string.h>  // For automatic conversion of std::string to Python string.

#include <memory>
#include <string>

#include "nanobind/nanobind.h"
#include "xla/python/profiler/build_profile_data_nb_module.h"

namespace xla {

void BuildProfileDataClasses(nanobind::module_& m);

}  // namespace xla

#endif  // XLA_PYTHON_PROFILER_BUILD_PROFILE_DATA_NB_MODULE_H_
