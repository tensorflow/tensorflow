/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tsl/profiler/utils/traceme_global_flags.h"

#include <atomic>

namespace tsl {
namespace profiler {

#ifdef _WIN32
#define DECL_DLL_EXPORT __declspec(dllexport)
#else
#define DECL_DLL_EXPORT
#endif
// DLL imported variables cannot be initialized on Windows. This file is
// included only on DLL exports.
DECL_DLL_EXPORT std::atomic<bool> g_enable_source_location(true);

}  // namespace profiler
}  // namespace tsl
