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

#include "xla/python/compile_only_ifrt/client.h"

#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

[[maybe_unused]] char CompileOnlyMemory::ID = 0;
[[maybe_unused]] char CompileOnlyDevice::ID = 0;
[[maybe_unused]] char CompileOnlyIfRtClient::ID = 0;

}  // namespace xla
