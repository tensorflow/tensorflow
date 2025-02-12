/* Copyright 2022 The OpenXLA Authors.

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

#include <utility>

#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tests/pjrt_client_registry.h"

namespace xla {
namespace {

// Register a CPU PjRt client for tests.
const bool kUnused = (RegisterPjRtClientTestFactory([]() {
                        xla::CpuClientOptions options;
                        options.cpu_device_count = 4;
                        return xla::GetXlaPjrtCpuClient(std::move(options));
                      }),
                      true);

}  // namespace
}  // namespace xla
