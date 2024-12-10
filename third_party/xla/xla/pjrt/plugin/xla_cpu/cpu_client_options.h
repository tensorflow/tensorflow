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

#ifndef XLA_PJRT_PLUGIN_XLA_CPU_CPU_CLIENT_OPTIONS_H_
#define XLA_PJRT_PLUGIN_XLA_CPU_CPU_CLIENT_OPTIONS_H_

#include <functional>
#include <memory>
#include <optional>

#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/hlo_module_config.h"

namespace xla {

// Options for creating an XLA:CPU PjRtClient.
struct CpuClientOptions {
  // Used to control whether asynchronous computation dispatch is available for
  // this client. Only applies to non-parallel computations, because collectives
  // may exist when there are multiple cpu devices and we need to do async
  // dispatch in that case. If it is set to be `false`, we will always run
  // computations inline.
  bool asynchronous = true;

  // Number of CPU devices. If not provided, the value of
  // --xla_force_host_platform_device_count is used.
  std::optional<int> cpu_device_count = std::nullopt;

  int max_inflight_computations_per_device = 32;

  // My process ID.
  int process_id = 0;

  // Distributed collectives implementation. Optional. If not provided, an
  // in-process collectives implementation will be used.
  std::shared_ptr<cpu::CollectivesInterface> collectives;

  // If defined this function will be called on the HloModuleConfig before
  // compilation, and allows users to set custom flags.
  std::function<void(HloModuleConfig&)> customize_hlo_module_config;
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XLA_CPU_CPU_CLIENT_OPTIONS_H_
