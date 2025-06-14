/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/c/pjrt_c_api_phase_compile_plugin_internal.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"

namespace pjrt {
namespace phase_compile_cpu_plugin {

namespace {

PJRT_Error* PJRT_Cpu_Topology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_TopologyDescription_Create_Args",
      PJRT_TopologyDescription_Create_Args_STRUCT_SIZE, args->struct_size));

  std::vector<std::string> machine_attributes;
  machine_attributes.push_back("abc");
  auto cpu_devices = std::vector<xla::CpuTopology::CpuDevice>();
  auto topology_description = std::make_unique<xla::CpuTopologyDescription>(
      xla::CpuId(), xla::CpuName(), "<unknown>", cpu_devices,
      machine_attributes);
  args->topology =
      pjrt::CreateWrapperDeviceTopology(std::move(topology_description));
  return nullptr;
}

PJRT_Error* PJRT_PhaseCompile_Get_Compiler(
    PJRT_PhaseCompile_Get_Compiler_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "No implementation provided for PJRT_PhaseCompile_Get_Compiler")};
}

}  // namespace

// Helper to register the phase compile CPU compiler and create a PJRT API
// with the phase compile extension.
const PJRT_Api* GetPhaseCompilePjrtApi() {
  static PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::CreatePhaseCompileExtension(nullptr,
                                        PJRT_PhaseCompile_Get_Compiler);

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      /*create_fn=*/nullptr, /*execute_context_create_fn=*/nullptr,
      PJRT_Cpu_Topology_Create, pjrt::PJRT_Plugin_Initialize_NoOp,
      &phase_compile_extension.base, pjrt::PJRT_Plugin_Attributes_Xla);

  return &pjrt_api;
}

}  // namespace phase_compile_cpu_plugin
}  // namespace pjrt
