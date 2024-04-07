/* Copyright 2019 The OpenXLA Authors.

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
#include "absl/base/call_once.h"
#include "absl/memory/memory.h"
#include "xla/backends/profiler/gpu/cupti_error_manager.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "xla/backends/profiler/gpu/cupti_wrapper.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/stringpiece.h"

namespace xla {
namespace profiler {

bool IsCuptiUseStubInterface() {
  // TODO: b/149634979: Remove this after NVIDIA issue 4459155 resolved.
  static constexpr tsl::StringPiece cupti_use_stub_interface_env =
      "TF_GPU_CUPTI_USE_STUB_INTERFACE";
  static absl::once_flag once;  // NOLINT(clang-diagnostic-unreachable-code)
  static bool cupti_use_stub_interface = false;
  absl::call_once(once, [&] {
    tsl::ReadBoolFromEnvVar(cupti_use_stub_interface_env, false,
                            &cupti_use_stub_interface)
        .IgnoreError();
    if (cupti_use_stub_interface) {
      LOG(INFO) << cupti_use_stub_interface_env << " is set to true, "
                << "XLA Profiler is using stub CUPTI interface to work around "
                << "potential serious bug in CUPTI lib. Such control may be "
                << "removed/disabled in future if the known issue is resolved!";
    }
  });
  return cupti_use_stub_interface;
}

CuptiInterface* GetCuptiInterface() {
  static CuptiInterface* cupti_interface =
      IsCuptiUseStubInterface()
          ? new CuptiErrorManager(std::make_unique<CuptiWrapperStub>())
          : new CuptiErrorManager(std::make_unique<CuptiWrapper>());
  return cupti_interface;
}

}  // namespace profiler
}  // namespace xla
