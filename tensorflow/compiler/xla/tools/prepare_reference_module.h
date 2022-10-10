/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_PREPARE_REFERENCE_MODULE_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_PREPARE_REFERENCE_MODULE_H_

#include <functional>
#include <memory>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_runner_interface.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/lib/status.h"
#include "tensorflow/compiler/xla/stream_executor/platform.h"

namespace xla {

// A helper function that takes a HloModule, derives a HloModuleConfig from it
// which disables fast-math und sets the DebugOptions from flags, then runs the
// deoptimization pipeline (or calls 'module_modifier_hook' if provided). This
// is meant to produce a reference module that is comparable to our custom test
// platforms.
StatusOr<std::unique_ptr<HloModule>> PrepareReferenceModule(
    const HloModule& test_module, HloRunnerInterface* test_runner,
    const std::function<void(HloModuleConfig*)>& config_modifier_hook = {},
    const std::function<Status(const HloModule&, HloRunnerInterface*,
                               HloModule*)>& module_modifier_hook = {});

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_PREPARE_REFERENCE_MODULE_H_
