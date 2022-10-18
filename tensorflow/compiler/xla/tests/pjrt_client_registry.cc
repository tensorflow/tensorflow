/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tests/pjrt_client_registry.h"

#include <functional>
#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/service/hlo_runner_pjrt.h"

namespace xla {

PjRtClientTestFactoryRegistry& GetGlobalPjRtClientTestFactory() {
  static auto* const factory = new PjRtClientTestFactoryRegistry;
  return *factory;
}

StatusOr<std::unique_ptr<HloRunnerInterface>> GetHloRunnerForTest(
    se::Platform* test_platform) {
  if (ShouldUsePjRt()) {
    TF_ASSIGN_OR_RETURN(auto client, GetGlobalPjRtClientTestFactory().Get()());
    return std::unique_ptr<HloRunnerInterface>(
        new HloRunnerPjRt(std::move(client)));
  } else {
    return std::unique_ptr<HloRunnerInterface>(new HloRunner(test_platform));
  }
}

void RegisterPjRtClientTestFactory(
    std::function<StatusOr<std::unique_ptr<PjRtClient>>()> factory) {
  GetGlobalPjRtClientTestFactory().Register(std::move(factory));
}

bool ShouldUsePjRt() {
  return GetGlobalPjRtClientTestFactory().HasRegisteredFactory();
}

}  // namespace xla
