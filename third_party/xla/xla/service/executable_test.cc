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

#include "xla/service/executable.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace {

class TestExecutable : public Executable {
 public:
  explicit TestExecutable(std::shared_ptr<HloModule> module)
      : Executable{std::move(module)} {}

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    return absl::UnimplementedError("Not needed for this test.");
  }
};

class ExecutableTest : public HloTestBase {};

TEST_F(ExecutableTest, HloProtoGetterIsThreadCompatible) {
  // Executable::hlo_proto() is doing some lazy initialization of a
  // part of `hlo_proto_`. This test ensures that this is done in a
  // thread-compatible way.
  // Note that this test needs to run with --config=tsan to reliably
  // detect any potential data races.
  constexpr absl::string_view kHloModule = R"(
    HloModule module

    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  TestExecutable executable(module);

  auto proto = std::make_unique<HloProto>();
  executable.set_hlo_proto(std::move(proto));

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test",
                                 /*num_threads=*/2);
    for (int i = 0; i < 2; ++i) {
      pool.Schedule([&] { executable.hlo_proto()->SerializeAsString(); });
    }
  }
}

}  // namespace
}  // namespace xla
