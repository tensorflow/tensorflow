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

#include <array>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tests/aot_utils.h"
#include "xla/tests/pjrt_client_registry.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

using MultithreadedCompilation = HloHardwareIndependentTestBase;

std::unique_ptr<HloRunnerInterface> CreateRunnerOrDie() {
  CHECK(ShouldUsePjRt());
  absl::StatusOr<std::unique_ptr<PjRtClient>> client =
      GetGlobalPjRtClientTestFactory().Get()();
  CHECK_OK(client);
  return MakeHloRunnerPjRtAotAware(*std::move(client));
}

//  In this test, we are taking the same module and compiling it `num_threads`
//  times in parallel and are making it dump hlo files for layout assignment.
//  There was a race-conditon for a member variable and it is fixed. This test
//  is to verify this fix. The return status along with the contents of HLO
//  output is checked to make sure they are identical (since the input is the
//  same).
TEST_F(MultithreadedCompilation, EightModuleCompilation) {
  std::string hlo_text = R"(
  HloModule m1, entry_computation_layout={(f32[3,3,45,1]{3,2,1,0})->f32[3,3,45,1]{3,2,1,0}}
  ENTRY m1 {
    arg0.1 = f32[3,3,45,1]{3,2,1,0} parameter(0), parameter_replication={false}
    constant.4 = f32[] constant(0.0801233649)
    broadcast.5 = f32[3,3,45,1]{3,2,1,0} broadcast(constant.4), dimensions={}
    ROOT multiply.6 = f32[3,3,45,1]{3,2,1,0} multiply(arg0.1, broadcast.5)
})";
  constexpr int kNumThreads = 32;
  HloModuleConfig config = GetModuleConfigForTest(/*replica_count=*/1,
                                                  /*num_partitions=*/1);
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_dump_hlo_pass_re("layout-assignment");
  config.set_debug_options(debug_options);

  std::array<std::unique_ptr<HloModule>, kNumThreads> modules;
  for (int i = 0; i < kNumThreads; i++) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo_text, config));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_embed_ir_in_executable(true);
    modules[i] = std::move(module);
  }

  std::array<std::unique_ptr<OpaqueExecutable>, kNumThreads> executables;
  std::array<const HloModule*, kNumThreads> compiled_modules;
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "threads-",
                                        kNumThreads);
    for (int i = 0; i < kNumThreads; i++) {
      thread_pool.Schedule([i, &module = modules[i],
                            &executable = executables[i],
                            &compiled_module = compiled_modules[i]]() {
        std::unique_ptr<HloRunnerInterface> runner = CreateRunnerOrDie();
        absl::StatusOr<std::unique_ptr<OpaqueExecutable>> new_executable =
            runner->CreateExecutable(std::move(module),
                                     /*run_hlo_passes=*/true);
        EXPECT_OK(new_executable);
        if (!new_executable.status().ok()) {
          return;
        }
        absl::StatusOr<const HloModule*> new_compiled_module =
            runner->HloModuleFromWrapped(new_executable->get());
        EXPECT_OK(new_compiled_module);
        if (!new_compiled_module.status().ok()) {
          return;
        }
        executable = *std::move(new_executable);
        compiled_module = *new_compiled_module;
        VLOG(2) << "Adding executable obtained from thread: " << i;
      });
    }
  }

  ::tsl::protobuf::util::MessageDifferencer differencer;
  bool first_time = true;
  HloModuleProto first_hlo_proto;
  for (int i = 0; i < kNumThreads; i++) {
    HloModuleProto curr_hlo_proto = compiled_modules[i]->ToProto();
    if (first_time) {
      first_hlo_proto = std::move(curr_hlo_proto);
      first_time = false;
      const google::protobuf::FieldDescriptor* ignore_field =
          HloModuleProto::descriptor()->FindFieldByName("id");
      ASSERT_NE(ignore_field, nullptr);
      differencer.IgnoreField(ignore_field);
    } else {
      EXPECT_TRUE(differencer.Compare(first_hlo_proto, curr_hlo_proto));
    }
  }
}

}  // namespace xla
