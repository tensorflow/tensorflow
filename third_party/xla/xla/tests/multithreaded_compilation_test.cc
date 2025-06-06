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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace xla {

class MultithreadedCompilation : public HloTestBase {};

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
  constexpr int num_threads = 32;
  auto config = GetModuleConfigForTest(/*replica_count=*/2,
                                       /*num_partitions=*/1);
  auto debug_options = config.debug_options();
  debug_options.set_xla_dump_hlo_pass_re("layout-assignment");
  config.set_debug_options(debug_options);

  std::vector<std::unique_ptr<HloModule>> modules(num_threads);
  for (int i = 0; i < num_threads; i++) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    modules[i] = std::move(module);
  }

  absl::Mutex mu;
  std::vector<std::unique_ptr<OpaqueExecutable>> executables;
  auto do_compilation = [&](int iteration) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<OpaqueExecutable> executable,
                        CreateExecutable(std::move(modules[iteration]), true));
    absl::MutexLock lock(&mu);
    executables.push_back(std::move(executable));
    VLOG(2) << "Adding executable obtained from thread: " << iteration;
    return absl::OkStatus();
  };

  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "threads-",
                                        num_threads);
    for (int i = 0; i < num_threads; i++) {
      thread_pool.Schedule([&, i]() { TF_EXPECT_OK(do_compilation(i)); });
    }
  }

  ::tsl::protobuf::util::MessageDifferencer differencer;
  bool first_time = true;
  HloProto first_hlo_proto;
  for (const std::unique_ptr<OpaqueExecutable>& exec : executables) {
    TF_ASSERT_OK_AND_ASSIGN(
        const HloProto* const curr_hlo_proto,
        test_runner_as_hlo_runner().HloProtoFromWrapped(exec.get()));
    if (first_time) {
      first_hlo_proto = *curr_hlo_proto;
      first_time = false;
      auto ignore_field =
          curr_hlo_proto->hlo_module().GetDescriptor()->FindFieldByName("id");
      EXPECT_NE(ignore_field, nullptr);
      differencer.IgnoreField(ignore_field);
    } else {
      EXPECT_TRUE(differencer.Compare(first_hlo_proto, *curr_hlo_proto));
    }
  }
}
}  // namespace xla
