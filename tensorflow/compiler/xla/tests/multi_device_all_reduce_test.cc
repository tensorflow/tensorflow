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

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/core/status_test_util.h"

// Tests cross-GPU all-reduce operatons.
//
// This test requires multiple GPUs.  For instructions on running this within
// Google, see go/multi-gpu-unit-test.

namespace xla {
namespace {

using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class MultiDeviceAllReduceTest : public HloTestBase {
 protected:
  std::unique_ptr<HloModule> MakeCrsModule(int64 num_elems,
                                           const HloModuleConfig& config) {
    const char* kTemplate = R"(
      HloModule test

      add {
        x = f32[] parameter(0)
        y = f32[] parameter(1)
        add = f32[] add(x, y)
      }

      ENTRY test_computation {
        p = f32[NUM_ELEMS] parameter(0)
        ROOT crs = f32[NUM_ELEMS] all-reduce(p), to_apply=add
      }
    )";
    return ParseHloString(
               absl::StrReplaceAll(kTemplate,
                                   {{"NUM_ELEMS", absl::StrCat(num_elems)}}),
               config)
        .ValueOrDie();
  }
};

// Returns the non-empty subsets of {0, 1, ..., n}.  For example,
// PowerSetOfIota(3) = {{0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}}.
std::vector<std::vector<int64>> PowerSetOfIota(int64 n) {
  std::vector<std::vector<int64>> power_set;
  for (int64 i = 1; i < (1 << n); ++i) {
    power_set.emplace_back();
    for (int64 j = 0; j < n; ++j) {
      if (i & (1 << j)) {
        power_set.back().push_back(j);
      }
    }
  }
  return power_set;
}

// Makes a DeviceAssignment assigning replica-id i to devices[i].
DeviceAssignment MakeDeviceAssn(std::vector<int64> devices) {
  DeviceAssignment assn(/*replica_count=*/devices.size(),
                        /*computation_count=*/1);
  for (int64 i = 0; i < devices.size(); ++i) {
    assn(i, 0) = devices[i];
  }
  return assn;
}

// Shorter alias for this function.
absl::flat_hash_set<int> OpenNcclChannels() {
  return gpu::NcclAllReduceThunk::DevicesWithOpenNcclChannels();
}

XLA_TEST_F(MultiDeviceAllReduceTest, TwoReplicasOneOperand) {
  auto config = GetModuleConfigForTest();
  config.set_replica_count(2);
  auto module = MakeCrsModule(/*num_elems=*/3, config);
  auto literal = LiteralUtil::CreateR1<float>({1, 2, 3});
  auto expected = LiteralUtil::CreateR1<float>({2, 4, 6});
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(std::move(module), {&literal}, 2,
                                            /*use_threads=*/true));
  EXPECT_EQ(expected, results[0]);
  EXPECT_EQ(expected, results[1]);
}

// Tries all-to-all operations across all 2^kNumDevices - 1 combinations of
// devices in sequence.
XLA_TEST_F(MultiDeviceAllReduceTest, AllCombinations) {
  const int64 kNumDevices = 4;
  const int64 kNumElems = 1024;

  for (std::vector<int64> devices : PowerSetOfIota(kNumDevices)) {
    SCOPED_TRACE(absl::StrFormat("Running on devices {%s}",
                                 absl::StrJoin(devices, ", ")));

    DeviceAssignment device_assn = MakeDeviceAssn(devices);

    auto config = GetModuleConfigForTest();
    config.set_replica_count(devices.size());
    config.set_static_device_assignment(device_assn);

    auto module = MakeCrsModule(kNumElems, config);

    std::vector<float> input_vec(kNumElems);
    absl::c_iota(input_vec, 0);
    auto input_literal = LiteralUtil::CreateR1<float>(input_vec);

    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<Literal> results,
        ExecuteReplicated(std::move(module), {&input_literal},
                          /*num_replicas=*/devices.size(), &device_assn,
                          /*run_hlo_passes=*/true, /*use_threads=*/true));
  }
}

// Check that the NCCL data structures in our all-reduce implementation are
// cached as we expect.
XLA_TEST_F(MultiDeviceAllReduceTest, NcclChannelCaching) {
  const int64 kNumElems = 1024;

  std::vector<float> input_vec(kNumElems);
  absl::c_iota(input_vec, 0);
  auto input_literal = LiteralUtil::CreateR1<float>(input_vec);

  // Initially no NCCL channels should be open.
  EXPECT_THAT(OpenNcclChannels(), IsEmpty());

  // Create three Executables, touching devices {0,1}, {1,2}, and {0,1,2}.
  struct ExecutableInfo {
    std::unique_ptr<Executable> executable;
    DeviceAssignment device_assn;
    HloRunner::ReplicatedExecuteOptions opts;
  };
  std::vector<ExecutableInfo> executables;
  for (const auto& devices :
       std::vector<std::vector<int64>>{{0, 1}, {1, 2}, {0, 1, 2}}) {
    executables.emplace_back();
    auto& e = executables.back();

    e.device_assn = MakeDeviceAssn(devices);

    auto config = GetModuleConfigForTest();
    config.set_replica_count(devices.size());
    config.set_static_device_assignment(e.device_assn);
    auto module = MakeCrsModule(kNumElems, config);
    e.executable =
        test_runner_
            .CreateExecutable(std::move(module), /*run_hlo_passes=*/true)
            .ValueOrDie();

    e.opts.num_replicas = devices.size();
    e.opts.use_threads = true;
    e.opts.arguments.push_back(&input_literal);
  }

  auto run_executable = [&](int64 i) {
    auto& e = executables[i];
    TF_ASSERT_OK(
        test_runner_
            .ExecuteReplicated(e.executable.get(), e.opts, &e.device_assn)
            .status());
  };

  // Compiling executables above shouldn't cause us to open any channels.
  EXPECT_THAT(OpenNcclChannels(), IsEmpty());

  // Run the executables and check that channels are opened as we expect.
  run_executable(0);
  EXPECT_THAT(OpenNcclChannels(), UnorderedElementsAre(0, 1));

  run_executable(2);
  EXPECT_THAT(OpenNcclChannels(), UnorderedElementsAre(0, 1, 2));

  run_executable(1);
  EXPECT_THAT(OpenNcclChannels(), UnorderedElementsAre(0, 1, 2));

  // Tear down the executables and check that channels are closed as we expect.
  // Note that after we tear down an executable *all* the nccl channels may go
  // away, so we rerun all of the executables that haven't been torn down.
  executables[2].executable.reset();
  run_executable(0);
  run_executable(1);
  EXPECT_THAT(OpenNcclChannels(), UnorderedElementsAre(0, 1, 2));

  executables[0].executable.reset();
  run_executable(1);
  EXPECT_THAT(OpenNcclChannels(), UnorderedElementsAre(1, 2));

  executables[1].executable.reset();
  EXPECT_THAT(OpenNcclChannels(), IsEmpty());
}

}  // namespace
}  // namespace xla
