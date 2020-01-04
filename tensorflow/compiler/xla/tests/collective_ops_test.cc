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

#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"

// Tests cross-GPU operations.
//
// This test requires at least four GPUs.  For instructions on running this
// within Google, see go/multi-gpu-unit-test.

namespace xla {
namespace {

using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class CollectiveOpsTest : public HloTestBase {
 protected:
  std::unique_ptr<HloModule> MakeCrsModule(
      const Shape& shape, std::vector<std::vector<int64>> replica_groups,
      const HloModuleConfig& config, std::string op = "add",
      std::string datatype = "f32") {
    std::string hlo_template = R"(
      HloModule test

      apply_op {
        x = DATATYPE[] parameter(0)
        y = DATATYPE[] parameter(1)
        ROOT apply_op = DATATYPE[] OP(x, y)
      }

      ENTRY test_computation {
        p = SHAPE parameter(0)
        p2 = SHAPE bitcast(p)
        crs = SHAPE all-reduce(p2), replica_groups=REPLICA_GROUPS, to_apply=apply_op
        copy = SHAPE copy(crs)
        ROOT out = SHAPE bitcast(copy)
      }
    )";
    std::vector<string> replica_group_strs;
    for (const auto& g : replica_groups) {
      replica_group_strs.push_back(
          absl::StrFormat("{%s}", absl::StrJoin(g, ",")));
    }
    std::string shape_str = shape.ToString(/*print_layout=*/false);
    if (shape_str == "f32[1]") {
      // Exercise the scalar codepath.
      hlo_template = absl::StrReplaceAll(
          hlo_template,
          {{"DATATYPE[SHAPE] bitcast(p)", "DATATYPE[] bitcast(p)"},
           {"DATATYPE[SHAPE] all-reduce", "DATATYPE[] all-reduce"},
           {"DATATYPE[SHAPE] copy", "DATATYPE[] copy"}});
    }
    std::string parameterized_hlo = absl::StrReplaceAll(
        hlo_template,
        {{"SHAPE", shape_str},
         {"REPLICA_GROUPS",
          absl::StrFormat("{%s}", absl::StrJoin(replica_group_strs, ", "))},
         {"OP", op},
         {"DATATYPE", datatype}});
    return ParseAndReturnVerifiedModule(parameterized_hlo, config).ValueOrDie();
  }

  template <typename LiteralType>
  void TestTwoReplicasOneOperand(std::string op, Literal input_value,
                                 Literal expected_value) {
    const int kNumReplicas = 2;
    std::string dtype = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<LiteralType>());
    auto config = GetModuleConfigForTest();
    config.set_replica_count(kNumReplicas);
    auto module = MakeCrsModule(
        /*shape_str=*/input_value.shape(),
        /*replica_groups=*/{}, config,
        /*op=*/op, /*datatype=*/dtype);
    TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                            ExecuteReplicated(std::move(module), {&input_value},
                                              /*num_replicas=*/kNumReplicas,
                                              /*use_threads=*/true));
    for (int replica_idx = 0; replica_idx < kNumReplicas; replica_idx++) {
      EXPECT_TRUE(LiteralTestUtil::NearOrEqual(
          expected_value, results[replica_idx], ErrorSpec{1e-5, 1e-5}));
    }
  }

  template <typename LiteralType>
  void TestAllOps() {
    auto cast = [&](int value) { return static_cast<LiteralType>(value); };
    auto to_literal = [&](absl::Span<const LiteralType> values) {
      return LiteralUtil::CreateR1<LiteralType>(values);
    };
    Literal input_value = to_literal({cast(1), cast(2), cast(3)});
    TestTwoReplicasOneOperand<LiteralType>(
        "add",
        /*input_value=*/input_value.Clone(),
        /*expected_value=*/to_literal({cast(2), cast(4), cast(6)}));
    TestTwoReplicasOneOperand<LiteralType>(
        "multiply",
        /*input_value=*/input_value.Clone(),
        /*expected_value=*/to_literal({cast(1), cast(4), cast(9)}));
    TestTwoReplicasOneOperand<LiteralType>(
        "maximum",
        /*input_value=*/input_value.Clone(),
        /*expected_value=*/to_literal({cast(1), cast(2), cast(3)}));
    TestTwoReplicasOneOperand<LiteralType>(
        "minimum",
        /*input_value=*/input_value.Clone(),
        /*expected_value=*/to_literal({cast(1), cast(2), cast(3)}));
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

template <typename T>
static Eigen::half ToHalf(T value) {
  return static_cast<Eigen::half>(value);
}

XLA_TEST_F(CollectiveOpsTest, AllReduce_sum_float32_2D) {
  TestTwoReplicasOneOperand<float>(
      "add",
      /*input_value=*/LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}}),
      /*expected_value=*/LiteralUtil::CreateR2<float>({{2, 4}, {6, 8}}));
}

XLA_TEST_F(CollectiveOpsTest, AllReduceSingleOutput_float32) {
  TestTwoReplicasOneOperand<float>(
      "add",
      /*input_value=*/LiteralUtil::CreateR1<float>({1}),
      /*expected_value=*/LiteralUtil::CreateR1<float>({2}));
}

XLA_TEST_F(CollectiveOpsTest, AllReduceTwoReplicasOneOperand_int8) {
  TestAllOps<int8>();
}

XLA_TEST_F(CollectiveOpsTest, AllReduceTwoReplicasOneOperand_uint8) {
  TestAllOps<uint8>();
}

XLA_TEST_F(CollectiveOpsTest, AllReduceTwoReplicasOneOperand_uint32) {
  TestAllOps<uint32>();
}

XLA_TEST_F(CollectiveOpsTest, AllReduceTwoReplicasOneOperand_int32) {
  TestAllOps<int32>();
}

XLA_TEST_F(CollectiveOpsTest, AllReduceTwoReplicasOneOperand_int64) {
  TestAllOps<int64>();
}

XLA_TEST_F(CollectiveOpsTest, AllReduceTwoReplicasOneOperand_uint64) {
  TestAllOps<uint64>();
}

XLA_TEST_F(CollectiveOpsTest, AllReduceTwoReplicasOneOperand_float32) {
  TestAllOps<float>();
}

XLA_TEST_F(CollectiveOpsTest, AllReduceTwoReplicasOneOperand_double) {
  TestAllOps<double>();
}

XLA_TEST_F(CollectiveOpsTest, AllReduceTwoReplicasOneOperand_half) {
  TestAllOps<Eigen::half>();
}

// Tries all-to-all operations across all 2^kNumDevices - 1 combinations of
// devices in sequence.
XLA_TEST_F(CollectiveOpsTest, AllReduce_AllCombinations) {
  const int64 kNumDevices = 4;
  const int64 kNumElems = 1024;

  for (std::vector<int64> devices : PowerSetOfIota(kNumDevices)) {
    SCOPED_TRACE(absl::StrFormat("Running on devices {%s}",
                                 absl::StrJoin(devices, ", ")));

    DeviceAssignment device_assn = MakeDeviceAssn(devices);

    auto config = GetModuleConfigForTest();
    config.set_replica_count(devices.size());
    config.set_static_device_assignment(device_assn);

    std::vector<float> input_vec(kNumElems);
    absl::c_iota(input_vec, 0);
    auto input_literal = LiteralUtil::CreateR1<float>(input_vec);

    auto module = MakeCrsModule(input_literal.shape(),
                                /*replica_groups=*/{}, config);

    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<Literal> results,
        ExecuteReplicated(std::move(module), {&input_literal},
                          /*num_replicas=*/devices.size(), &device_assn,
                          /*run_hlo_passes=*/true, /*use_threads=*/true));
  }
}

// Check that the NCCL data structures in our all-reduce implementation are
// cached as we expect.
XLA_TEST_F(CollectiveOpsTest, DISABLED_ON_CPU(AllReduce_NcclChannelCaching)) {
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
    auto module = MakeCrsModule(input_literal.shape(),
                                /*replica_groups=*/{}, config);
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

// Runs the same executable many times concurrently.  The all-reduces should not
// conflict with one another.
XLA_TEST_F(CollectiveOpsTest, AllReduce_ManyConcurrentAllReduces) {
  const int64 kNumElems = 1024;
  const int64 kNumThreads = 200;
  const int64 kRunsPerThread = 10;

  std::vector<float> input_vec(kNumElems);
  absl::c_iota(input_vec, 0);
  auto input_literal = LiteralUtil::CreateR1<float>(input_vec);

  auto config = GetModuleConfigForTest();
  config.set_replica_count(2);
  auto executable =
      test_runner_
          .CreateExecutable(MakeCrsModule(input_literal.shape(),
                                          /*replica_groups=*/{}, config),
                            /*run_hlo_passes=*/true)
          .ValueOrDie();
  std::vector<int64> devices = {0, 1};
  auto device_assn = MakeDeviceAssn(devices);

  HloRunner::ReplicatedExecuteOptions opts;
  opts.num_replicas = devices.size();
  opts.use_threads = true;
  opts.arguments.push_back(&input_literal);

  tensorflow::BlockingCounter done(kNumThreads * kRunsPerThread);
  tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(), TestName(),
                                      kNumThreads);
  for (int64 i = 0; i < kNumThreads * kRunsPerThread; ++i) {
    pool.Schedule([&] {
      TF_ASSERT_OK(
          test_runner_.ExecuteReplicated(executable.get(), opts, &device_assn)
              .status());
      done.DecrementCount();
    });
  }
  done.Wait();
}

// Runs an all-reduce with three partitions:
//  {0}, {1,2}, {3}
// meaning, the all-reduce is a nop for devices 0 and 3, and only devices 1 and
// 2 actually exchange data with each other.
XLA_TEST_F(CollectiveOpsTest, AllReduce_ThreeReplicaGroups) {
  // Test a prime number so it's not all powers of 2.
  const int64 kNumElems = 137;

  auto config = GetModuleConfigForTest();
  config.set_replica_count(4);
  std::vector<float> input_vec(kNumElems);
  absl::c_iota(input_vec, 0);
  auto input_literal = LiteralUtil::CreateR1<float>(input_vec);
  auto module = MakeCrsModule(
      /*shape_str=*/input_literal.shape(),
      /*replica_groups=*/{{0}, {1, 2}, {3}}, config);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), {&input_literal}, /*num_replicas=*/4,
                        /*use_threads=*/true));

  ASSERT_EQ(results.size(), 4);

  std::vector<float> input_vec_doubled;
  for (float n : input_vec) {
    input_vec_doubled.push_back(n * 2);
  }
  auto input_literal_doubled = LiteralUtil::CreateR1<float>(input_vec_doubled);

  EXPECT_TRUE(LiteralTestUtil::Equal(input_literal, results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(input_literal_doubled, results[1]));
  EXPECT_TRUE(LiteralTestUtil::Equal(input_literal_doubled, results[2]));
  EXPECT_TRUE(LiteralTestUtil::Equal(input_literal, results[3]));
}

XLA_TEST_F(CollectiveOpsTest, ReplicaId) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    ROOT out = u32[] copy(id)
  }
  )";
  const int64 kNumReplicas = 4;

  auto config = GetModuleConfigForTest();
  config.set_replica_count(kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(std::move(module), {}, kNumReplicas,
                                            /*use_threads=*/true));

  ASSERT_EQ(results.size(), kNumReplicas);
  for (uint32 i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR0(i), results[i]));
  }
}

XLA_TEST_F(CollectiveOpsTest, DISABLED_ON_CPU(CollectivePermute_Simple)) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    replica = u32[] replica-id()
    ten = u32[] constant(10)
    sum = u32[] add(replica, ten)
    p = u32[2] broadcast(sum), dimensions={}
    ROOT permute = u32[2] collective-permute(p), source_target_pairs={{1,0}, {0,1}, {2,2}}
  }
  )";
  const int64 kNumReplicas = 4;

  auto config = GetModuleConfigForTest();
  config.set_replica_count(kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(std::move(module), {}, kNumReplicas,
                                            /*use_threads=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<uint32>({11, 11}),
                                     results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<uint32>({10, 10}),
                                     results[1]));
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<uint32>({12, 12}),
                                     results[2]));
  // Nothing writes to replica 3, so it is memzero'ed.
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<uint32>({0, 0}),
                                     results[3]));
}

}  // namespace
}  // namespace xla
