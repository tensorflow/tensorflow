/* Copyright 2026 The OpenXLA Authors.

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
#include <string>
#include <vector>

#include "xla/tests/xla_test_backend_predicates.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tools/multihost_hlo_runner/create_client.h"
#include "xla/tools/multihost_hlo_runner/functional_hlo_runner.h"
#include "xla/tools/multihost_hlo_runner/hlo_input_output_format.h"
#include "xla/tools/multihost_hlo_runner/hlo_runner_profiler.h"
#include "xla/tools/multihost_hlo_runner/simple_multipass_tracer_for_test.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::UnorderedElementsAreArray;

std::string GetHloPath(absl::string_view file_name) {
  return tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                           "multihost_hlo_runner", "data", file_name);
}

absl::StatusOr<std::unique_ptr<xla::PjRtClient>> GetPjRtClient() {
  if (test::DeviceTypeIs(test::kCpu)) {
    LOG(INFO) << "Running on CPU";
    return CreateHostClient();
  }
  LOG(INFO) << "Running on GPU";
  return CreateGpuClient({});
}

class MultiPassFunctionalHloRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    xla::profiler::RegisterSimpleMultiPassTracerForTest();
  }

  void TearDown() override {
    tsl::profiler::ClearRegisteredMultiPassProfilersForTest();
  }
};

TEST_F(MultiPassFunctionalHloRunnerTest, SimpleMultiPassTracerTest) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                       GetPjRtClient());

  // 1. Run non-multipass profiling session.
  std::string profile_dump_path_non_multipass =
      tsl::io::JoinPath(testing::TempDir(), "xspace_non_multipass.pb");
  FunctionalHloRunner::RunningOptions running_options_non_multipass;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HLORunnerProfiler> profiler_non_multipass,
      HLORunnerProfiler::Create(profile_dump_path_non_multipass,
                                /*keep_xspace=*/true,
                                /*enable_multipass_profiling=*/false));
  running_options_non_multipass.profiler = profiler_non_multipass.get();
  running_options_non_multipass.multipass_profiling = false;

  // We must use a valid HLO file to run.
  EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client,
      /* preproc_options= */ {},
      /* raw_compile_options = */ {}, running_options_non_multipass,
      GetHloPath("single_device.hlo"), InputFormat::kText));

  const tensorflow::profiler::XSpace* xspace_non_multipass =
      profiler_non_multipass->GetXSpace();
  ASSERT_NE(xspace_non_multipass, nullptr);

  // 2. Run multipass profiling session.
  std::string profile_dump_path_multipass =
      tsl::io::JoinPath(testing::TempDir(), "xspace_multipass.pb");
  FunctionalHloRunner::RunningOptions running_options_multipass;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HLORunnerProfiler> profiler_multipass,
                       HLORunnerProfiler::Create(
                           profile_dump_path_multipass, /*keep_xspace=*/true,
                           /*enable_multipass_profiling=*/true));
  running_options_multipass.profiler = profiler_multipass.get();
  running_options_multipass.multipass_profiling = true;

  EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client,
      /* preproc_options= */ {},
      /* raw_compile_options = */ {}, running_options_multipass,
      GetHloPath("single_device.hlo"), InputFormat::kText));

  const tensorflow::profiler::XSpace* xspace_multipass =
      profiler_multipass->GetXSpace();
  ASSERT_NE(xspace_multipass, nullptr);

  // 3. Verify that the planes collected in both modes are identical.
  std::vector<std::string> plane_names_non_multipass;
  for (const auto& plane : xspace_non_multipass->planes()) {
    LOG(INFO) << "Non multipass profiling results plane: " << plane.name();
    plane_names_non_multipass.push_back(plane.name());
  }

  std::vector<std::string> plane_names_multipass;
  for (const auto& plane : xspace_multipass->planes()) {
    LOG(INFO) << "Multipass profiling results plane: " << plane.name();
    plane_names_multipass.push_back(plane.name());
  }

  EXPECT_THAT(plane_names_multipass, Not(IsEmpty()));
  EXPECT_THAT(plane_names_multipass,
              UnorderedElementsAreArray(plane_names_non_multipass));
}

TEST_F(MultiPassFunctionalHloRunnerTest,
       CreateFailsWhenMultiNodeWithoutKvStore) {
  std::string profile_dump_path =
      tsl::io::JoinPath(testing::TempDir(), "xspace_multinode.pb");
  EXPECT_THAT(HLORunnerProfiler::Create(profile_dump_path, /*keep_xspace=*/true,
                                        /*enable_multipass_profiling=*/true,
                                        /*kv_store=*/nullptr, /*task_id=*/0,
                                        /*num_nodes=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MultiPassFunctionalHloRunnerTest, CreateFailsWhenTaskIdOutOfBounds) {
  auto kv_store = std::make_shared<InMemoryKeyValueStore>();
  std::string profile_dump_path =
      tsl::io::JoinPath(testing::TempDir(), "xspace_invalid_task_id.pb");
  EXPECT_THAT(HLORunnerProfiler::Create(profile_dump_path, /*keep_xspace=*/true,
                                        /*enable_multipass_profiling=*/true,
                                        kv_store, /*task_id=*/-1,
                                        /*num_nodes=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(HLORunnerProfiler::Create(profile_dump_path, /*keep_xspace=*/true,
                                        /*enable_multipass_profiling=*/true,
                                        kv_store, /*task_id=*/2,
                                        /*num_nodes=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MultiPassFunctionalHloRunnerTest,
       UploadSessionWithoutCreateSessionDoesNotCrash) {
  std::string profile_dump_path =
      tsl::io::JoinPath(testing::TempDir(), "xspace_empty_upload.pb");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HLORunnerProfiler> profiler,
      HLORunnerProfiler::Create(profile_dump_path, /*keep_xspace=*/true,
                                /*enable_multipass_profiling=*/true));
  profiler->UploadSession();
  EXPECT_NE(profiler->GetXSpace(), nullptr);
}

TEST_F(MultiPassFunctionalHloRunnerTest, AllReduceOrSingleRankAndBoundsCheck) {
  InMemoryKeyValueStore kv_store;
  EXPECT_THAT(AllReduceOr(/*kv_store=*/nullptr, /*rank_id=*/0, /*num_ranks=*/2,
                          /*local_value=*/true, "key", absl::Seconds(1)),
              IsOkAndHolds(true));
  EXPECT_THAT(AllReduceOr(&kv_store, /*rank_id=*/0, /*num_ranks=*/1,
                          /*local_value=*/false, "key", absl::Seconds(1)),
              IsOkAndHolds(false));
  EXPECT_THAT(AllReduceOr(&kv_store, /*rank_id=*/-1, /*num_ranks=*/2,
                          /*local_value=*/false, "key", absl::Seconds(1)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(AllReduceOr(&kv_store, /*rank_id=*/2, /*num_ranks=*/2,
                          /*local_value=*/false, "key", absl::Seconds(1)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MultiPassFunctionalHloRunnerTest, AllReduceOrMultiRankConsensus) {
  InMemoryKeyValueStore kv_store;

  auto run_two_ranks = [&](bool rank0_val, bool rank1_val,
                           absl::string_view key) {
    absl::StatusOr<bool> res0;
    absl::StatusOr<bool> res1;
    std::unique_ptr<tsl::Thread> t1(
        tsl::Env::Default()->StartThread(tsl::ThreadOptions(), "rank1", [&]() {
          res1 = AllReduceOr(&kv_store, /*rank_id=*/1, /*num_ranks=*/2,
                             rank1_val, key, absl::Seconds(5));
        }));
    res0 = AllReduceOr(&kv_store, /*rank_id=*/0, /*num_ranks=*/2, rank0_val,
                       key, absl::Seconds(5));
    t1.reset();
    EXPECT_EQ(res0, res1);
    return res0;
  };

  EXPECT_THAT(run_two_ranks(false, false, "step_0"), IsOkAndHolds(false));
  EXPECT_THAT(run_two_ranks(false, true, "step_1"), IsOkAndHolds(true));
  EXPECT_THAT(run_two_ranks(true, false, "step_2"), IsOkAndHolds(true));
  EXPECT_THAT(run_two_ranks(true, true, "step_3"), IsOkAndHolds(true));
}

TEST_F(MultiPassFunctionalHloRunnerTest,
       AllReduceOrTimeoutPropagatesErrorTokenToPeer) {
  InMemoryKeyValueStore kv_store;
  absl::StatusOr<bool> res0;
  absl::StatusOr<bool> res2;

  // 3 ranks, where rank 1 never participates. Rank 0 times out waiting for
  // rank 1 and publishes kAllReduceErrorToken so rank 2 unblocks promptly
  // instead of waiting for its 10-second timeout.
  std::unique_ptr<tsl::Thread> t2(
      tsl::Env::Default()->StartThread(tsl::ThreadOptions(), "rank2", [&]() {
        res2 = AllReduceOr(&kv_store, /*rank_id=*/2, /*num_ranks=*/3,
                           /*local_value=*/false, "timeout_key",
                           absl::Seconds(10));
      }));
  res0 =
      AllReduceOr(&kv_store, /*rank_id=*/0, /*num_ranks=*/3,
                  /*local_value=*/true, "timeout_key", absl::Milliseconds(50));
  t2.reset();

  EXPECT_FALSE(res0.ok());
  EXPECT_THAT(res2, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(MultiPassFunctionalHloRunnerTest,
       CollectiveMultipassHloProfilerSynchronizesPasses) {
  auto kv_store = std::make_shared<InMemoryKeyValueStore>();
  std::string dump_path_0 =
      tsl::io::JoinPath(testing::TempDir(), "xspace_collective_rank0.pb");

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HLORunnerProfiler> profiler_0,
      HLORunnerProfiler::Create(dump_path_0, /*keep_xspace=*/true,
                                /*enable_multipass_profiling=*/true, kv_store,
                                /*task_id=*/0, /*num_nodes=*/2,
                                /*collective_id=*/7));

  profiler_0->CreateSession();

  // SimpleMultiPassTracer on Rank 0 requires 2 local passes (passes 0 and 1).
  // Simulate Peer Rank 1 requiring 3 passes (passes 0, 1, and 2) and false on
  // pass 3. CollectiveMultipassHloProfiler on Rank 0 must synchronize and run
  // all 3 passes, then stop on pass 3.
  std::unique_ptr<tsl::Thread> peer_rank(
      tsl::Env::Default()->StartThread(tsl::ThreadOptions(), "rank1", [&]() {
        for (int pass_idx = 0; pass_idx < 4; ++pass_idx) {
          bool peer_need_more = (pass_idx < 3);
          std::string keystr =
              absl::StrCat("hlo_runner_need_more_pass_7_1_", pass_idx);
          EXPECT_THAT(
              AllReduceOr(kv_store.get(), /*rank_id=*/1, /*num_ranks=*/2,
                          peer_need_more, keystr, absl::Seconds(5)),
              IsOkAndHolds(pass_idx < 3));
        }
      }));

  int passes_rank_0 = 0;
  while (true) {
    ASSERT_OK_AND_ASSIGN(bool need_more_pass, profiler_0->NeedMorePasses());
    if (!need_more_pass) {
      break;
    }
    profiler_0->StartPass();
    profiler_0->PushRange("pass");
    profiler_0->PopRange();
    profiler_0->StopPass();
    passes_rank_0++;
  }
  peer_rank.reset();

  EXPECT_EQ(passes_rank_0, 3);

  profiler_0->UploadSession();
  EXPECT_NE(profiler_0->GetXSpace(), nullptr);
}

TEST_F(MultiPassFunctionalHloRunnerTest,
       CollectiveMultipassHloProfilerPropagatesErrorFromAllReduceOr) {
  auto kv_store = std::make_shared<InMemoryKeyValueStore>();
  std::string dump_path_1 =
      tsl::io::JoinPath(testing::TempDir(), "xspace_collective_rank1.pb");

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HLORunnerProfiler> profiler_1,
      HLORunnerProfiler::Create(dump_path_1, /*keep_xspace=*/true,
                                /*enable_multipass_profiling=*/true, kv_store,
                                /*task_id=*/1, /*num_nodes=*/2,
                                /*collective_id=*/9));

  profiler_1->CreateSession();

  // Simulate rank 0 publishing an error token during AllReduceOr.
  ASSERT_OK(kv_store->Set("hlo_runner_need_more_pass_9_1_0_decision", "ERROR"));

  EXPECT_THAT(profiler_1->NeedMorePasses(),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
