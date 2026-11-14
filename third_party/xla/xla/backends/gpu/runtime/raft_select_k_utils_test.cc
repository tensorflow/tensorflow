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

#include "xla/backends/gpu/runtime/raft_select_k_utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::HasSubstr;
using CudaCC = ::stream_executor::CudaComputeCapability;

int64_t WithExpectedSafetyMargin(int64_t base_bytes) {
  const int64_t margin = std::max<int64_t>(4096, base_bytes / 16);
  return ((base_bytes + margin + 255) / 256) * 256;
}

TEST(RaftSelectKUtilsTest, GetRaftSelectKMaxSmCountCoversCrossSkuAndMig) {
  // Pascal / Volta / Ampere / Ada full-die ceilings.
  EXPECT_EQ(
      GetRaftSelectKMaxSmCount(CudaCC::Pascal(), /*reported_sm_count=*/56), 60);
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC::Volta(), /*reported_sm_count=*/80),
            84);
  EXPECT_EQ(
      GetRaftSelectKMaxSmCount(CudaCC::Ampere(), /*reported_sm_count=*/108),
      144);
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC{8, 9}, /*reported_sm_count=*/142),
            144);

  // Hopper: H100 MIG (16), H100 PCIe (114), and H100 SXM / H200 (132) all
  // normalize to full GH100 die (144).
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC::H100Accelerated(), 16), 144);
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC::Hopper(), 114), 144);
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC::H100Accelerated(), 132), 144);

  // Blackwell (SM 10.0 / 10.3): B200 MIG (18), B200 (148), GB200 (152),
  // GB300 (152), and B300 (158) all normalize to full dual-die Blackwell (160).
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC::B200Accelerated(), 18), 160);
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC::B200Accelerated(), 148), 160);
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC::B200Accelerated(), 152), 160);
  EXPECT_EQ(
      GetRaftSelectKMaxSmCount(
          CudaCC{10, 3, CudaCC::FeatureExtension::kAcceleratedFeatures}, 152),
      160);
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC{10, 3}, 158), 160);

  // Consumer/workstation Blackwell GB202 (SM 12.0, RTX PRO 6000 = 188).
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC{12, 0}, 188), 192);

  // Rubin (SM 10.7, Vera Rubin NVL72 = 212) and unknown future architectures.
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC::Rubin(), 212),
            kDefaultRaftSelectKMaxSmCount);
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC{13, 0}, 0),
            kDefaultRaftSelectKMaxSmCount);

  // Never shrinks a larger explicitly reported SM count.
  EXPECT_EQ(GetRaftSelectKMaxSmCount(CudaCC::B200Accelerated(), 300), 300);
}

TEST(RaftSelectKUtilsTest, AlgoNameAndAbslStringify) {
  EXPECT_EQ(RaftSelectKAlgoName(RaftSelectKAlgo::kAuto), "kAuto");
  EXPECT_EQ(RaftSelectKAlgoName(RaftSelectKAlgo::kRadix8bits), "kRadix8bits");
  EXPECT_EQ(RaftSelectKAlgoName(RaftSelectKAlgo::kRadix11bits), "kRadix11bits");
  EXPECT_EQ(RaftSelectKAlgoName(RaftSelectKAlgo::kRadix11bitsExtraPass),
            "kRadix11bitsExtraPass");
  EXPECT_EQ(RaftSelectKAlgoName(RaftSelectKAlgo::kWarpAuto), "kWarpAuto");
  EXPECT_EQ(RaftSelectKAlgoName(RaftSelectKAlgo::kWarpImmediate),
            "kWarpImmediate");
  EXPECT_EQ(RaftSelectKAlgoName(RaftSelectKAlgo::kWarpFiltered),
            "kWarpFiltered");
  EXPECT_EQ(RaftSelectKAlgoName(RaftSelectKAlgo::kWarpDistributed),
            "kWarpDistributed");
  EXPECT_EQ(RaftSelectKAlgoName(RaftSelectKAlgo::kWarpDistributedShm),
            "kWarpDistributedShm");
  EXPECT_EQ(absl::StrCat(RaftSelectKAlgo::kRadix11bits), "kRadix11bits");
}

TEST(RaftSelectKUtilsTest, ChooseSelectKAlgorithmFloatLargeKBranches) {
  // k > 163, rows > 803
  EXPECT_EQ(choose_select_k_algorithm_float(804, 1024, 257),
            RaftSelectKAlgo::kRadix8bits);
  EXPECT_EQ(choose_select_k_algorithm_float(804, 3121, 256),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_float(804, 3120, 164),
            RaftSelectKAlgo::kWarpFiltered);

  // k > 163, rows <= 803, cols > 19332
  EXPECT_EQ(choose_select_k_algorithm_float(1, 106382977, 200),
            RaftSelectKAlgo::kRadix11bitsExtraPass);
  EXPECT_EQ(choose_select_k_algorithm_float(803, 106382976, 257),
            RaftSelectKAlgo::kRadix11bits);
  EXPECT_EQ(choose_select_k_algorithm_float(385, 19333, 256),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_float(384, 110947, 256),
            RaftSelectKAlgo::kRadix11bits);
  EXPECT_EQ(choose_select_k_algorithm_float(384, 110946, 164),
            RaftSelectKAlgo::kWarpFiltered);

  // k > 163, rows <= 803, cols <= 19332
  EXPECT_EQ(choose_select_k_algorithm_float(803, 19332, 257),
            RaftSelectKAlgo::kRadix8bits);
  EXPECT_EQ(choose_select_k_algorithm_float(803, 19332, 256),
            RaftSelectKAlgo::kWarpFiltered);
}

TEST(RaftSelectKUtilsTest, ChooseSelectKAlgorithmFloatSmallAndMediumKBranches) {
  // k == 1
  EXPECT_EQ(choose_select_k_algorithm_float(1024, 100000, 1),
            RaftSelectKAlgo::kWarpImmediate);

  // 1 < k <= 163, cols > 34520, k > 73
  EXPECT_EQ(choose_select_k_algorithm_float(133, 34521, 74),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_float(3, 12021889, 74),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_float(2, 12021889, 74),
            RaftSelectKAlgo::kRadix11bits);
  EXPECT_EQ(choose_select_k_algorithm_float(37, 259192, 74),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_float(36, 12021888, 74),
            RaftSelectKAlgo::kRadix11bits);
  EXPECT_EQ(choose_select_k_algorithm_float(49, 259191, 74),
            RaftSelectKAlgo::kRadix11bits);
  EXPECT_EQ(choose_select_k_algorithm_float(48, 34521, 74),
            RaftSelectKAlgo::kWarpFiltered);

  // 1 < k <= 73, cols > 34520
  EXPECT_EQ(choose_select_k_algorithm_float(1, 34521, 5),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_float(19, 34521, 4),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_float(18, 1652996, 4),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_float(18, 1652995, 4),
            RaftSelectKAlgo::kWarpImmediate);

  // 1 < k <= 163, cols <= 34520
  EXPECT_EQ(choose_select_k_algorithm_float(364, 1116, 2),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_float(1537, 1115, 2),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_float(1536, 1115, 2),
            RaftSelectKAlgo::kWarpImmediate);
  EXPECT_EQ(choose_select_k_algorithm_float(363, 1035, 86),
            RaftSelectKAlgo::kWarpFiltered);
  EXPECT_EQ(choose_select_k_algorithm_float(363, 1034, 86),
            RaftSelectKAlgo::kWarpImmediate);
  EXPECT_EQ(choose_select_k_algorithm_float(82, 12289, 5),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_float(82, 12288, 5),
            RaftSelectKAlgo::kWarpImmediate);
  EXPECT_EQ(choose_select_k_algorithm_float(82, 34520, 4),
            RaftSelectKAlgo::kWarpImmediate);
  EXPECT_EQ(choose_select_k_algorithm_float(81, 34520, 85),
            RaftSelectKAlgo::kWarpImmediate);
}

TEST(RaftSelectKUtilsTest, ChooseSelectKAlgorithmBFloat16LargeKBranches) {
  // k > 40, cols > 24576, rows > 768
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(769, 24577, 193),
            RaftSelectKAlgo::kRadix8bits);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(769, 196609, 81),
            RaftSelectKAlgo::kRadix11bits);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(1537, 196609, 80),
            RaftSelectKAlgo::kWarpDistributed);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(1536, 196609, 80),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(769, 196608, 192),
            RaftSelectKAlgo::kWarpDistributedShm);

  // k > 40, cols > 24576, rows <= 768, k > 192
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(768, 6291457, 193),
            RaftSelectKAlgo::kRadix11bitsExtraPass);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(768, 3145729, 769),
            RaftSelectKAlgo::kRadix11bits);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(768, 3145728, 769),
            RaftSelectKAlgo::kRadix11bitsExtraPass);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(768, 393217, 768),
            RaftSelectKAlgo::kRadix11bits);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(768, 393216, 193),
            RaftSelectKAlgo::kRadix11bitsExtraPass);

  // k > 40, cols > 24576, rows <= 768, k <= 192
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(768, 6291457, 81),
            RaftSelectKAlgo::kRadix11bitsExtraPass);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(7, 25165825, 80),
            RaftSelectKAlgo::kWarpDistributed);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(6, 25165825, 80),
            RaftSelectKAlgo::kRadix11bitsExtraPass);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(768, 25165824, 80),
            RaftSelectKAlgo::kRadix11bitsExtraPass);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(768, 98305, 41),
            RaftSelectKAlgo::kRadix11bits);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(49, 98304, 41),
            RaftSelectKAlgo::kRadix11bitsExtraPass);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(48, 98304, 41),
            RaftSelectKAlgo::kWarpFiltered);

  // k > 40, cols <= 24576
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(768, 24576, 257),
            RaftSelectKAlgo::kRadix8bits);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(769, 24576, 256),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(49, 6145, 256),
            RaftSelectKAlgo::kRadix8bits);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(49, 6144, 256),
            RaftSelectKAlgo::kWarpFiltered);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(48, 6145, 256),
            RaftSelectKAlgo::kWarpFiltered);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(48, 6144, 81),
            RaftSelectKAlgo::kWarpImmediate);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(48, 6144, 80),
            RaftSelectKAlgo::kWarpFiltered);
}

TEST(RaftSelectKUtilsTest,
     ChooseSelectKAlgorithmBFloat16SmallAndMediumKBranches) {
  // k == 1
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(1024, 100000, 1),
            RaftSelectKAlgo::kWarpImmediate);

  // 1 < k <= 40, cols > 24576
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(1, 98305, 2),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(49, 98304, 2),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(48, 98304, 25),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(48, 98304, 24),
            RaftSelectKAlgo::kWarpImmediate);

  // 1 < k <= 40, cols <= 24576
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(385, 24576, 40),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_bfloat16(384, 24576, 40),
            RaftSelectKAlgo::kWarpImmediate);
}

TEST(RaftSelectKUtilsTest, ChooseSelectKAlgorithmUInt64Branches) {
  // k == 1
  EXPECT_EQ(choose_select_k_algorithm_uint64(1024, 100000, 1),
            RaftSelectKAlgo::kWarpImmediate);

  // 1 < k <= 129
  EXPECT_EQ(choose_select_k_algorithm_uint64(1, 22090, 2),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_uint64(342, 22089, 129),
            RaftSelectKAlgo::kWarpDistributedShm);
  EXPECT_EQ(choose_select_k_algorithm_uint64(341, 22089, 129),
            RaftSelectKAlgo::kWarpImmediate);

  // k > 129, cols > 20215
  EXPECT_EQ(choose_select_k_algorithm_uint64(1014, 20216, 130),
            RaftSelectKAlgo::kRadix11bitsExtraPass);
  EXPECT_EQ(choose_select_k_algorithm_uint64(1013, 20216, 130),
            RaftSelectKAlgo::kRadix11bits);

  // k > 129, cols <= 20215
  EXPECT_EQ(choose_select_k_algorithm_uint64(1024, 20215, 257),
            RaftSelectKAlgo::kRadix8bits);
  EXPECT_EQ(choose_select_k_algorithm_uint64(1024, 20215, 256),
            RaftSelectKAlgo::kWarpFiltered);
}

TEST(RaftSelectKUtilsTest, EstimateScratchBytesRejectsInvalidInputs) {
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(0, 1024, 8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be positive")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(1, 0, 8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be positive")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(1, 1024, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be positive")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(1, 32, 64),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("requires k <= n")));
  const uint32_t kOverInt32Max =
      static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1u;
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(kOverInt32Max, 1024, 1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("batch exceeds int32_t max")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(1, kOverInt32Max, 1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("n exceeds int32_t max")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(
                  1, std::numeric_limits<int32_t>::max(), 1, /*sm_count=*/148),
              IsOkAndHolds(Gt(0)));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(1u << 22, 2048, 1024),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("batch * k exceeds uint32_t max")));

  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(0, 1024, 8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be positive")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(1, 32, 64),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("requires k <= n")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(kOverInt32Max, 1024, 1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("batch exceeds int32_t max")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(1, kOverInt32Max, 1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("n exceeds int32_t max")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(1u << 22, 2048, 1024),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("batch * k exceeds uint32_t max")));

  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(0, 1024, 8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be positive")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(1, 32, 64),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("requires k <= n")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(kOverInt32Max, 1024, 1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("batch exceeds int32_t max")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(1, kOverInt32Max, 1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("n exceeds int32_t max")));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(1u << 22, 2048, 1024),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("batch * k exceeds uint32_t max")));
}

TEST(RaftSelectKUtilsTest, EstimateScratchBytesWarpSortShapes) {
  // Tiny single-block WarpSort: n=64 <= 2*32=64 => num_of_block=1 =>
  // base = 256 + 256 = 512, + 4 KB safety floor = 4,608.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(1, 64, 1, /*sm_count=*/18),
              IsOkAndHolds(Eq(4608)));

  // Large-batch single-block WarpSort: batch=1024 >= 32*18=576 =>
  // num_of_block=1 => base = 2 * (1024 * 128 * 4) = 1,048,576,
  // + 6.25% proportional safety margin (65,536) = 1,114,112.
  EXPECT_THAT(
      EstimateRaftSelectKScratchBytesFloat(1024, 1024, 128, /*sm_count=*/18),
      IsOkAndHolds(Eq(1114112)));

  // WarpSort runs in 1 phase, so kMaxAcrossPhases and kSumAcrossPhases match.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(
                  16, 32768, 64, /*sm_count=*/148,
                  RaftScratchPhaseMode::kSumAcrossPhases),
              IsOkAndHolds(Eq(*EstimateRaftSelectKScratchBytesFloat(
                  16, 32768, 64, /*sm_count=*/148,
                  RaftScratchPhaseMode::kMaxAcrossPhases))));
}

TEST(RaftSelectKUtilsTest, EstimateScratchBytesRadixPhasesAndModes) {
  // Radix8bits with k == n: Phase 1 allocates 0 bytes, Phase 2 allocates
  // Align256((4+1)*4) + 2*Align256(4*512*4) + 2*Align256(4*512*4) + 256
  // = 256 + 16384 + 16384 + 256 = 33,280.
  EXPECT_THAT(
      EstimateRaftSelectKScratchBytesFloat(
          4, 512, 512, /*sm_count=*/18, RaftScratchPhaseMode::kMaxAcrossPhases),
      IsOkAndHolds(Eq(WithExpectedSafetyMargin(33280))));
  EXPECT_THAT(
      EstimateRaftSelectKScratchBytesFloat(
          4, 512, 512, /*sm_count=*/18, RaftScratchPhaseMode::kSumAcrossPhases),
      IsOkAndHolds(Eq(WithExpectedSafetyMargin(33280))));

  // Radix8bits one-block with n < 2048 => buf_len = 0 => Phase 1 = 0,
  // Phase 2 = 256 + 4 * (32 * 512 * 4) + 256 = 262,656.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(32, 1024, 512,
                                                   /*sm_count=*/18),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(262656))));

  // Radix8bits one-block (batch=256, n=16384, k=512, sm_count=18):
  //   buf_len = (16384 / 2048) * 64 = 512, row_buf = 8192
  //   chunk_one = min(256, 10*18*4) = 256 => Phase 1 = 2,097,152
  //   Phase 2 = 1280 + 4 * 524,288 + 256 = 2,098,688.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(
                  256, 16384, 512, /*sm_count=*/18,
                  RaftScratchPhaseMode::kMaxAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(2098688))));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(
                  256, 16384, 512, /*sm_count=*/18,
                  RaftScratchPhaseMode::kSumAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(2097152 + 2098688))));

  // Radix11bitsExtraPass multi-block (batch=1, n=110000000, k=256, sm=18):
  //   chunk_one = 1, chunk_multi = 1, buf_len = 3,437,440,
  //   Phase 1 = 768 + 8192 + 54,999,040 = 55,008,000, Phase 2 = 4,608.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(
                  1, 110000000, 256, /*sm_count=*/18,
                  RaftScratchPhaseMode::kMaxAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(55008000))));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(
                  1, 110000000, 256, /*sm_count=*/18,
                  RaftScratchPhaseMode::kSumAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(55008000 + 4608))));

  // Radix11bits multi-block where n > 2^24 rounds UP in IEEE-754 float inside
  // RAFT's `calc_buf_len<float>(len)` (RatioT = float):
  //   batch=2, n=16,779,263, k=74, sm_count=148:
  //   float(16779263u) = 16779264.0f => 16779264.0f / 32.0f = 524,352
  //   (64 elements = 1,024 bytes/row larger than integer (n / 2048) * 64 =
  //   524,288).
  //   chunk_multi = 2 => counters = 1,280, histograms = 16,384,
  //   bufs = 2 * 524,352 * 16 = 16,779,264 => Phase 1 = 16,796,928;
  //   Phase 2 = 256 + 768 + 768 + 1,792 = 3,584.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(
                  2, 16779263, 74, /*sm_count=*/148,
                  RaftScratchPhaseMode::kMaxAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(16796928))));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesFloat(
                  2, 16779263, 74, /*sm_count=*/148,
                  RaftScratchPhaseMode::kSumAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(16796928 + 3584))));
}

TEST(RaftSelectKUtilsTest, EstimateScratchBytesBFloat16WarpSortAndRadixShapes) {
  // Tiny single-block WarpSort: n=64 <= 2*32=64 => num_of_block=1 =>
  // base = Align256(2) + Align256(4) = 512, + 4 KB safety floor = 4,608.
  EXPECT_THAT(
      EstimateRaftSelectKScratchBytesBFloat16(1, 64, 1, /*sm_count=*/18),
      IsOkAndHolds(Eq(4608)));

  // Large-batch single-block WarpDistributedShm (batch=1024, n=16384, k=256,
  // sm_count=18): num_of_block=1 =>
  // tmp_val = 1024 * 256 * 2 = 524,288, tmp_idx = 1024 * 256 * 4 = 1,048,576,
  // base = 1,572,864.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(1024, 16384, 256,
                                                      /*sm_count=*/18),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(1572864))));

  // WarpDistributed (batch=2048, n=262144, k=64, sm_count=18):
  // num_of_block=1 => tmp_val = 262,144, tmp_idx = 524,288, base = 786,432.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(2048, 262144, 64,
                                                      /*sm_count=*/18),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(786432))));

  // Radix8bits one-block with n = 4096 < 6144 => buf_len = (4096 / 48) & ~127
  // = 85 & ~127 = 0 => Phase 1 = 0,
  // Phase 2 = 256 + 2 * 32,768 + 2 * 65,536 + 256 = 197,120.
  EXPECT_THAT(
      EstimateRaftSelectKScratchBytesBFloat16(32, 4096, 512, /*sm_count=*/18),
      IsOkAndHolds(Eq(WithExpectedSafetyMargin(197120))));

  // Radix8bits one-block (batch=512, n=16384, k=256, sm_count=18):
  //   buf_len = (16384 / 48) & ~127 = 256,
  //   row_buf_bytes = 256 * 2 * (2 + 4) = 3,072,
  //   chunk_one = min(512, 10*18*4) = 512 => Phase 1 = 1,572,864,
  //   Phase 2 = 2304 + 2 * 262,144 + 2 * 524,288 + 256 = 1,575,424.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(
                  512, 16384, 256, /*sm_count=*/18,
                  RaftScratchPhaseMode::kMaxAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(1575424))));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(
                  512, 16384, 256, /*sm_count=*/18,
                  RaftScratchPhaseMode::kSumAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(1572864 + 1575424))));

  // Radix11bits multi-block (batch=1, n=131072, k=64, sm_count=18):
  //   buf_len = (131072 / 48) & ~127 = 2,688, row_buf_bytes = 2688 * 12 =
  //   32,256, chunk_multi = 1 => counters = 768, histograms = 8,192,
  //   bufs = 32,256 => Phase 1 = 41,216;
  //   Phase 2 = 256 + 2 * 256 + 2 * 256 + 256 = 1,536.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(
                  1, 131072, 64, /*sm_count=*/18,
                  RaftScratchPhaseMode::kMaxAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(41216))));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(
                  1, 131072, 64, /*sm_count=*/18,
                  RaftScratchPhaseMode::kSumAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(41216 + 1536))));

  // Radix11bitsExtraPass multi-block where n > 2^24 rounds UP in IEEE-754 float
  // inside RAFT's `calc_buf_len<nv_bfloat16, uint32_t, float>(len)`:
  //   batch=1, n=16,779,263, k=128, sm_count=148:
  //   float(16779263u) = 16779264.0f => 16779264.0f / 48.0f = 349,568
  //   (128 elements = 1,536 bytes/row larger than integer (n / 48) & ~127 =
  //   349,440).
  //   chunk_multi = 1 => counters = 768, histograms = 8,192,
  //   bufs = 349,568 * 12 = 4,194,816 => Phase 1 = 4,203,776;
  //   Phase 2 = 256 + 2 * 256 + 2 * 512 + 256 = 2,048.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(
                  1, 16779263, 128, /*sm_count=*/148,
                  RaftScratchPhaseMode::kMaxAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(4203776))));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesBFloat16(
                  1, 16779263, 128, /*sm_count=*/148,
                  RaftScratchPhaseMode::kSumAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(4203776 + 2048))));
}

TEST(RaftSelectKUtilsTest, EstimateScratchBytesUInt64WarpSortAndRadixShapes) {
  // Tiny single-block WarpImmediate: n=64 <= 2*32=64 => num_of_block=1 =>
  // base = Align256(8) + Align256(4) = 512, + 4 KB safety floor = 4,608.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(1, 64, 1, /*sm_count=*/18),
              IsOkAndHolds(Eq(4608)));

  // Large-batch single-block WarpFiltered (batch=1024, n=16384, k=256,
  // sm_count=18): num_of_block=1 =>
  // tmp_val = 1024 * 256 * 8 = 2,097,152, tmp_idx = 1024 * 256 * 4 = 1,048,576,
  // base = 3,145,728.
  EXPECT_THAT(
      EstimateRaftSelectKScratchBytesUInt64(1024, 16384, 256, /*sm_count=*/18),
      IsOkAndHolds(Eq(WithExpectedSafetyMargin(3145728))));

  // Large-batch single-block WarpDistributedShm (batch=1024, n=262144, k=128,
  // sm_count=18): num_of_block=1 =>
  // tmp_val = 1,048,576, tmp_idx = 524,288, base = 1,572,864.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(1024, 262144, 128,
                                                    /*sm_count=*/18),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(1572864))));

  // Radix8bits one-block with n = 1024 < 1536 => buf_len = (1024 / 24) & ~63
  // = 42 & ~63 = 0 => Phase 1 = 0,
  // Phase 2 = 256 + 2 * 131,072 + 2 * 65,536 + 256 = 393,728.
  EXPECT_THAT(
      EstimateRaftSelectKScratchBytesUInt64(32, 1024, 512, /*sm_count=*/18),
      IsOkAndHolds(Eq(WithExpectedSafetyMargin(393728))));

  // Radix8bits one-block (batch=512, n=16384, k=512, sm_count=18):
  //   buf_len = (16384 / 24) & ~63 = 640,
  //   row_buf_bytes = 640 * 2 * (8 + 4) = 15,360,
  //   chunk_one = min(512, 10*18*4) = 512 => Phase 1 = 7,864,320,
  //   Phase 2 = 2304 + 2 * 2,097,152 + 2 * 1,048,576 + 256 = 6,294,016.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(
                  512, 16384, 512, /*sm_count=*/18,
                  RaftScratchPhaseMode::kMaxAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(7864320))));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(
                  512, 16384, 512, /*sm_count=*/18,
                  RaftScratchPhaseMode::kSumAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(7864320 + 6294016))));

  // Radix11bits multi-block (batch=16, n=262144, k=1024, sm_count=18):
  //   buf_len = (262144 / 24) & ~63 = 10,880, row_buf_bytes = 10880 * 24 =
  //   261,120, chunk_multi = 16 => counters = 10,240, histograms = 131,072,
  //   bufs = 4,177,920 => Phase 1 = 4,319,232;
  //   Phase 2 = 256 + 2 * 131,072 + 2 * 65,536 + 256 = 393,728.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(
                  16, 262144, 1024, /*sm_count=*/18,
                  RaftScratchPhaseMode::kMaxAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(4319232))));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(
                  16, 262144, 1024, /*sm_count=*/18,
                  RaftScratchPhaseMode::kSumAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(4319232 + 393728))));

  // Radix11bitsExtraPass (batch=1024, n=262144, k=256, sm_count=18):
  //   chunk_one = 720 => Phase 1 (one_block) = 720 * 261,120 = 188,006,400;
  //   chunk_multi = 45 => Phase 1 (multi_block) = 12,147,840;
  //   Phase 1 = 188,006,400, Phase 2 = 6,296,064.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(
                  1024, 262144, 256, /*sm_count=*/18,
                  RaftScratchPhaseMode::kMaxAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(188006400))));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(
                  1024, 262144, 256, /*sm_count=*/18,
                  RaftScratchPhaseMode::kSumAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(188006400 + 6296064))));

  // Radix11bits multi-block where n > 2^24 rounds UP in IEEE-754 float inside
  // RAFT's `calc_buf_len<uint64_t, uint32_t, float>(len)`:
  //   batch=1, n=16,777,727, k=130, sm_count=148:
  //   float(16777727u) = 16777728.0f => 16777728.0f / 24.0f = 699,072
  //   (64 elements = 1,536 bytes/row larger than integer (n / 24) & ~63 =
  //   699,008).
  //   chunk_multi = 1 => counters = 768, histograms = 8,192,
  //   bufs = 699,072 * 24 = 16,777,728 => Phase 1 = 16,786,688;
  //   Phase 2 = 256 + 2 * 1,280 + 2 * 768 + 256 = 4,608.
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(
                  1, 16777727, 130, /*sm_count=*/148,
                  RaftScratchPhaseMode::kMaxAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(16786688))));
  EXPECT_THAT(EstimateRaftSelectKScratchBytesUInt64(
                  1, 16777727, 130, /*sm_count=*/148,
                  RaftScratchPhaseMode::kSumAcrossPhases),
              IsOkAndHolds(Eq(WithExpectedSafetyMargin(16786688 + 4608))));
}

TEST(RaftSelectKUtilsTest, NonPositiveSmCountUsesConservativeDefault) {
  absl::StatusOr<int64_t> default_bytes =
      EstimateRaftSelectKScratchBytesFloat(16, 65536, 512);
  ASSERT_THAT(default_bytes, IsOkAndHolds(Gt(0)));
  EXPECT_THAT(
      EstimateRaftSelectKScratchBytesFloat(16, 65536, 512, /*sm_count=*/0),
      IsOkAndHolds(Eq(*default_bytes)));
  EXPECT_THAT(
      EstimateRaftSelectKScratchBytesFloat(16, 65536, 512, /*sm_count=*/-1),
      IsOkAndHolds(Eq(*default_bytes)));
}

TEST(RaftSelectKUtilsTest, SweepInvariantsAlignmentAndPhaseMonotonicity) {
  constexpr uint32_t kBatches[] = {1, 8, 64, 256, 1024};
  constexpr uint32_t kNs[] = {512, 1024, 16384, 65536, 262144};
  constexpr uint32_t kKs[] = {1, 4, 32, 128, 256, 512, 1024};

  for (uint32_t batch : kBatches) {
    for (uint32_t n : kNs) {
      for (uint32_t k : kKs) {
        if (k > n) {
          continue;
        }
        absl::StatusOr<int64_t> mig_sum_phase =
            EstimateRaftSelectKScratchBytesFloat(
                batch, n, k, /*sm_count=*/18,
                RaftScratchPhaseMode::kSumAcrossPhases);
        absl::StatusOr<int64_t> max_phase =
            EstimateRaftSelectKScratchBytesFloat(
                batch, n, k, /*sm_count=*/148,
                RaftScratchPhaseMode::kMaxAcrossPhases);
        absl::StatusOr<int64_t> sum_phase =
            EstimateRaftSelectKScratchBytesFloat(
                batch, n, k, /*sm_count=*/148,
                RaftScratchPhaseMode::kSumAcrossPhases);
        absl::StatusOr<int64_t> full_die_sum_phase =
            EstimateRaftSelectKScratchBytesFloat(
                batch, n, k, /*sm_count=*/160,
                RaftScratchPhaseMode::kSumAcrossPhases);
        ASSERT_THAT(mig_sum_phase, IsOkAndHolds(Ge(4608)));
        ASSERT_THAT(max_phase, IsOkAndHolds(Ge(4608)));
        ASSERT_THAT(sum_phase, IsOkAndHolds(Ge(*max_phase)));
        ASSERT_THAT(sum_phase, IsOkAndHolds(Ge(*mig_sum_phase)));
        ASSERT_THAT(full_die_sum_phase, IsOkAndHolds(Ge(*sum_phase)));
        EXPECT_EQ(*max_phase % 256, 0);
        EXPECT_EQ(*sum_phase % 256, 0);
        EXPECT_EQ(*full_die_sum_phase % 256, 0);

        absl::StatusOr<int64_t> bf16_mig_sum_phase =
            EstimateRaftSelectKScratchBytesBFloat16(
                batch, n, k, /*sm_count=*/18,
                RaftScratchPhaseMode::kSumAcrossPhases);
        absl::StatusOr<int64_t> bf16_max_phase =
            EstimateRaftSelectKScratchBytesBFloat16(
                batch, n, k, /*sm_count=*/148,
                RaftScratchPhaseMode::kMaxAcrossPhases);
        absl::StatusOr<int64_t> bf16_sum_phase =
            EstimateRaftSelectKScratchBytesBFloat16(
                batch, n, k, /*sm_count=*/148,
                RaftScratchPhaseMode::kSumAcrossPhases);
        absl::StatusOr<int64_t> bf16_full_die_sum_phase =
            EstimateRaftSelectKScratchBytesBFloat16(
                batch, n, k, /*sm_count=*/160,
                RaftScratchPhaseMode::kSumAcrossPhases);
        ASSERT_THAT(bf16_mig_sum_phase, IsOkAndHolds(Ge(4608)));
        ASSERT_THAT(bf16_max_phase, IsOkAndHolds(Ge(4608)));
        ASSERT_THAT(bf16_sum_phase, IsOkAndHolds(Ge(*bf16_max_phase)));
        ASSERT_THAT(bf16_sum_phase, IsOkAndHolds(Ge(*bf16_mig_sum_phase)));
        ASSERT_THAT(bf16_full_die_sum_phase, IsOkAndHolds(Ge(*bf16_sum_phase)));
        EXPECT_EQ(*bf16_max_phase % 256, 0);
        EXPECT_EQ(*bf16_sum_phase % 256, 0);
        EXPECT_EQ(*bf16_full_die_sum_phase % 256, 0);

        absl::StatusOr<int64_t> u64_mig_sum_phase =
            EstimateRaftSelectKScratchBytesUInt64(
                batch, n, k, /*sm_count=*/18,
                RaftScratchPhaseMode::kSumAcrossPhases);
        absl::StatusOr<int64_t> u64_max_phase =
            EstimateRaftSelectKScratchBytesUInt64(
                batch, n, k, /*sm_count=*/148,
                RaftScratchPhaseMode::kMaxAcrossPhases);
        absl::StatusOr<int64_t> u64_sum_phase =
            EstimateRaftSelectKScratchBytesUInt64(
                batch, n, k, /*sm_count=*/148,
                RaftScratchPhaseMode::kSumAcrossPhases);
        absl::StatusOr<int64_t> u64_full_die_sum_phase =
            EstimateRaftSelectKScratchBytesUInt64(
                batch, n, k, /*sm_count=*/160,
                RaftScratchPhaseMode::kSumAcrossPhases);
        ASSERT_THAT(u64_mig_sum_phase, IsOkAndHolds(Ge(4608)));
        ASSERT_THAT(u64_max_phase, IsOkAndHolds(Ge(4608)));
        ASSERT_THAT(u64_sum_phase, IsOkAndHolds(Ge(*u64_max_phase)));
        ASSERT_THAT(u64_sum_phase, IsOkAndHolds(Ge(*u64_mig_sum_phase)));
        ASSERT_THAT(u64_full_die_sum_phase, IsOkAndHolds(Ge(*u64_sum_phase)));
        EXPECT_EQ(*u64_max_phase % 256, 0);
        EXPECT_EQ(*u64_sum_phase % 256, 0);
        EXPECT_EQ(*u64_full_die_sum_phase % 256, 0);
      }
    }
  }
}

}  // namespace
}  // namespace xla::gpu
