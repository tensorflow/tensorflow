/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tools/matmul_perf_table_gen.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/cost_model/hlo_op_profile.pb.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class MatmulPerfTableGenTest : public HloTestBase {
  void SetUp() override {
    if (!backend()
             .default_stream_executor()
             ->GetDeviceDescription()
             .gpu_compute_capability()
             .IsCuda()) {
      GTEST_SKIP() << "Not built with --config=cuda";
    }
  }
};

TEST_F(MatmulPerfTableGenTest, DryRunsSpecifiedSweepSpace) {
  MatmulPerfTableGen::Config cfg;
  cfg.b_spec.start = 1;
  cfg.b_spec.stop = 1;
  cfg.b_spec.step = 1;
  cfg.k_spec.start = 1;
  cfg.k_spec.stop = 1;
  cfg.k_spec.step = 1;
  cfg.m_spec.start = 1;
  cfg.m_spec.stop = 1;
  cfg.m_spec.step = 1;
  cfg.n_spec.start = 2;
  cfg.n_spec.stop = 8;
  cfg.n_spec.step = 2;
  cfg.dry_run = true;
  cfg.dtypes.emplace_back(
      MatmulPerfTableGen::DataTypeSpec{"bf16", "bf16", "bf16"});

  MatmulPerfTableGen gen(cfg);
  DeviceHloInstructionProfiles profiles = gen.ComputeTable();

  EXPECT_EQ(profiles.entries_size(), 1);
  EXPECT_EQ(profiles.entries().begin()->second.entries_size(), 4);
}

TEST_F(MatmulPerfTableGenTest, DryRunsFactorSweepSpace) {
  MatmulPerfTableGen::Config cfg;
  cfg.b_spec.start = 1;
  cfg.b_spec.stop = 1;
  cfg.b_spec.step = 1;
  cfg.k_spec.start = 1;
  cfg.k_spec.stop = 1;
  cfg.k_spec.step = 1;
  cfg.m_spec.start = 1;
  cfg.m_spec.stop = 1;
  cfg.m_spec.step = 1;
  cfg.n_spec.start = 2;
  cfg.n_spec.stop = 8;
  cfg.n_spec.factor = 2;
  cfg.dry_run = true;
  cfg.dtypes.emplace_back(
      MatmulPerfTableGen::DataTypeSpec{"bf16", "bf16", "bf16"});

  MatmulPerfTableGen gen(cfg);
  DeviceHloInstructionProfiles profiles = gen.ComputeTable();

  EXPECT_EQ(profiles.entries_size(), 1);
  EXPECT_EQ(profiles.entries().begin()->second.entries_size(), 3);
}

TEST_F(MatmulPerfTableGenTest, SweepSpaceSavesOperands) {
  MatmulPerfTableGen::Config cfg;
  cfg.b_spec.start = 1;
  cfg.b_spec.stop = 1;
  cfg.b_spec.step = 1;
  cfg.k_spec.start = 1;
  cfg.k_spec.stop = 1;
  cfg.k_spec.step = 1;
  cfg.m_spec.start = 1;
  cfg.m_spec.stop = 1;
  cfg.m_spec.step = 1;
  cfg.n_spec.start = 1;
  cfg.n_spec.stop = 1;
  cfg.n_spec.step = 1;
  cfg.dry_run = true;
  cfg.dtypes.emplace_back(
      MatmulPerfTableGen::DataTypeSpec{"bf16", "bf16", "bf16"});

  MatmulPerfTableGen gen(cfg);
  DeviceHloInstructionProfiles profiles = gen.ComputeTable();

  EXPECT_EQ(profiles.entries_size(), 1);
  EXPECT_EQ(profiles.entries().begin()->second.entries_size(), 1);
  EXPECT_EQ(profiles.entries().begin()->second.entries(0).operands_size(), 2);
}

TEST_F(MatmulPerfTableGenTest, SweepSpaceSavesFlops) {
  MatmulPerfTableGen::Config cfg;
  cfg.b_spec.start = 2;
  cfg.b_spec.stop = 2;
  cfg.b_spec.step = 1;
  cfg.k_spec.start = 8;
  cfg.k_spec.stop = 8;
  cfg.k_spec.step = 1;
  cfg.m_spec.start = 3;
  cfg.m_spec.stop = 3;
  cfg.m_spec.step = 1;
  cfg.n_spec.start = 7;
  cfg.n_spec.stop = 7;
  cfg.n_spec.step = 1;
  cfg.dry_run = true;
  cfg.dtypes.emplace_back(
      MatmulPerfTableGen::DataTypeSpec{"bf16", "bf16", "bf16"});

  MatmulPerfTableGen gen(cfg);
  DeviceHloInstructionProfiles profiles = gen.ComputeTable();

  EXPECT_EQ(profiles.entries_size(), 1);
  EXPECT_EQ(profiles.entries().begin()->second.entries_size(), 1);
  // b = 2, m = 8, n = 3, k = 7 => # flops = 2 * 8 * 3 * 7 * 2 = 672.
  // with a dry run on, t = 42ns, gflops/s = 672 / 42 = 16 => flops/s = 16 *
  // 1e9.
  EXPECT_EQ(profiles.entries().begin()->second.entries(0).flops(), 16 * 1e9);
}

TEST_F(MatmulPerfTableGenTest, CompactsTable) {
  MatmulPerfTableGen::Config cfg;
  cfg.b_spec.start = 2;
  cfg.b_spec.stop = 2;
  cfg.b_spec.step = 1;
  cfg.k_spec.start = 8;
  cfg.k_spec.stop = 8;
  cfg.k_spec.step = 1;
  cfg.m_spec.start = 3;
  cfg.m_spec.stop = 3;
  cfg.m_spec.step = 1;
  cfg.n_spec.start = 7;
  cfg.n_spec.stop = 7;
  cfg.n_spec.step = 1;
  cfg.dry_run = true;
  cfg.dtypes.emplace_back(
      MatmulPerfTableGen::DataTypeSpec{"bf16", "bf16", "bf16"});
  cfg.dtypes.emplace_back(
      MatmulPerfTableGen::DataTypeSpec{"bf16", "f16", "f32"});

  MatmulPerfTableGen gen(cfg);
  TF_ASSERT_OK_AND_ASSIGN(GemmPerfTable compact_table,
                          MatmulPerfTableGen::Compact(gen.ComputeTable()));

  EXPECT_EQ(compact_table.entries_size(), 1);
  EXPECT_EQ(compact_table.entries().begin()->second.entries_size(), 1);
  const GemmPerfTableEntry& entry =
      compact_table.entries().begin()->second.entries()[0];
  EXPECT_EQ(entry.b(), 2);
  EXPECT_EQ(entry.m(), 3);
  EXPECT_EQ(entry.k(), 8);
  EXPECT_EQ(entry.n(), 7);
  EXPECT_EQ(entry.flops().at("bf16xbf16->bf16"), 16 * 1e9);
  EXPECT_EQ(entry.flops().at("bf16xf16->f32"), 16 * 1e9);
}

TEST_F(MatmulPerfTableGenTest, CompactTableInDeterministicOrder) {
  MatmulPerfTableGen::Config cfg;
  cfg.b_spec.start = 1;
  cfg.b_spec.stop = 8;
  cfg.b_spec.step = 1;
  cfg.k_spec.start = 8;
  cfg.k_spec.stop = 8;
  cfg.k_spec.step = 1;
  cfg.m_spec.start = 3;
  cfg.m_spec.stop = 3;
  cfg.m_spec.step = 1;
  cfg.n_spec.start = 7;
  cfg.n_spec.stop = 7;
  cfg.n_spec.step = 1;
  cfg.dry_run = true;
  cfg.dtypes.emplace_back(
      MatmulPerfTableGen::DataTypeSpec{"bf16", "bf16", "bf16"});

  MatmulPerfTableGen gen(cfg);
  TF_ASSERT_OK_AND_ASSIGN(GemmPerfTable compact_table,
                          MatmulPerfTableGen::Compact(gen.ComputeTable()));

  EXPECT_EQ(compact_table.entries_size(), 1);
  EXPECT_EQ(compact_table.entries().begin()->second.entries_size(), 8);

  // Expect entries in increasing order of b.
  int64_t expect_b = 1;
  for (const GemmPerfTableEntry& entry :
       compact_table.entries().begin()->second.entries()) {
    EXPECT_EQ(entry.b(), expect_b++);
  }
}

TEST_F(MatmulPerfTableGenTest, MergeGemmTables) {
  const absl::string_view kGemmTableOld = R"pb(
    entries {
      key: "sm_90"
      value {
        entries {
          b: 1
          m: 1024
          n: 2048
          k: 256
          flops { key: "bf16xbf16->bf16" value: 123000 }
          flops { key: "f32xf32->f32" value: 456000 }
        }
      }
    }
  )pb";
  const absl::string_view kGemmTableNew = R"pb(
    entries {
      key: "sm_90"
      value {
        entries {
          b: 2
          m: 256
          n: 2048
          k: 2048
          flops { key: "bf16xbf16->bf16" value: 789000 }
          flops { key: "f32xf32->f32" value: 123000 }
        }
      }
    }
    entries {
      key: "sm_100"
      value {
        entries {
          b: 2
          m: 256
          n: 2048
          k: 2048
          flops { key: "bf16xbf16->bf16" value: 789 }
          flops { key: "f32xf32->f32" value: 123 }
        }
      }
    }
  )pb";
  const absl::string_view kGemmTableExpected = R"pb(
    entries {
      key: "sm_90"
      value {
        entries {
          b: 1
          m: 1024
          n: 2048
          k: 256
          flops { key: "bf16xbf16->bf16" value: 123000 }
          flops { key: "f32xf32->f32" value: 456000 }
        }
        entries {
          b: 2
          m: 256
          n: 2048
          k: 2048
          flops { key: "bf16xbf16->bf16" value: 789000 }
          flops { key: "f32xf32->f32" value: 123000 }
        }
      }
    }
    entries {
      key: "sm_100"
      value {
        entries {
          b: 2
          m: 256
          n: 2048
          k: 2048
          flops { key: "bf16xbf16->bf16" value: 789 }
          flops { key: "f32xf32->f32" value: 123 }
        }
      }
    }
  )pb";
  GemmPerfTable old_perf_table;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(kGemmTableOld,
                                                         &old_perf_table));
  GemmPerfTable new_perf_table;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(kGemmTableNew,
                                                         &new_perf_table));
  GemmPerfTable expected_merged_perf_table;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      kGemmTableExpected, &expected_merged_perf_table));
  GemmPerfTable actual_merged_perf_table =
      MatmulPerfTableGen::Merge({old_perf_table, new_perf_table});
  EXPECT_THAT(expected_merged_perf_table,
              tsl::proto_testing::IgnoringRepeatedFieldOrdering(
                  tsl::proto_testing::EqualsProto(actual_merged_perf_table)));
}

}  // namespace
}  // namespace xla::gpu
