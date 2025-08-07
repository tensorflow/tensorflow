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

#include <variant>

#include <gtest/gtest.h>
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class MatmulPerfTableGenTest : public HloTestBase {
  void SetUp() override {
    if (!IsCuda()) {
      GTEST_SKIP() << "Not built with --config=cuda";
    }
  }

 protected:
  bool IsCuda() {
    return std::holds_alternative<stream_executor::CudaComputeCapability>(
        backend()
            .default_stream_executor()
            ->GetDeviceDescription()
            .gpu_compute_capability());
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

}  // namespace
}  // namespace xla::gpu
