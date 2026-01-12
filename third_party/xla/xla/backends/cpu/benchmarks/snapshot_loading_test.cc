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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

TEST(SnapshotLoadingTest, LoadHloSnapshot) {
  constexpr absl::string_view hlo = R"(
    HloModule add

    ENTRY e {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnUnverifiedModule(hlo, HloModuleConfig()));

  auto literal_x = LiteralUtil::CreateR0<float>(5.0f);
  auto literal_y = LiteralUtil::CreateR0<float>(2.0f);

  HloSnapshot snapshot;
  *snapshot.mutable_hlo()->mutable_hlo_module() = module->ToProto();

  snapshot.mutable_arguments()->Add(literal_x.ToProto());
  snapshot.mutable_arguments()->Add(literal_y.ToProto());

  std::string tmp_snapshot_path = tsl::testing::TmpDir() + "/hlo_snapshot.pb";

  ASSERT_OK(
      tsl::WriteBinaryProto(tsl::Env::Default(), tmp_snapshot_path, snapshot));

  TF_ASSERT_OK_AND_ASSIGN(
      auto hlo_module_and_inputs,
      LoadHloModuleAndMaybeIterationLiterals(tmp_snapshot_path));

  EXPECT_EQ(hlo_module_and_inputs.second->arguments_size(), 2);

  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_literal_x,
      Literal::CreateFromProto(hlo_module_and_inputs.second->arguments(0)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_literal_y,
      Literal::CreateFromProto(hlo_module_and_inputs.second->arguments(1)));

  EXPECT_EQ(loaded_literal_x, literal_x);
  EXPECT_EQ(loaded_literal_y, literal_y);
}

TEST(SnapshotLoadingTest, LoadHloUnoptimizedSnapshot) {
  constexpr absl::string_view hlo = R"(
    HloModule add

    ENTRY e {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnUnverifiedModule(hlo, HloModuleConfig()));

  auto literal_x = LiteralUtil::CreateR0<float>(5.0f);
  auto literal_y = LiteralUtil::CreateR0<float>(2.0f);

  HloUnoptimizedSnapshot snapshot;
  *snapshot.mutable_hlo_module() = module->ToProto();

  auto* partition = snapshot.add_partitions();

  partition->mutable_arguments()->Add(literal_x.ToProto());
  partition->mutable_arguments()->Add(literal_y.ToProto());

  std::string tmp_snapshot_path =
      tsl::testing::TmpDir() + "/hlo_unoptimized_snapshot.pb";

  ASSERT_OK(
      tsl::WriteBinaryProto(tsl::Env::Default(), tmp_snapshot_path, snapshot));

  TF_ASSERT_OK_AND_ASSIGN(
      auto hlo_module_and_inputs,
      LoadHloModuleAndMaybeIterationLiterals(tmp_snapshot_path));

  EXPECT_EQ(hlo_module_and_inputs.second->arguments_size(), 2);

  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_literal_x,
      Literal::CreateFromProto(hlo_module_and_inputs.second->arguments(0)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_literal_y,
      Literal::CreateFromProto(hlo_module_and_inputs.second->arguments(1)));

  EXPECT_EQ(loaded_literal_x, literal_x);
  EXPECT_EQ(loaded_literal_y, literal_y);
}

}  // namespace
}  // namespace xla::cpu
