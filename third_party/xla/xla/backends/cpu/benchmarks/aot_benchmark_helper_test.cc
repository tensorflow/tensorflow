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

#include "xla/backends/cpu/benchmarks/aot_benchmark_helper.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/test_target_triple_helper.h"

namespace xla::cpu {
namespace {

TEST(AotBenchmarkHelperTest, GetAotCompilationOptions) {
  std::shared_ptr<xla::AotCompilationOptions> options =
      xla::cpu::GetAotCompilationOptions();

  xla::cpu::CpuAotCompilationOptions* cpu_options =
      dynamic_cast<xla::cpu::CpuAotCompilationOptions*>(options.get());

  EXPECT_NE(cpu_options, nullptr);

  EXPECT_EQ(cpu_options->entry_point_name(),
            xla::cpu::internal::kEntryPointNameDefault);
  EXPECT_EQ(cpu_options->relocation_model(),
            xla::cpu::CpuAotCompilationOptions::RelocationModel::BigPic);
  EXPECT_EQ(cpu_options->triple(), kTargetTripleForHost);
  EXPECT_EQ(cpu_options->cpu_name(), kTargetCpuForHost);
  EXPECT_EQ(cpu_options->features(), "");
}

TEST(AotBenchmarkHelperTest, GetAotCompilationOptionsCustomValues) {
  const std::string entry_point_name = "entry_point_name";
  const CpuAotCompilationOptions::RelocationModel relocation_model =
      CpuAotCompilationOptions::RelocationModel::BigPie;
  const std::string features = "features";

  std::shared_ptr<xla::AotCompilationOptions> options =
      xla::cpu::GetAotCompilationOptions(entry_point_name, relocation_model,
                                         features);

  xla::cpu::CpuAotCompilationOptions* cpu_options =
      dynamic_cast<xla::cpu::CpuAotCompilationOptions*>(options.get());

  EXPECT_NE(cpu_options, nullptr);

  EXPECT_EQ(cpu_options->entry_point_name(), entry_point_name);
  EXPECT_EQ(cpu_options->relocation_model(), relocation_model);
  EXPECT_EQ(cpu_options->triple(), kTargetTripleForHost);
  EXPECT_EQ(cpu_options->cpu_name(), kTargetCpuForHost);
  EXPECT_EQ(cpu_options->features(), features);
}

}  // namespace

}  // namespace xla::cpu
