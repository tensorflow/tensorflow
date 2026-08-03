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
#include "xla/backends/gpu/runtime/annotation.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;

TEST(AnnotationTest, KernelAnnotationPopulatesUniqueId) {
  auto module = std::make_unique<HloModule>("test_module", HloModuleConfig());
  HloComputation::Builder builder("test_computation");
  // Add a dummy instruction first (gets local_id 0)
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));
  // Add the instruction to test (gets local_id 1)
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloComputation* computation = module->AddEntryComputation(builder.Build());
  HloInstruction* constant = computation->root_instruction();

  ModuleAnnotation module_annotation(*module);
  KernelAnnotation kernel_annotation(module_annotation, *constant);

  absl::string_view title(kernel_annotation);
  EXPECT_THAT(constant->unique_id(), 1);
  EXPECT_THAT(title, HasSubstr(absl::StrCat("unique_hlo_op_id=",
                                            constant->unique_id())));
}

}  // namespace
}  // namespace xla::gpu
