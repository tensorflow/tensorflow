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

#include "xla/hlo/separate_compilation/hlo_module_linking.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/separate_compilation/hlo_linking_manifest.h"
#include "xla/hlo/separate_compilation/hlo_module_splitting.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::separate_compilation {
namespace {

// Function to normalize an HloModule by removing/replacing names.
void NormalizeHloModule(xla::HloModule& module) {
  // 1. Clear names from module (as clone keeps the original names)
  module.set_name("module");
  VLOG(6) << module.name() << " instr count: " << module.instruction_count();

  // 2. Get computations in post order
  std::vector<HloComputation*> computations_post_order =
      module.MakeComputationPostOrder(/*dfs_postorder=*/true);

  // 3. Initialize visited set and counter
  absl::flat_hash_set<const HloComputation*> visited;
  int computation_counter = 0;
  // 4. Number computations in post order
  for (HloComputation* computation : computations_post_order) {
    if (!visited.insert(computation).second) {
      continue;
    }
    VLOG(6) << computation->ToString()
            << " instr count: " << computation->instruction_count();
    computation->SetAndSanitizeName(
        absl::StrCat("computation_", computation_counter++));
    // 5. Number instructions inside the computation
    int instruction_counter = 0;
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      VLOG(6) << "  " << instruction->ToString();
      instruction->SetAndSanitizeName(
          absl::StrCat("instruction_", instruction_counter++));
    }
  }
}

bool AreHloModulesEquivalent(xla::HloModule& module1, xla::HloModule& module2) {
  auto module1_copy = module1.Clone("module1_copy");
  auto module2_copy = module2.Clone("module2_copy");

  NormalizeHloModule(*module1_copy);
  NormalizeHloModule(*module2_copy);

  xla::HloPrintOptions options;
  options.set_canonicalize_computations(true);

  auto module1_str = module1_copy->ToString(options);
  auto module2_str = module2_copy->ToString(options);

  VLOG(6) << module2_str;
  VLOG(6) << "Module 1 normalized string: " << module1_str << "\n"
          << "Module 2 normalized string: " << module2_str;
  return module1_str == module2_str;
}

class LinkingTest : public HloHardwareIndependentTestBase {};

TEST_F(LinkingTest, SingleCallLinking) {
  constexpr absl::string_view module_text = R"(
    HloModule simple_module

    %comp {
      %p = f32[] parameter(0)
      ROOT %result = f32[] negate(%p)
    }

    ENTRY %main {
      %p = f32[] parameter(0)
      ROOT %result = f32[] call(%p), to_apply=%comp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto original_module,
                          ParseAndReturnVerifiedModule(module_text));

  TF_ASSERT_OK_AND_ASSIGN(auto module_split_group,
                          CreateHloModuleSplitGroup(*original_module));
  // LOG the split group.
  for (const auto& split : module_split_group->module_splits) {
    LOG(INFO) << "Split: " << split->submodule->name();
    LOG(INFO) << "Split module: " << split->submodule->ToString();
    LOG(INFO) << "Computation map:";
    for (const auto& [original, cloned] : split->computation_map) {
      LOG(INFO) << "original: " << original->name()
                << " ==>> clones: " << cloned->name();
    }
    LOG(INFO) << "Call sites:";
    for (const auto* call_site : split->call_sites) {
      LOG(INFO) << "  " << call_site->name();
    }
    LOG(INFO) << " Stub links:";
    for (const auto& [stub, comp] : split->stub_map) {
      LOG(INFO) << " stub: " << stub->name()
                << " ==>> original: " << comp->name();
    }
  }
  // LOG the address book.
  for (const auto& [comp, split] : module_split_group->address_book) {
    LOG(INFO) << "Original: " << comp->name()
              << " Split: " << split->submodule->name();
  }
  // LOG the linking manifest stub links
  LOG(INFO) << "Linking manifest stub links:";
  for (const auto& [stub, comp] :
       module_split_group->linking_manifest.stub_links) {
    LOG(INFO) << "  " << stub->name() << " ==>> " << comp->name();
  }

  const HloLinkingManifest& linking_manifest =
      module_split_group->linking_manifest;
  auto* original_root = FindComputation(original_module.get(), "main");
  TF_ASSERT_OK_AND_ASSIGN(
      const HloComputation* split_group_root,
      module_split_group->GetClonedComputation(original_root));

  TF_ASSERT_OK_AND_ASSIGN(auto linked_module,
                          LinkComputation(linking_manifest, split_group_root));
  HloVerifier verifier(HloVerifierOpts{});
  ASSERT_OK(verifier.Run(linked_module.get()));

  EXPECT_TRUE(AreHloModulesEquivalent(*original_module, *linked_module));

  TF_ASSERT_OK_AND_ASSIGN(stream_executor::Platform * platform,
                          PlatformUtil::GetPlatform("cpu"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler,
                          Compiler::GetForPlatform(platform->id()));
  EXPECT_OK(compiler->RunHloPasses(original_module->Clone(),
                                   /*executor=*/nullptr,
                                   Compiler::CompileOptions{}));
  VLOG(6) << linked_module->ToString();
  ASSERT_OK(compiler->RunHloPasses(std::move(linked_module),
                                   /*executor=*/nullptr,
                                   Compiler::CompileOptions{}));
}

TEST_F(LinkingTest, ChainGraphLinking) {
  constexpr absl::string_view module_text = R"(
    HloModule nested_calls

    %comp3 {
      %p = f32[] parameter(0)
      ROOT %result = f32[] exponential(%p)
    }

    %comp2 {
      %p = f32[] parameter(0)
      %c = f32[] constant(3.0)
      %call_res = f32[] call(%p), to_apply=%comp3
      ROOT %result = f32[] add(%call_res, %c)
    }

    %comp1 {
      %p = f32[] parameter(0)
      %c = f32[] constant(2.0)
      %call_res = f32[] call(%p), to_apply=%comp2
      ROOT %result = f32[] multiply(%call_res, %c)
    }

    ENTRY %main {
      %p = f32[] parameter(0)
      %call_res = f32[] call(%p), to_apply=%comp1
      ROOT %result = f32[] add(%call_res, %p)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto original_module,
                          ParseAndReturnVerifiedModule(module_text));

  TF_ASSERT_OK_AND_ASSIGN(auto module_split_group,
                          CreateHloModuleSplitGroup(*original_module));

  const HloLinkingManifest& linking_manifest =
      module_split_group->linking_manifest;
  auto* original_root = FindComputation(original_module.get(), "main");
  TF_ASSERT_OK_AND_ASSIGN(
      const HloComputation* split_group_root,
      module_split_group->GetClonedComputation(original_root));

  TF_ASSERT_OK_AND_ASSIGN(auto linked_module,
                          LinkComputation(linking_manifest, split_group_root));
  HloVerifier verifier(HloVerifierOpts{});
  ASSERT_OK(verifier.Run(linked_module.get()));

  EXPECT_TRUE(AreHloModulesEquivalent(*original_module, *linked_module));
  TF_ASSERT_OK_AND_ASSIGN(stream_executor::Platform * platform,
                          PlatformUtil::GetPlatform("cpu"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler,
                          Compiler::GetForPlatform(platform->id()));
  VLOG(6) << linked_module->ToString();
  EXPECT_OK(compiler->RunHloPasses(std::move(linked_module),
                                   /*executor=*/nullptr,
                                   Compiler::CompileOptions{}));
  EXPECT_OK(compiler->RunHloPasses(original_module->Clone(),
                                   /*executor=*/nullptr,
                                   Compiler::CompileOptions{}));
}

TEST_F(LinkingTest, DiamondGraphLinking) {
  constexpr absl::string_view module_text = R"(
    HloModule shared_callee_module

    %fusion.2 (x: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      ROOT %add = f32[] negate(f32[] %x)
    }

    %y {
      %p = f32[] parameter(0)
      ROOT %result = f32[] exponential(%p)
    }

    %x {
      %p = f32[] parameter(0)
      %call_y = f32[] call(%p), to_apply=%y
      ROOT %result = f32[] cosine(%call_y)
    }

    %a {
      %p = f32[] parameter(0)
      %c = f32[] constant(5.0)
      %call_x = f32[] call(%p), to_apply=%x
      ROOT %result = f32[] add(%call_x, %c)
    }

    %b {
      %p = f32[] parameter(0)
      %f = f32[] fusion(%p), kind=kLoop, calls=%fusion.2
      %call_x = f32[] call(%p), to_apply=%x
      ROOT %result = f32[] subtract(%call_x, %f)
    }

    ENTRY %main {
      %p_entry = f32[] parameter(0)
      %call_a = f32[] call(%p_entry), to_apply=%a
      %call_b = f32[] call(%p_entry), to_apply=%b
      ROOT %result = f32[] add(%call_a, %call_b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto original_module,
                          ParseAndReturnVerifiedModule(module_text));

  TF_ASSERT_OK_AND_ASSIGN(auto module_split_group,
                          CreateHloModuleSplitGroup(*original_module));

  const HloLinkingManifest& linking_manifest =
      module_split_group->linking_manifest;
  auto* original_root = FindComputation(original_module.get(), "main");
  TF_ASSERT_OK_AND_ASSIGN(
      const HloComputation* split_group_root,
      module_split_group->GetClonedComputation(original_root));

  TF_ASSERT_OK_AND_ASSIGN(auto linked_module,
                          LinkComputation(linking_manifest, split_group_root));
  HloVerifier verifier(HloVerifierOpts{});
  ASSERT_OK(verifier.Run(linked_module.get()));

  EXPECT_TRUE(AreHloModulesEquivalent(*original_module, *linked_module));

  TF_ASSERT_OK_AND_ASSIGN(stream_executor::Platform * platform,
                          PlatformUtil::GetPlatform("cpu"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler,
                          Compiler::GetForPlatform(platform->id()));
  EXPECT_OK(compiler->RunHloPasses(original_module->Clone(),
                                   /*executor=*/nullptr,
                                   Compiler::CompileOptions{}));
  VLOG(6) << linked_module->ToString();
  EXPECT_OK(compiler->RunHloPasses(std::move(linked_module),
                                   /*executor=*/nullptr,
                                   Compiler::CompileOptions{}));
}

}  // namespace
}  // namespace xla::separate_compilation
