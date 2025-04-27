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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "xla/backends/cpu/codegen/emitters/cpu_fusion_emitter_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/llvm_compiler.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

using CpuCompilerInternalsTest = HloTestBase;

std::optional<int64_t> GetMetadataInt(absl::Nullable<llvm::Metadata*> value) {
  if (value == nullptr) {
    return std::nullopt;
  }
  auto* cam = llvm::dyn_cast<llvm::ConstantAsMetadata>(value);
  if (cam == nullptr) {
    return std::nullopt;
  }
  auto* c = llvm::dyn_cast<llvm::ConstantInt>(cam->getValue());
  if (c == nullptr) {
    return std::nullopt;
  }
  return c->getSExtValue();
}

std::optional<std::string> GetMetadataString(
    absl::Nullable<llvm::Metadata*> value) {
  if (value == nullptr) {
    return std::nullopt;
  }
  auto* md_string = llvm::dyn_cast<llvm::MDString>(value);
  if (md_string == nullptr) {
    return std::nullopt;
  }
  return md_string->getString().str();
}

std::optional<int64_t> GetXlaDylibIndex(const llvm::Module& llvm_module) {
  llvm::Metadata* md = llvm_module.getModuleFlag("xla_dylib_index");
  return GetMetadataInt(md);
}

std::optional<std::string> GetXlaBackendExtraOptions(
    const llvm::Module& llvm_module) {
  llvm::Metadata* md = llvm_module.getModuleFlag("xla_backend_extra_options");
  return GetMetadataString(md);
}

static constexpr absl::string_view kAddScatterHlo = R"(
  add {
    %lhs = f32[] parameter(0)
    %rhs = f32[] parameter(1)
    ROOT %add.2 = f32[] add(%lhs, %rhs)
  }

  ENTRY main {
    %a = f32[50,64,8] parameter(0)
    %b = f32[50,64,8] parameter(1)
    %operand = f32[50,64,8] add(%a, %b)
    %indices = s32[500,1]{1,0} parameter(2)
    %updates = f32[500,1,64,8] parameter(3)
    ROOT %scatter = f32[50,64,8] scatter(%operand, %indices, %updates),
      update_window_dims={1,2,3},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      to_apply=add
  }
)";

TEST_F(CpuCompilerInternalsTest, DylibWithThunks) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kAddScatterHlo));
  DebugOptions& debug_options =
      hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_use_thunk_runtime(true);
  debug_options.set_xla_cpu_use_fusion_emitters(false);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(hlo_module)));

  int64_t max_seen = -1;
  auto pre_opt_hook = [&](const llvm::Module& llvm_module) {
    std::optional<int64_t> dylib_index = GetXlaDylibIndex(llvm_module);
    if (dylib_index) {
      max_seen = std::max(max_seen, *dylib_index);
    }
  };

  LLVMCompiler* compiler = static_cast<LLVMCompiler*>(backend().compiler());
  compiler->SetPreOptimizationHook(pre_opt_hook);
  ASSERT_TRUE(compiler
                  ->RunBackend(std::move(optimized_module),
                               backend().default_stream_executor(),
                               /*device_allocator=*/nullptr)
                  .ok());
  compiler->RemovePreOptimizationHook();

  EXPECT_GT(max_seen, 0) << "max dylib_index(" << max_seen << ") too low; "
                         << "expected to use more dylibs.";
}

TEST_F(CpuCompilerInternalsTest, JustOneDylibWithThunks) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kAddScatterHlo));
  DebugOptions& debug_options =
      hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_use_thunk_runtime(true);
  debug_options.set_xla_cpu_use_fusion_emitters(false);
  debug_options.set_xla_cpu_parallel_codegen_split_count(1);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(hlo_module)));

  int64_t max_seen = -1;
  auto pre_opt_hook = [&](const llvm::Module& llvm_module) {
    std::optional<int64_t> dylib_index = GetXlaDylibIndex(llvm_module);
    if (dylib_index) {
      max_seen = std::max(max_seen, *dylib_index);
    }
  };

  LLVMCompiler* compiler = static_cast<LLVMCompiler*>(backend().compiler());
  compiler->SetPreOptimizationHook(pre_opt_hook);
  ASSERT_TRUE(compiler
                  ->RunBackend(std::move(optimized_module),
                               backend().default_stream_executor(),
                               /*device_allocator=*/nullptr)
                  .ok());
  compiler->RemovePreOptimizationHook();

  EXPECT_EQ(max_seen, 0) << "max dylib_index(" << max_seen
                         << ") != 0, but only "
                         << "one dylib is allowed.";
}

}  // namespace
}  // namespace cpu
}  // namespace xla
