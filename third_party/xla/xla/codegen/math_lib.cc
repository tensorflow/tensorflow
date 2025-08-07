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

#include "xla/codegen/math_lib.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "xla/codegen/math/erf.h"
#include "xla/codegen/math/exp.h"
#include "xla/codegen/math/fptrunc.h"
#include "xla/codegen/math/intrinsic.h"
#include "xla/codegen/math/ldexp.h"
#include "xla/codegen/math/log1p.h"
#include "xla/codegen/math/rsqrt.h"
#include "xla/codegen/math/string_interner.h"
#include "xla/codegen/math/tanh.h"
#include "xla/codegen/math/vec_name_mangler.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen {
namespace {
using ::xla::codegen::intrinsics::Type;

// Allows unpacking a vector of types into individual arguments.
template <typename F, typename Container, size_t... Is>
decltype(auto) apply_vector(F&& f, const Container& v,
                            std::index_sequence<Is...>) {
  return f(v[Is]...);
}

template <size_t N, typename F, typename Container>
decltype(auto) apply_vector(F&& f, const Container& v) {
  return apply_vector(f, v, std::make_index_sequence<N>{});
}

std::vector<Type> ParseTypesFromFunctionName(absl::string_view function_name) {
  // The `to` in a typed function name is used to specify the return type, so
  // we ignore it when parsing the function name.
  static constexpr absl::string_view kIgnoredParts[] = {"to"};
  std::vector<Type> types;
  auto parts = absl::StrSplit(function_name, '.');
  size_t i = 0;
  for (absl::string_view part : parts) {
    // Skip the first two parts, which will be `xla.<func_name>`:
    if (i++ < 2 || std::find(std::begin(kIgnoredParts), std::end(kIgnoredParts),
                             part) != std::end(kIgnoredParts)) {
      continue;
    }
    types.push_back(Type::FromName(part));
  }
  return types;
}

}  // namespace

using intrinsics::Type;

template <typename Intrinsic>
class IntrinsicAdapter : public MathFunction {
 public:
  absl::string_view FunctionName() const override { return Intrinsic::kName; }
  std::vector<std::vector<Type>> SupportedVectorTypes(
      llvm::TargetMachine* target_machine) const override {
    if constexpr (std::is_invocable_v<decltype(Intrinsic::SupportedVectorTypes),
                                      llvm::TargetMachine*>) {
      return Intrinsic::SupportedVectorTypes(target_machine);
    } else {
      return Intrinsic::SupportedVectorTypes();
    }
  }

  llvm::Function* CreateDefinition(llvm::Module& module,
                                   llvm::TargetMachine* target_machine,
                                   absl::string_view name) const override {
    std::vector<Type> types = ParseTypesFromFunctionName(name);
    return apply_vector<Intrinsic::kNumArgs>(
               [&](auto... args) {
                 if constexpr (std::is_invocable_v<
                                   decltype(Intrinsic::CreateDefinition),
                                   llvm::Module*, llvm::TargetMachine*,
                                   decltype(args)...>) {
                   return Intrinsic::CreateDefinition(&module, target_machine,
                                                      args...);
                 } else {
                   return Intrinsic::CreateDefinition(&module, args...);
                 }
               },
               types)
        .value();
  }

  std::string GenerateVectorizedFunctionName(
      absl::Span<const Type> types) const override {
    return apply_vector<Intrinsic::kNumArgs>(
        [](auto... args) { return Intrinsic::Name(args...); }, types);
  }
  std::string GenerateMangledSimdPrefix(
      absl::Span<const Type> types) const override {
    std::vector<math::VecParamCardinality> param_cardinalities;
    auto front = types.front();
    // Remove the return type if it's in the types list:
    for (const auto& type :
         types.first(types.size() - Intrinsic::kLastArgIsReturnType)) {
      if (type.is_scalar()) {
        param_cardinalities.push_back(math::VecParamCardinality::kScalar);
      } else {
        param_cardinalities.push_back(math::VecParamCardinality::kVector);
      }
      CHECK(type.vector_width() == front.vector_width())
          << "All types must have the same vector width.";
    }
    return math::GetMangledNamePrefix(Intrinsic::kIsMasked,
                                      front.vector_width().value_or(1),
                                      param_cardinalities);
  }
};

MathFunctionLib::MathFunctionLib(llvm::TargetMachine* target_machine)
    : target_machine_(target_machine) {
  math_functions_.push_back(
      std::make_unique<IntrinsicAdapter<intrinsics::Ldexp>>());
  math_functions_.push_back(
      std::make_unique<IntrinsicAdapter<intrinsics::Exp>>());
  math_functions_.push_back(
      std::make_unique<IntrinsicAdapter<intrinsics::FpTrunc>>());
  math_functions_.push_back(
      std::make_unique<IntrinsicAdapter<intrinsics::Log1p>>());
  math_functions_.push_back(
      std::make_unique<IntrinsicAdapter<intrinsics::Erf>>());
  math_functions_.push_back(
      std::make_unique<IntrinsicAdapter<intrinsics::Rsqrt>>());
  math_functions_.push_back(
      std::make_unique<IntrinsicAdapter<intrinsics::Tanh>>());
}

namespace {

// Iterate all function calls in LLVM IR and call callback.
void VisitFunctionCalls(llvm::Module& module,
                        std::function<void(llvm::CallInst&)> callback) {
  for (llvm::Function& function : module) {
    for (llvm::BasicBlock& block : function) {
      for (llvm::Instruction& inst : block) {
        if (llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
          callback(*call);
        }
      }
    }
  }
}

// Returns the VecCallInfo that we need to generate definitions for all calls
// to math approximations in the module. Assumes that the module has already
// been optimized and that all calls to math approximations are unary.
absl::flat_hash_map<absl::string_view, absl::flat_hash_set<absl::string_view>>
GetCalledApproximatableFunctions(
    llvm::Module& module,
    absl::flat_hash_map<absl::string_view, absl::string_view> targets) {
  absl::flat_hash_map<absl::string_view, absl::flat_hash_set<absl::string_view>>
      called_targets;
  VisitFunctionCalls(module, [&](const llvm::CallInst& call) {
    if (auto it = targets.find(call.getCalledFunction()->getName());
        it != targets.end()) {
      called_targets[it->second].insert(it->first);
    }
  });
  return called_targets;
}

}  // anonymous namespace

std::vector<llvm::VecDesc> MathFunctionLib::Vectorizations() {
  std::vector<llvm::VecDesc> vec_descs;
  for (const auto& math_func : math_functions_) {
    // For each floating point type supported, we add all vector widths to every
    // other vector width as a possible vectorization.
    for (const auto& target_types :
         math_func->SupportedVectorTypes(target_machine_)) {
      for (const auto& vector_types :
           math_func->SupportedVectorTypes(target_machine_)) {
        if (target_types.front().element_type() !=
            vector_types.front().element_type()) {
          continue;
        }
        absl::string_view target_name = math::StringInterner::Get().Intern(
            math_func->GenerateVectorizedFunctionName(target_types));
        absl::string_view vec_name = math::StringInterner::Get().Intern(
            math_func->GenerateVectorizedFunctionName(vector_types));
        targets_[vec_name] = math_func->FunctionName();
        if (target_name == vec_name) {
          continue;
        }
        size_t vector_width = vector_types.front().vector_width().value_or(1);
        llvm::VecDesc vec_desc = {
            target_name,
            vec_name,
            llvm::ElementCount::getFixed(vector_width),
            false,
            math::StringInterner::Get().Intern(
                math_func->GenerateMangledSimdPrefix(vector_types)),
            std::nullopt};
        vec_descs.push_back(vec_desc);
      }
    }
  }
  return vec_descs;
}

void CreateDefinitionAndReplaceDeclaration(llvm::Module& module,
                                           absl::string_view name,
                                           llvm::TargetMachine* target_machine,
                                           MathFunction& math_func) {
  // The Vectorization pass may have already inserted a declaration
  // of this function that we need to rename and later remove to avoid
  // name collisions.
  llvm::Function* existing_func = module.getFunction(name);
  if (existing_func && existing_func->isDeclaration()) {
    existing_func->setName(std::string(name) + ".old_decl");
  }
  llvm::Function* definition =
      math_func.CreateDefinition(module, target_machine, name);
  definition->setLinkage(llvm::Function::InternalLinkage);
  definition->addFnAttr(llvm::Attribute::AlwaysInline);
  llvm::verifyFunction(*definition);
  if (existing_func && existing_func->isDeclaration()) {
    // Remove the declaration and replace all uses with the
    // new definition.
    existing_func->replaceAllUsesWith(definition);
    existing_func->eraseFromParent();
  }
}

absl::flat_hash_set<absl::string_view> MathFunctionLib::RewriteMathFunctions(
    llvm::Module& module) {
  // Find each called target function, generate the definition and insert it
  // into the module.
  // Keep track of the function names we replaced so we can remove them from
  // llvm.compiler.used later.
  absl::flat_hash_set<absl::string_view> replaced_functions;
  for (const auto& [function_name, signatures] :
       GetCalledApproximatableFunctions(module, targets_)) {
    for (const auto& math_func : math_functions_) {
      if (math_func->FunctionName() == function_name) {
        for (const auto& signature : signatures) {
          CreateDefinitionAndReplaceDeclaration(module, signature,
                                                target_machine_, *math_func);
          replaced_functions.insert(signature);
        }
      }
    }
  }

  CHECK(!llvm::verifyModule(module)) << "Module is invalid after optimization\n"
                                     << llvm_ir::DumpToString(&module);
  return replaced_functions;
}

}  // namespace xla::codegen
