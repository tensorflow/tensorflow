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

#include "xla/codegen/math_approximations_lib.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/FMF.h"
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
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "xla/codegen/math/ldexp.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::codegen {

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
absl::flat_hash_map<absl::string_view, absl::flat_hash_set<PrimitiveType>>
GetCalledApproximatableFunctions(
    llvm::Module& module,
    absl::flat_hash_map<std::string, absl::string_view> target_to_approx) {
  absl::flat_hash_map<absl::string_view, absl::flat_hash_set<PrimitiveType>>
      called_targets;
  VisitFunctionCalls(module, [&](const llvm::CallInst& call) {
    if (auto it = target_to_approx.find(call.getCalledFunction()->getName());
        it != target_to_approx.end()) {
      called_targets[it->second].insert(
          llvm_ir::PrimitiveTypeFromIrType(call.getArgOperand(0)->getType()));
    }
  });
  return called_targets;
}

}  // anonymous namespace

class LdexpF64MathFunction final : public MathFunction {
 public:
  absl::string_view FunctionName() const override { return "ldexp"; }
  std::vector<std::string> TargetFunctions() const override {
    return {"xla.ldexp.f64.i32"};
  }
  std::vector<VectorType> SupportedVectorTypes() const override {
    return {
        {xla::F64, 1},
        {xla::F64, 2},
        {xla::F64, 4},
        {xla::F64, 8},
    };
  }

  std::string GenerateVectorizedFunctionName(
      VectorType vector_type) const override {
    return math::LdexpF64FunctionName(vector_type.width);
  }

  std::string GenerateMangledSimdName(VectorType vector_type) const override {
    return absl::StrCat(MathFunction::GenerateMangledSimdName(vector_type),
                        "v");
  }

  llvm::Function* CreateDefinition(llvm::Module& module, absl::string_view name,
                                   VectorType vector_type) const override {
    llvm::Type* float_type =
        llvm_ir::PrimitiveTypeToIrType(vector_type.dtype, module.getContext());
    llvm::Type* vec_type = float_type;
    if (vector_type.width > 1) {
      vec_type = llvm::VectorType::get(float_type, vector_type.width, false);
    }
    return math::CreateLdexpF64(&module, vec_type);
  }
};

MathFunctionLib::MathFunctionLib() {
  math_functions_.push_back(std::make_unique<LdexpF64MathFunction>());

  for (const auto& math_approximation : math_functions_) {
    for (const absl::string_view target_function :
         math_approximation->TargetFunctions()) {
      target_to_approx_[target_function] = math_approximation->FunctionName();
    }
  }
}

std::vector<llvm::VecDesc> MathFunctionLib::Vectorizations() {
  std::vector<llvm::VecDesc> vec_descs;
  for (const auto& math_func : math_functions_) {
    for (const std::string& target_function : math_func->TargetFunctions()) {
      const std::string& target_function_interned =
          InternString(target_function);
      for (const auto& vector_type : math_func->SupportedVectorTypes()) {
        llvm::VecDesc vec_desc = {
            target_function_interned,
            InternString(
                math_func->GenerateVectorizedFunctionName(vector_type)),
            llvm::ElementCount::getFixed(vector_type.width),
            false,
            InternString(math_func->GenerateMangledSimdName(vector_type)),
            std::nullopt};
        vec_descs.push_back(vec_desc);
      }
    }
  }
  return vec_descs;
}
void MathFunctionLib::BeforeOptimization(llvm::Module& module) {
  // Find each called target function, generate the definition and insert it
  // into the module. We do not need to replace the calls with the generated
  // definitions, because the calls will be replaced by the vector passes (see
  // Vectorizations) and then rely on the approximation definitions presence
  // in the module.
  for (const auto& [function_name, dtypes] :
       GetCalledApproximatableFunctions(module, target_to_approx_)) {
    for (const auto& math_func : math_functions_) {
      if (math_func->FunctionName() == function_name) {
        for (const auto& vector_type : math_func->SupportedVectorTypes()) {
          if (dtypes.contains(vector_type.dtype)) {
            const std::string& name = InternString(
                math_func->GenerateVectorizedFunctionName(vector_type));
            llvm::Function* definition =
                math_func->CreateDefinition(module, name, vector_type);
            // definition->setLinkage(llvm::Function::ExternalLinkage);
            definition->addFnAttr(llvm::Attribute::AlwaysInline);
            llvm::verifyFunction(*definition);
            definitions_.push_back(name);
          }
        }
      }
    }
  }
}

void MathFunctionLib::AfterOptimization(
    llvm::Module& module, const llvm::PipelineTuningOptions& pto,
    const llvm::OptimizationLevel opt_level) {
  for (absl::string_view name : definitions_) {
    llvm::Function* definition = module.getFunction(name);
    if (!definition) {
      continue;
    }
    definition->setLinkage(llvm::Function::InternalLinkage);
  }

  // Re-optimize the module to get rid of any unused definitions and find
  // efficiencies post-vectorization replacement.
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassInstrumentationCallbacks pic;
  llvm::StandardInstrumentations si(module.getContext(), false);
  si.registerCallbacks(pic, &mam);

  std::string error_string;
  llvm::Triple target_triple(module.getTargetTriple());
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(
      module.getTargetTriple(), error_string);
  if (!target) {
    llvm::errs() << "error: could not create target: " << error_string << "\n";
    return;
  }
  llvm::TargetOptions options;
  std::unique_ptr<llvm::TargetMachine> target_machine(
      target->createTargetMachine(module.getTargetTriple(), "", "", options,
                                  llvm::Reloc::PIC_));

  llvm::PassBuilder pb(target_machine.get(), pto, {}, &pic);

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager pm;

  if (opt_level == llvm::OptimizationLevel::O0) {
    pm.addPass(pb.buildO0DefaultPipeline(opt_level));
  } else {
    pm.addPass(pb.buildPerModuleDefaultPipeline(opt_level));
  }

  pm.run(module, mam);
  CHECK(!llvm::verifyModule(module)) << "Module is invalid after optimization\n"
                                     << llvm_ir::DumpToString(&module);
}

}  // namespace xla::codegen
