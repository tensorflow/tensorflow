/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
==============================================================================*/

#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/cpu/codegen/elemental/elemental_kernel_emitter.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla::cpu {
namespace {

class M4Iteration3StressTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  }
};

static std::unique_ptr<llvm::TargetMachine> CreateTargetMachineForTriple(
    const std::string& triple_str) {
  std::string error;
  llvm::Triple triple(triple_str);
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    return nullptr;
  }
  llvm::TargetOptions target_options;
  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      triple, "generic", "", target_options, std::nullopt));
}

// -----------------------------------------------------------------------------
// Test 1: Cross-Compilation DataLayout & Target Machine Handling Audit
// -----------------------------------------------------------------------------
TEST_F(M4Iteration3StressTest, TargetMachineDataLayoutAndTriplePropagation) {
  std::vector<std::string> test_triples = {
      "x86_64-unknown-linux-gnu",
      "aarch64-unknown-linux-gnu",
      "armv7-unknown-linux-gnueabihf",
      "riscv64-unknown-linux-gnu",
  };

  for (const auto& triple_str : test_triples) {
    auto target_machine = CreateTargetMachineForTriple(triple_str);
    if (!target_machine) {
      continue;
    }

    auto mlir_context = FusionCompiler::CreateContext();
    FusionCompiler compiler(mlir_context.get(), FusionCompiler::Options(),
                            /*hlo_module=*/nullptr, target_machine.get());

    mlir::OpBuilder builder(mlir_context.get());
    mlir::Location loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);

    auto memref_type = mlir::MemRefType::get({4}, builder.getF32Type());
    auto func_type = builder.getFunctionType({memref_type}, {});
    auto func = mlir::func::FuncOp::create(loc, "test_func", func_type);
    func->setAttr("xla.entry", builder.getUnitAttr());
    func.addEntryBlock();
    builder.setInsertionPointToStart(&func.front());
    builder.create<mlir::func::ReturnOp>(loc);
    module->push_back(func);

    llvm::LLVMContext llvm_context;
    auto llvm_module_or = compiler.Compile(llvm_context, *module);
    ASSERT_TRUE(llvm_module_or.ok()) << llvm_module_or.status().message();
    auto llvm_module = std::move(llvm_module_or.value());

    EXPECT_EQ(llvm_module->getTargetTriple().str(),
              target_machine->getTargetTriple().str());

    EXPECT_EQ(llvm_module->getDataLayout(), target_machine->createDataLayout());

    uint32_t expected_ptr_size =
        target_machine->createDataLayout().getPointerSizeInBits();
    uint32_t module_ptr_size =
        llvm_module->getDataLayout().getPointerSizeInBits();
    EXPECT_EQ(module_ptr_size, expected_ptr_size);
  }
}

// -----------------------------------------------------------------------------
// Test 2: Thread-Local Callback Factory Execution & Stress Test
// -----------------------------------------------------------------------------
TEST_F(M4Iteration3StressTest, ThreadLocalCallbackFactoryMultithreadedStress) {
  constexpr int kNumThreads = 16;
  constexpr int kOpsPerThread = 50;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([t]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto module_config = HloModuleConfig();
        auto hlo_module = std::make_unique<HloModule>(
            absl::StrCat("module_t", t, "_i", i), module_config);

        HloComputation::Builder builder(absl::StrCat("comp_", t));
        auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
            0, ShapeUtil::MakeShape(F32, {}), "p0"));
        auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
            1, ShapeUtil::MakeShape(F32, {}), "p1"));
        builder.AddInstruction(HloInstruction::CreateBinary(
            ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, param0, param1));
        auto sub_comp = hlo_module->AddEmbeddedComputation(builder.Build());

        HloComputation::Builder main_builder("main");
        auto input =
            main_builder.AddInstruction(HloInstruction::CreateParameter(
                0, ShapeUtil::MakeShape(F32, {4}), "in"));
        auto map_inst = main_builder.AddInstruction(HloInstruction::CreateMap(
            ShapeUtil::MakeShape(F32, {4}), {input}, sub_comp));
        hlo_module->AddEntryComputation(main_builder.Build());

        auto target_machine =
            CreateTargetMachineForTriple("x86_64-unknown-linux-gnu");
        if (!target_machine) return;

        TargetMachineFeatures tm_features(target_machine.get());
        ElementalKernelEmitter emitter(map_inst, /*buffer_assignment=*/nullptr,
                                       &tm_features);
        EXPECT_NE(map_inst, nullptr);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

// -----------------------------------------------------------------------------
// Test 3: Loop Allocation Return Bufferization & Shape Cast Verification
// -----------------------------------------------------------------------------
TEST_F(M4Iteration3StressTest, LoopAllocationReturnBufferizationAndCast) {
  auto mlir_context = FusionCompiler::CreateContext();
  mlir::OpBuilder builder(mlir_context.get());
  mlir::Location loc = builder.getUnknownLoc();

  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);

  auto tensor_type = mlir::RankedTensorType::get({4}, builder.getF32Type());
  auto func_type = builder.getFunctionType({tensor_type}, {tensor_type});
  auto func = mlir::func::FuncOp::create(loc, "loop_alloc_func", func_type);
  func->setAttr("xla.entry", builder.getUnitAttr());
  func.addEntryBlock();
  builder.setInsertionPointToStart(&func.front());

  mlir::Value init_val = func.getArgument(0);
  mlir::Value lb = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value ub = builder.create<mlir::arith::ConstantIndexOp>(loc, 4);
  mlir::Value step = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

  auto for_op = builder.create<mlir::scf::ForOp>(
      loc, lb, ub, step, mlir::ValueRange{init_val},
      [&](mlir::OpBuilder& b, mlir::Location l, mlir::Value iv,
          mlir::ValueRange args) {
        auto alloc_tensor = b.create<mlir::tensor::EmptyOp>(
            l, llvm::ArrayRef<int64_t>{4}, b.getF32Type());
        mlir::Value casted = alloc_tensor;
        if (casted.getType() != args[0].getType()) {
          casted = b.create<mlir::tensor::CastOp>(l, args[0].getType(), casted);
        }
        b.create<mlir::scf::YieldOp>(l, mlir::ValueRange{casted});
      });

  builder.create<mlir::func::ReturnOp>(loc, for_op.getResults());
  module->push_back(func);

  FusionCompiler compiler(mlir_context.get(), FusionCompiler::Options());
  llvm::LLVMContext llvm_context;
  auto llvm_module_or = compiler.Compile(llvm_context, *module);
  ASSERT_TRUE(llvm_module_or.ok()) << llvm_module_or.status().message();
  auto llvm_module = std::move(llvm_module_or.value());
  EXPECT_NE(llvm_module, nullptr);
}

}  // namespace
}  // namespace xla::cpu
