/* Copyright 2018 The OpenXLA Authors.

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

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/service/llvm_ir/alias_analysis.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

class CpuNoAliasTest : public CpuCodegenTest {};

// Creates a simple HLO ir_module (runs concat(concat(x, y), x)), and then
// inspects the aliasing information for loads to its buffers.
TEST_F(CpuNoAliasTest, Concat) {
  HloComputation::Builder builder(TestName());

  Literal literal = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto param_shape = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* param_x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "x"));
  HloInstruction* param_y = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "y"));
  HloInstruction* concat1 =
      builder.AddInstruction(HloInstruction::CreateConcatenate(
          ShapeUtil::MakeShape(F32, {2, 4}), {param_x, param_y}, 1));
  HloInstruction* concat2 =
      builder.AddInstruction(HloInstruction::CreateConcatenate(
          ShapeUtil::MakeShape(F32, {2, 6}), {concat1, param_x}, 1));
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {2, 6}), HloOpcode::kAdd, concat2, concat2));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  // Now that we have an HLO module, build an llvm_ir::AliasAnalysis for it.
  auto status_or_buffer_assn = BufferAssigner::Run(
      hlo_module.get(),
      std::make_unique<DependencyHloOrdering>(hlo_module.get()),
      backend().compiler()->BufferSizeBytesFunction(),
      [](LogicalBuffer::Color) { return /*alignment=*/1; });
  ASSERT_EQ(status_or_buffer_assn.status(), absl::OkStatus());

  llvm::LLVMContext context;
  llvm_ir::AliasAnalysis aa(*hlo_module, *status_or_buffer_assn.value(),
                            &context);

  // Construct an LLVM module containing loads that we annotate as being from
  // the buffers in the HLO module.  We'll inspect these loads to ensure that
  // they have the expected alias information.
  llvm::Module ir_module("test", context);
  llvm::Function* func = llvm::dyn_cast<llvm::Function>(
      ir_module.getOrInsertFunction("test_fn", llvm::Type::getVoidTy(context))
          .getCallee());
  llvm::BasicBlock* bb = llvm::BasicBlock::Create(context, "body", func);
  llvm::IRBuilder<> b(bb);
  auto* zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0);

  llvm::ArrayType* array2d_type = llvm::ArrayType::get(
      llvm::ArrayType::get(llvm::Type::getFloatTy(context), 100), 100);

  {
    auto param_x_val = llvm::cast<llvm::GlobalVariable>(
        ir_module.getOrInsertGlobal("param_x", array2d_type));
    llvm_ir::IrArray param_x_array(param_x_val, param_x_val->getValueType(),
                                   param_shape);
    aa.AddAliasingInformationToIrArray(*param_x, &param_x_array);
    llvm_ir::IrArray::Index zero_2d({zero, zero}, param_shape, zero->getType());
    param_x_array.EmitReadArrayElement(zero_2d, &b)
        ->setName("read_param_x_array");
  }

  {
    auto concat1_val = llvm::cast<llvm::GlobalVariable>(
        ir_module.getOrInsertGlobal("concat1", array2d_type));
    auto shape = ShapeUtil::MakeShape(F32, {2, 4});
    llvm_ir::IrArray concat1_array(concat1_val, concat1_val->getValueType(),
                                   shape);
    aa.AddAliasingInformationToIrArray(*concat1, &concat1_array);
    llvm_ir::IrArray::Index zero_2d({zero, zero}, shape, zero->getType());
    concat1_array.EmitReadArrayElement(zero_2d, &b)
        ->setName("read_concat1_array");
  }

  {
    auto concat2_val = llvm::cast<llvm::GlobalVariable>(
        ir_module.getOrInsertGlobal("concat2", array2d_type));
    auto shape = ShapeUtil::MakeShape(F32, {2, 6});
    llvm_ir::IrArray concat2_array(concat2_val, concat2_val->getValueType(),
                                   shape);
    aa.AddAliasingInformationToIrArray(*concat2, &concat2_array);
    llvm_ir::IrArray::Index zero_2d({zero, zero}, shape, zero->getType());
    concat2_array.EmitReadArrayElement(zero_2d, &b)
        ->setName("read_concat2_array");
  }

  {
    auto concat2_val = llvm::cast<llvm::GlobalVariable>(
        ir_module.getOrInsertGlobal("add", array2d_type));
    auto shape = ShapeUtil::MakeShape(F32, {2, 6});
    llvm_ir::IrArray add_array(concat2_val, concat2_val->getValueType(), shape);
    aa.AddAliasingInformationToIrArray(*add, &add_array);
    llvm_ir::IrArray::Index zero_2d({zero, zero}, shape, zero->getType());
    add_array.EmitReadArrayElement(zero_2d, &b)->setName("read_add_array");
  }

  // Check the AA info in the loads.
  const char* filecheck_pattern = R"(
    CHECK: %read_param_x_array = load {{.*}} !noalias [[param_x_noalias:![0-9]+]]
    CHECK: %read_concat1_array = load {{.*}} !alias.scope [[concat1_scope:![0-9]+]], !noalias [[concat1_noalias:![0-9]+]]
    CHECK: %read_concat2_array = load {{.*}} !alias.scope [[concat1_noalias]], !noalias [[concat1_scope]]
    CHECK: %read_add_array = load {{.*}} !alias.scope [[concat1_noalias]]{{$}}
    CHECK-DAG: [[buf_size32:![0-9]+]] = !{!"buffer:{{.*}} size:32
    CHECK-DAG: [[buf_size48:![0-9]+]] = !{!"buffer:{{.*}} size:48
    CHECK-DAG: [[param_x_noalias]] = !{[[buf_size48]], [[buf_size32]]}
    CHECK-DAG: [[concat1_scope]] = !{[[buf_size32]]}
    CHECK-DAG: [[concat1_noalias]] = !{[[buf_size48]]}
  )";

  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_match,
      RunFileCheck(llvm_ir::DumpToString(&ir_module), filecheck_pattern));
  EXPECT_TRUE(filecheck_match);
}

}  // namespace cpu
}  // namespace xla
