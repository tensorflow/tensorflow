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

#include "xla/python/ifrt/ir/ifrt_interfaces.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/ifrt_ir_test_base.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"

namespace xla {
namespace ifrt {
namespace {

class IfrtInterfacesTest : public test_util::IfrtIrTestBase {};

TEST_F(IfrtInterfacesTest, MembershipAndAccessorsAcrossAllOps) {
  constexpr const char* kModuleStr = R"mlir(
!array_d01 = !ifrt.array<tensor<2x2xi32>,
                         #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!array_d23 = !ifrt.array<tensor<2x2xi32>,
                         #ifrt.sharding_param<1x1 to [0] on 2>, [2,3]>
!array_3d = !ifrt.array<tensor<1x2x2xi32>,
                        #ifrt.sharding_param<1x1x1 to [0] on 2>, [0,1]>
!array_d0 = !ifrt.array<tensor<2x2xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array_d1 = !ifrt.array<tensor<2x2xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!array_sharded = !ifrt.array<tensor<2x4xi32>,
                             #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
module {
  func.func private @atom(%a: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %a : tensor<2x2xi32>
  }
  ifrt.LoadedExecutable @loaded_exec on devices [0,1]
      : (!array_d01) -> !array_d01

  func.func @main(%arg0: !array_d01, %arg1: !array_d0, %arg2: !array_d1,
                  %ctrl_in: !ifrt.control) attributes {ifrt.function} {
    %r0, %ctrl_r = ifrt.Reshard(%arg0) after %ctrl_in <donated = true>
        : (!array_d01) -> !array_d23
    %c0, %ctrl_c = ifrt.CopyArrays(%arg0) after %ctrl_r
        : (!array_d01) -> !array_d23
    %b0, %ctrl_b = ifrt.BitcastArrays(%arg0) after %ctrl_c <donated = true>
        : (!array_d01) -> !array_3d
    %m0, %ctrl_m = ifrt.RemapArrays(%arg1, %arg2)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                  #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
        after %ctrl_b <donated = true>
        : (!array_d0, !array_d1) -> (!array_sharded)
    %call0, %ctrl_call = ifrt.Call @atom(%arg0) after %ctrl_m on devices [0,1]
        : (!array_d01) -> !array_d01
    %le0, %ctrl_le = ifrt.CallLoadedExecutable @loaded_exec(%call0)
        after %ctrl_call : (!array_d01) -> !array_d01
    %after_ctrl = "ifrt.After"(%le0) : (!array_d01) -> !ifrt.control
    %asm, %ctrl_asm = ifrt.Assemble(%arg1, %arg2)
        : (!array_d0, !array_d1) -> !array_sharded
    %d0, %d1, %ctrl_dis = ifrt.Disassemble(%asm)
        : (!array_sharded) -> (!array_d0, !array_d1)
    return
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));

  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);

  int array_op_count = 0;
  int pairwise_count = 0;
  for (mlir::Operation& op : main_func.getBody().front()) {
    if (auto array_op = llvm::dyn_cast<IfrtArrayOpInterface>(&op)) {
      ++array_op_count;
      EXPECT_FALSE(array_op.getInputs().empty());
      EXPECT_FALSE(array_op.getOutputs().empty());
      EXPECT_EQ(array_op.getControlInputs().size(), 1u);
      EXPECT_TRUE(array_op.getControlOutput() != nullptr);
      EXPECT_EQ(array_op.getInputsMutable().size(),
                array_op.getInputs().size());
      EXPECT_EQ(array_op.getControlInputsMutable().size(), 1u);
    } else {
      EXPECT_TRUE(
          (llvm::isa<AfterOp, AssembleOp, DisassembleOp, mlir::func::ReturnOp>(
              &op)));
    }

    if (auto pairwise_op = llvm::dyn_cast<IfrtPairwiseArrayOpInterface>(&op)) {
      ++pairwise_count;
      EXPECT_TRUE((llvm::isa<ReshardOp, CopyArraysOp, BitcastArraysOp>(&op)));
      IfrtArrayOpInterface as_base = pairwise_op;
      EXPECT_EQ(as_base.getOperation(), &op);
      EXPECT_EQ(pairwise_op.getInputs().size(), 1u);
      EXPECT_EQ(pairwise_op.getOutputs().size(), 1u);
      mlir::OpResult out0 = pairwise_op.getOutputs()[0];
      EXPECT_EQ(pairwise_op.getInputForOutput(out0),
                pairwise_op.getInputs()[0]);
      if (llvm::isa<CopyArraysOp>(&op)) {
        EXPECT_FALSE(pairwise_op.getDonated());
      } else {
        EXPECT_TRUE(pairwise_op.getDonated());
      }
    } else {
      EXPECT_FALSE((llvm::isa<ReshardOp, CopyArraysOp, BitcastArraysOp>(&op)));
    }
  }
  EXPECT_EQ(array_op_count, 6);
  EXPECT_EQ(pairwise_count, 3);
}

TEST_F(IfrtInterfacesTest, CloneAndReplaceReshardOp) {
  constexpr const char* kModuleStr = R"mlir(
!a0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!a1 = !ifrt.array<tensor<4xf32>,
                  #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
!a2 = !ifrt.array<tensor<8xi64>,
                  #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
!b0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<2x1 to [0] on 2>, [2,3]>
!b1 = !ifrt.array<tensor<4xf32>,
                  #ifrt.sharding_param<1 to [0] on 2>, [2,3]>
!b2 = !ifrt.array<tensor<8xi64>,
                  #ifrt.sharding_param<1 to [0] on 2>, [2,3]>
module {
  func.func @main(%x0: !a0, %x1: !a1, %x2: !a2, %ctrl_in: !ifrt.control)
      -> (!b0, !b2, !ifrt.control) attributes {ifrt.function} {
    %y0, %y1, %y2, %ctrl = ifrt.Reshard(%x0, %x1, %x2) after %ctrl_in
        <donated = true> {test.tag = "reshard_tag"}
        : (!a0, !a1, !a2) -> (!b0, !b1, !b2)
    return %y0, %y2, %ctrl : !b0, !b2, !ifrt.control
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);

  auto reshard_op = *main_func.getOps<ReshardOp>().begin();
  EXPECT_EQ(reshard_op.getInputForOutput(reshard_op.getOutputs()[2]),
            main_func.getArgument(2));

  mlir::IRRewriter rewriter(&mlir_context_);

  // keep.all() is a no-op returning the original op.
  llvm::BitVector keep_all(3, true);
  IfrtPairwiseArrayOpInterface same_op =
      reshard_op.replaceWithKeptOutputs(rewriter, keep_all);
  EXPECT_EQ(same_op.getOperation(), reshard_op.getOperation());

  // Drop middle output %y1 (index 1); keep index 0 and index 2.
  llvm::BitVector keep(3, true);
  keep.reset(1);
  IfrtPairwiseArrayOpInterface new_iface =
      reshard_op.replaceWithKeptOutputs(rewriter, keep);
  ASSERT_TRUE(new_iface);
  auto new_reshard = llvm::dyn_cast<ReshardOp>(new_iface.getOperation());
  ASSERT_TRUE(new_reshard);
  EXPECT_EQ(new_reshard.getInputs().size(), 2u);
  EXPECT_EQ(new_reshard.getOutputs().size(), 2u);
  EXPECT_EQ(new_reshard.getInputs()[0], main_func.getArgument(0));
  EXPECT_EQ(new_reshard.getInputs()[1], main_func.getArgument(2));
  EXPECT_EQ(new_reshard.getControlInputs().size(), 1u);
  EXPECT_EQ(new_reshard.getControlInputs()[0], main_func.getArgument(3));
  EXPECT_TRUE(new_reshard.getDonated());
  EXPECT_EQ(new_reshard->getDiscardableAttr("test.tag"),
            mlir::StringAttr::get(&mlir_context_, "reshard_tag"));

  auto ret_op = *main_func.getOps<mlir::func::ReturnOp>().begin();
  EXPECT_EQ(ret_op.getOperand(0), new_reshard.getOutputs()[0]);
  EXPECT_EQ(ret_op.getOperand(1), new_reshard.getOutputs()[1]);
  EXPECT_EQ(ret_op.getOperand(2), new_reshard.getControlOutput());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(IfrtInterfacesTest, ReplaceCopyArraysPreservesReuseAndErasesWhenEmpty) {
  constexpr const char* kModuleStr = R"mlir(
!a0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!a1 = !ifrt.array<tensor<4xf32>,
                  #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
!b0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [2,3]>
!b1 = !ifrt.array<tensor<4xf32>,
                  #ifrt.sharding_param<2 to [0] on 2>, [2,3]>
module {
  func.func @main(%x0: !a0, %x1: !a1, %ctrl_in: !ifrt.control) -> !b1
      attributes {ifrt.function} {
    %y0, %y1, %ctrl0 = ifrt.CopyArrays(%x0, %x1) after %ctrl_in
        <reuse = true> {test.copy_attr = 7 : i64}
        : (!a0, !a1) -> (!b0, !b1)
    %dead0, %ctrl1 = ifrt.CopyArrays(%x0) : (!a0) -> (!b0)
    return %y1 : !b1
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);

  llvm::SmallVector<CopyArraysOp> copy_ops(main_func.getOps<CopyArraysOp>());
  ASSERT_EQ(copy_ops.size(), 2u);

  mlir::IRRewriter rewriter(&mlir_context_);

  // Slice first CopyArraysOp to keep only output 1, preserving `reuse = true`.
  llvm::BitVector keep_second(2, false);
  keep_second.set(1);
  IfrtPairwiseArrayOpInterface sliced =
      copy_ops[0].replaceWithKeptOutputs(rewriter, keep_second);
  ASSERT_TRUE(sliced);
  auto new_copy = llvm::dyn_cast<CopyArraysOp>(sliced.getOperation());
  ASSERT_TRUE(new_copy);
  EXPECT_TRUE(new_copy.getReuse());
  EXPECT_FALSE(new_copy.getDonated());
  EXPECT_EQ(new_copy.getInputs().size(), 1u);
  EXPECT_EQ(new_copy.getInputs()[0], main_func.getArgument(1));
  EXPECT_EQ(new_copy.getControlInputs().size(), 1u);
  EXPECT_TRUE(new_copy->hasDiscardableAttr("test.copy_attr"));

  // Erase second CopyArraysOp completely when keep.none() and control unused.
  llvm::BitVector keep_none(1, false);
  IfrtPairwiseArrayOpInterface erased =
      copy_ops[1].replaceWithKeptOutputs(rewriter, keep_none);
  EXPECT_FALSE(erased);

  llvm::SmallVector<CopyArraysOp> remaining(main_func.getOps<CopyArraysOp>());
  EXPECT_EQ(remaining.size(), 1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(IfrtInterfacesTest, CloneAndReplaceBitcastArraysOp) {
  constexpr const char* kModuleStr = R"mlir(
!a0 = !ifrt.array<tensor<1x2x2xi32>,
                  #ifrt.sharding_param<1x1x1 to [0] on 2>, [0,1]>
!a1 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
module {
  func.func @main(%x0: !a0, %x1: !a1, %ctrl_in: !ifrt.control) -> !a1
      attributes {ifrt.function} {
    %y0, %y1, %ctrl = ifrt.BitcastArrays(%x0, %x1) after %ctrl_in
        <donated = true> {test.bitcast = true}
        : (!a0, !a1) -> (!a1, !a0)
    return %y0 : !a1
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);

  auto bitcast_op = *main_func.getOps<BitcastArraysOp>().begin();
  mlir::IRRewriter rewriter(&mlir_context_);
  rewriter.setInsertionPointToEnd(&main_func.getBody().front());

  // First test cloneWithKeptOutputs directly (insertion guard must preserve
  // rewriter insertion point and place cloned op right before bitcast_op).
  llvm::BitVector keep_first(2, false);
  keep_first.set(0);
  IfrtPairwiseArrayOpInterface cloned =
      bitcast_op.cloneWithKeptOutputs(rewriter, keep_first);
  ASSERT_TRUE(cloned);
  EXPECT_EQ(cloned->getNextNode(), bitcast_op.getOperation());
  EXPECT_EQ(rewriter.getInsertionPoint(), main_func.getBody().front().end());
  rewriter.eraseOp(cloned);

  // Now test replaceWithKeptOutputs via the interface handle.
  IfrtPairwiseArrayOpInterface iface = bitcast_op;
  IfrtPairwiseArrayOpInterface replaced =
      iface.replaceWithKeptOutputs(rewriter, keep_first);
  ASSERT_TRUE(replaced);
  auto new_bitcast = llvm::dyn_cast<BitcastArraysOp>(replaced.getOperation());
  ASSERT_TRUE(new_bitcast);
  EXPECT_TRUE(new_bitcast.getDonated());
  EXPECT_EQ(new_bitcast.getInputs().size(), 1u);
  EXPECT_EQ(new_bitcast.getOutputs().size(), 1u);
  EXPECT_EQ(new_bitcast.getControlInputs().size(), 1u);
  EXPECT_TRUE(new_bitcast->hasDiscardableAttr("test.bitcast"));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(IfrtInterfacesTest, CloneWithKeptOutputsReturnsNullWhenKeepNone) {
  constexpr const char* kModuleStr = R"mlir(
!a0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!b0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<2x1 to [0] on 2>, [2,3]>
module {
  func.func @main(%x0: !a0, %ctrl_in: !ifrt.control) -> !b0
      attributes {ifrt.function} {
    %y0, %ctrl = ifrt.Reshard(%x0) after %ctrl_in : (!a0) -> !b0
    return %y0 : !b0
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);

  ReshardOp reshard_op = *main_func.getOps<ReshardOp>().begin();
  mlir::IRRewriter rewriter(&mlir_context_);
  rewriter.setInsertionPointToEnd(&main_func.getBody().front());

  llvm::BitVector keep_none(1, false);
  IfrtPairwiseArrayOpInterface cloned =
      reshard_op.cloneWithKeptOutputs(rewriter, keep_none);
  EXPECT_FALSE(cloned);
  EXPECT_EQ(rewriter.getInsertionPoint(), main_func.getBody().front().end());

  llvm::SmallVector<ReshardOp> remaining(main_func.getOps<ReshardOp>());
  EXPECT_EQ(remaining.size(), 1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(IfrtInterfacesTest,
       ReplaceWithKeptOutputsForwardsSingleControlInputToUsers) {
  constexpr const char* kModuleStr = R"mlir(
!a0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!b0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<2x1 to [0] on 2>, [2,3]>
!c0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [2,3]>
module {
  func.func private @atom(%a: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %a : tensor<2x2xi32>
  }
  func.func @main(%x0: !a0, %ctrl_in0: !ifrt.control, %ctrl_in1: !ifrt.control)
      -> (!a0, !ifrt.control) attributes {ifrt.function} {
    %y0, %ctrl_r = ifrt.Reshard(%x0) after %ctrl_in0 : (!a0) -> !b0
    %cp0, %ctrl_c = ifrt.CopyArrays(%x0) after %ctrl_in1 : (!a0) -> !c0
    %out, %ctrl_call = ifrt.Call @atom(%x0) after %ctrl_in1, %ctrl_c
        on devices [0,1] : (!a0) -> !a0
    return %out, %ctrl_r : !a0, !ifrt.control
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);

  ReshardOp reshard_op = *main_func.getOps<ReshardOp>().begin();
  CopyArraysOp copy_op = *main_func.getOps<CopyArraysOp>().begin();
  CallOp call_op = *main_func.getOps<CallOp>().begin();
  mlir::func::ReturnOp return_op =
      *main_func.getOps<mlir::func::ReturnOp>().begin();
  mlir::IRRewriter rewriter(&mlir_context_);

  llvm::BitVector keep_none(1, false);
  EXPECT_FALSE(reshard_op.replaceWithKeptOutputs(rewriter, keep_none));
  EXPECT_FALSE(copy_op.replaceWithKeptOutputs(rewriter, keep_none));

  EXPECT_TRUE(main_func.getOps<ReshardOp>().empty());
  EXPECT_TRUE(main_func.getOps<CopyArraysOp>().empty());
  EXPECT_EQ(return_op.getOperand(1), main_func.getArgument(1));
  ASSERT_EQ(call_op.getControlInputs().size(), 1u);
  EXPECT_EQ(call_op.getControlInputs()[0], main_func.getArgument(2));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(IfrtInterfacesTest,
       ReplaceWithKeptOutputsDeduplicatesControlInputsForNonVariadicUser) {
  constexpr const char* kModuleStr = R"mlir(
!a0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!b0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<2x1 to [0] on 2>, [2,3]>
module {
  func.func @main(%x0: !a0, %ctrl_in0: !ifrt.control)
      -> !ifrt.control attributes {ifrt.function} {
    %y0, %ctrl_r = ifrt.Reshard(%x0) after %ctrl_in0, %ctrl_in0 : (!a0) -> !b0
    return %ctrl_r : !ifrt.control
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);

  ReshardOp reshard_op = *main_func.getOps<ReshardOp>().begin();
  mlir::func::ReturnOp return_op =
      *main_func.getOps<mlir::func::ReturnOp>().begin();
  mlir::IRRewriter rewriter(&mlir_context_);

  llvm::BitVector keep_none(1, false);
  EXPECT_FALSE(reshard_op.replaceWithKeptOutputs(rewriter, keep_none));

  EXPECT_TRUE(main_func.getOps<ReshardOp>().empty());
  EXPECT_EQ(return_op.getOperand(0), main_func.getArgument(1));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(IfrtInterfacesTest, ReplaceWithKeptOutputsDropsZeroControlInputs) {
  constexpr const char* kModuleStr = R"mlir(
!a0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!b0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<2x1 to [0] on 2>, [2,3]>
!c0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [2,3]>
module {
  func.func private @atom(%a: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %a : tensor<2x2xi32>
  }
  func.func @main(%x0: !a0, %other_ctrl: !ifrt.control) -> (!a0, !c0)
      attributes {ifrt.function} {
    %dead, %ctrl_dead = ifrt.Reshard(%x0) : (!a0) -> !b0
    %out0, %ctrl_call = ifrt.Call @atom(%x0) after %ctrl_dead
        on devices [0,1] : (!a0) -> !a0
    %out1, %ctrl_copy = ifrt.CopyArrays(%x0) after %ctrl_dead, %other_ctrl
        : (!a0) -> !c0
    return %out0, %out1 : !a0, !c0
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);

  ReshardOp reshard_op = *main_func.getOps<ReshardOp>().begin();
  CallOp call_op = *main_func.getOps<CallOp>().begin();
  CopyArraysOp copy_op = *main_func.getOps<CopyArraysOp>().begin();
  mlir::IRRewriter rewriter(&mlir_context_);

  llvm::BitVector keep_none(1, false);
  EXPECT_FALSE(reshard_op.replaceWithKeptOutputs(rewriter, keep_none));

  EXPECT_TRUE(main_func.getOps<ReshardOp>().empty());
  EXPECT_TRUE(call_op.getControlInputs().empty());
  ASSERT_EQ(copy_op.getControlInputs().size(), 1u);
  EXPECT_EQ(copy_op.getControlInputs()[0], main_func.getArgument(1));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(IfrtInterfacesTest,
       ReplaceWithKeptOutputsForwardsMultipleControlInputsToUsers) {
  constexpr const char* kModuleStr = R"mlir(
!a_d01 = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!b_d23 = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [2,3]>
!a_d0 = !ifrt.array<tensor<2x2xi32>,
                    #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!a_d1 = !ifrt.array<tensor<2x2xi32>,
                    #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
module {
  func.func private @atom(%a: tensor<2x4xi32>) -> tensor<2x4xi32> {
    return %a : tensor<2x4xi32>
  }
  func.func @main(%x0: !a_d01, %s0: !a_d0, %s1: !a_d1,
                  %c_pre: !ifrt.control, %c0: !ifrt.control,
                  %c1: !ifrt.control, %c_post: !ifrt.control)
      -> (!a_d01, !a_d0, !a_d1) attributes {ifrt.function} {
    %dead, %ctrl_dead = ifrt.CopyArrays(%x0) after %c0, %c1
        : (!a_d01) -> !b_d23
    %call_out, %ctrl_call = ifrt.Call @atom(%x0)
        after %c_pre, %ctrl_dead, %c_post on devices [0,1]
        : (!a_d01) -> !a_d01
    %asm_out, %ctrl_asm = ifrt.Assemble(%s0, %s1) after %ctrl_dead, %c1
        : (!a_d0, !a_d1) -> !a_d01
    %d0, %d1, %ctrl_dis = ifrt.Disassemble(%asm_out) after %ctrl_dead
        : (!a_d01) -> (!a_d0, !a_d1)
    return %call_out, %d0, %d1 : !a_d01, !a_d0, !a_d1
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);

  CopyArraysOp copy_op = *main_func.getOps<CopyArraysOp>().begin();
  CallOp call_op = *main_func.getOps<CallOp>().begin();
  AssembleOp assemble_op = *main_func.getOps<AssembleOp>().begin();
  DisassembleOp disassemble_op = *main_func.getOps<DisassembleOp>().begin();
  mlir::IRRewriter rewriter(&mlir_context_);

  llvm::BitVector keep_none(1, false);
  EXPECT_FALSE(copy_op.replaceWithKeptOutputs(rewriter, keep_none));

  EXPECT_TRUE(main_func.getOps<CopyArraysOp>().empty());

  mlir::Value c_pre = main_func.getArgument(3);
  mlir::Value c0 = main_func.getArgument(4);
  mlir::Value c1 = main_func.getArgument(5);
  mlir::Value c_post = main_func.getArgument(6);

  ASSERT_EQ(call_op.getControlInputs().size(), 4u);
  EXPECT_EQ(call_op.getControlInputs()[0], c_pre);
  EXPECT_EQ(call_op.getControlInputs()[1], c0);
  EXPECT_EQ(call_op.getControlInputs()[2], c1);
  EXPECT_EQ(call_op.getControlInputs()[3], c_post);

  ASSERT_EQ(assemble_op.getControlInputs().size(), 2u);
  EXPECT_EQ(assemble_op.getControlInputs()[0], c0);
  EXPECT_EQ(assemble_op.getControlInputs()[1], c1);

  ASSERT_EQ(disassemble_op.getControlInputs().size(), 2u);
  EXPECT_EQ(disassemble_op.getControlInputs()[0], c0);
  EXPECT_EQ(disassemble_op.getControlInputs()[1], c1);

  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(IfrtInterfacesTest, ControlOpInterfaceMembershipAndForwardingHelpers) {
  constexpr const char* kModuleStr = R"mlir(
!array_d01 = !ifrt.array<tensor<2x2xi32>,
                         #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!array_d0 = !ifrt.array<tensor<2x2xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array_d1 = !ifrt.array<tensor<2x2xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!array_sharded = !ifrt.array<tensor<2x4xi32>,
                             #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
module {
  func.func @main(%arg0: !array_d01, %arg1: !array_d0, %arg2: !array_d1,
                  %c0: !ifrt.control, %c1: !ifrt.control) -> !ifrt.control
      attributes {ifrt.function} {
    %after_ctrl = "ifrt.After"(%arg0) : (!array_d01) -> !ifrt.control
    %asm, %ctrl_asm = ifrt.Assemble(%arg1, %arg2) after %c0, %c1
        : (!array_d0, !array_d1) -> !array_sharded
    %d0, %d1, %ctrl_dis = ifrt.Disassemble(%asm) after %ctrl_asm
        : (!array_sharded) -> (!array_d0, !array_d1)
    %r0, %ctrl_r = ifrt.Reshard(%arg0) after %c0, %c1
        : (!array_d01) -> !array_d01
    return %ctrl_r : !ifrt.control
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_NE(main_func, nullptr);

  int control_op_count = 0;
  for (mlir::Operation& op : main_func.getBody().front()) {
    if (IfrtControlOpInterface control_op =
            llvm::dyn_cast<IfrtControlOpInterface>(&op);
        control_op != nullptr) {
      ++control_op_count;
      EXPECT_FALSE(control_op.getControlInputs().empty());
      EXPECT_NE(control_op.getControlOutput(), nullptr);
      EXPECT_EQ(control_op.getControlInputsMutable().size(),
                control_op.getControlInputs().size());
    } else {
      EXPECT_TRUE((llvm::isa<AfterOp, mlir::func::ReturnOp>(&op)));
    }
  }
  EXPECT_EQ(control_op_count, 3);

  AssembleOp assemble_op = *main_func.getOps<AssembleOp>().begin();
  DisassembleOp disassemble_op = *main_func.getOps<DisassembleOp>().begin();
  ReshardOp reshard_op = *main_func.getOps<ReshardOp>().begin();
  IfrtArrayOpInterface array_op = reshard_op;
  IfrtControlOpInterface base_from_array = array_op;
  EXPECT_EQ(base_from_array.getOperation(), reshard_op.getOperation());

  // `ctrl_asm` is consumed only by `DisassembleOp` (`IfrtControlOpInterface`),
  // so 0 or 2 control inputs can be forwarded.
  EXPECT_TRUE(CanForwardControlInputs(assemble_op.getControlOutput(),
                                      assemble_op.getControlInputs()));
  EXPECT_TRUE(CanForwardControlInputs(assemble_op.getControlOutput(),
                                      /*control_inputs=*/mlir::ValueRange()));

  // `ctrl_r` is consumed by `func.return` (non-variadic), so 2 distinct control
  // inputs or 0 control inputs cannot be forwarded, whereas 1 unique control
  // input (including duplicates) can.
  EXPECT_FALSE(CanForwardControlInputs(reshard_op.getControlOutput(),
                                       reshard_op.getControlInputs()));
  EXPECT_FALSE(CanForwardControlInputs(reshard_op.getControlOutput(),
                                       /*control_inputs=*/mlir::ValueRange()));
  llvm::SmallVector<mlir::Value> dup_single = {main_func.getArgument(3),
                                               main_func.getArgument(3)};
  EXPECT_TRUE(
      CanForwardControlInputs(reshard_op.getControlOutput(), dup_single));

  mlir::IRRewriter rewriter(&mlir_context_);
  ForwardControlInputsToUsers(assemble_op.getControlOutput(),
                              assemble_op.getControlInputs(), rewriter);
  ASSERT_EQ(disassemble_op.getControlInputs().size(), 2u);
  EXPECT_EQ(disassemble_op.getControlInputs()[0], main_func.getArgument(3));
  EXPECT_EQ(disassemble_op.getControlInputs()[1], main_func.getArgument(4));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(IfrtInterfacesTest, CloneWithNewInputsAndOutputs) {
  constexpr const char* kModuleStr = R"mlir(
!a0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!a1 = !ifrt.array<tensor<4xf32>,
                  #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
!b0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [2,3]>
!b1 = !ifrt.array<tensor<4xf32>,
                  #ifrt.sharding_param<2 to [0] on 2>, [2,3]>
module {
  func.func @main(%x0: !a0, %x1: !a1, %ctrl_in: !ifrt.control) -> (!b0, !b1)
      attributes {ifrt.function} {
    %y0, %ctrl0 = ifrt.CopyArrays(%x0) <reuse = true> {test.merged = "ok"}
        : (!a0) -> !b0
    %y1, %ctrl1 = ifrt.CopyArrays(%x1) <reuse = true> {test.merged = "ok"}
        : (!a1) -> !b1
    return %y0, %y1 : !b0, !b1
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_NE(main_func, nullptr);

  llvm::SmallVector<CopyArraysOp> copy_ops(main_func.getOps<CopyArraysOp>());
  ASSERT_EQ(copy_ops.size(), 2u);

  mlir::IRRewriter rewriter(&mlir_context_);
  rewriter.setInsertionPointAfter(copy_ops[1]);
  IfrtPairwiseArrayOpInterface first_iface = copy_ops.front();
  llvm::SmallVector<mlir::Value> merged_inputs = {main_func.getArgument(0),
                                                  main_func.getArgument(1)};
  llvm::SmallVector<mlir::Type> merged_output_types = {
      copy_ops[0].getOutputs()[0].getType(),
      copy_ops[1].getOutputs()[0].getType()};
  llvm::SmallVector<mlir::Value> merged_controls = {main_func.getArgument(2)};

  IfrtPairwiseArrayOpInterface merged_iface =
      first_iface.cloneWithNewInputsAndOutputs(
          rewriter, copy_ops[1].getLoc(), merged_inputs, merged_output_types,
          merged_controls);
  ASSERT_NE(merged_iface, nullptr);
  EXPECT_EQ(merged_iface.getLoc(), copy_ops[1].getLoc());
  EXPECT_EQ(merged_iface->getPrevNode(), copy_ops[1].getOperation());
  CopyArraysOp merged_copy =
      llvm::dyn_cast<CopyArraysOp>(merged_iface.getOperation());
  ASSERT_NE(merged_copy, nullptr);
  EXPECT_TRUE(merged_copy.getReuse());
  EXPECT_FALSE(merged_copy.getDonated());
  EXPECT_EQ(merged_copy.getInputs().size(), 2u);
  EXPECT_EQ(merged_copy.getOutputs().size(), 2u);
  ASSERT_EQ(merged_copy.getControlInputs().size(), 1u);
  EXPECT_EQ(merged_copy.getControlInputs()[0], main_func.getArgument(2));
  EXPECT_EQ(merged_copy->getDiscardableAttr("test.merged"),
            mlir::StringAttr::get(&mlir_context_, "ok"));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(IfrtInterfacesTest,
       CollectUsedOutputsReplaceKeptOutputsAndFilterDictArrayAttr) {
  constexpr const char* kModuleStr = R"mlir(
!a0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!b0 = !ifrt.array<tensor<2x2xi32>,
                  #ifrt.sharding_param<2x1 to [0] on 2>, [2,3]>
module {
  func.func @main(%x0: !a0) -> (!b0, !ifrt.control)
      attributes {ifrt.function} {
    %y0, %y1, %y2, %ctrl = ifrt.Reshard(%x0, %x0, %x0)
        : (!a0, !a0, !a0) -> (!b0, !b0, !b0)
    return %y2, %ctrl : !b0, !ifrt.control
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_NE(main_func, nullptr);

  ReshardOp reshard_op = *main_func.getOps<ReshardOp>().begin();
  UsedOutputsSlice slice = CollectUsedOutputs(reshard_op);
  ASSERT_EQ(slice.keep.size(), 3u);
  EXPECT_FALSE(slice.keep.test(0));
  EXPECT_FALSE(slice.keep.test(1));
  EXPECT_TRUE(slice.keep.test(2));
  EXPECT_TRUE(slice.erase.test(0));
  EXPECT_TRUE(slice.erase.test(1));
  EXPECT_FALSE(slice.erase.test(2));
  ASSERT_EQ(slice.old_to_new.size(), 3u);
  EXPECT_EQ(slice.old_to_new[0], -1);
  EXPECT_EQ(slice.old_to_new[1], -1);
  EXPECT_EQ(slice.old_to_new[2], 0);
  ASSERT_EQ(slice.kept_types.size(), 1u);
  EXPECT_EQ(slice.kept_types[0], reshard_op.getOutputs()[2].getType());

  mlir::IRRewriter rewriter(&mlir_context_);
  IfrtPairwiseArrayOpInterface cloned =
      reshard_op.cloneWithKeptOutputs(rewriter, slice.keep);
  ASSERT_NE(cloned, nullptr);
  ReplaceKeptOutputs(rewriter, reshard_op, cloned, slice.old_to_new);

  mlir::func::ReturnOp return_op =
      *main_func.getOps<mlir::func::ReturnOp>().begin();
  EXPECT_EQ(return_op.getOperand(0), cloned.getOutputs()[0]);
  EXPECT_EQ(return_op.getOperand(1), cloned.getControlOutput());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));

  // Verify `FilterDictArrayAttr` edge cases.
  EXPECT_EQ(FilterDictArrayAttr(/*dict_array=*/nullptr, slice.erase), nullptr);
  mlir::DictionaryAttr empty_dict = rewriter.getDictionaryAttr({});
  mlir::DictionaryAttr non_empty_dict = rewriter.getDictionaryAttr(
      {rewriter.getNamedAttr("ifrt.tag", rewriter.getUnitAttr())});
  mlir::ArrayAttr all_empty_after_filter =
      rewriter.getArrayAttr({non_empty_dict, empty_dict, empty_dict});
  EXPECT_EQ(FilterDictArrayAttr(all_empty_after_filter, slice.erase), nullptr);

  mlir::ArrayAttr kept_non_empty =
      rewriter.getArrayAttr({empty_dict, empty_dict, non_empty_dict});
  mlir::ArrayAttr filtered = FilterDictArrayAttr(kept_non_empty, slice.erase);
  ASSERT_NE(filtered, nullptr);
  ASSERT_EQ(filtered.size(), 1u);
  EXPECT_EQ(filtered[0], non_empty_dict);
}

TEST_F(IfrtInterfacesTest, CallOpInterfaceAndIoAliasQueries) {
  constexpr const char* kModuleStr = R"mlir(
!arr = !ifrt.array<tensor<2x2xi32>,
                   #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
module {
  func.func private @atom_a(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>,
                            %c: tensor<2x2xi32>)
      -> (tensor<2x2xi32>, tensor<2x2xi32>) {
    return %a, %b : tensor<2x2xi32>, tensor<2x2xi32>
  }
  func.func private @atom_b(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>,
                            %c: tensor<2x2xi32>)
      -> (tensor<2x2xi32>, tensor<2x2xi32>) {
    return %b, %a : tensor<2x2xi32>, tensor<2x2xi32>
  }
  ifrt.LoadedExecutable @exec_a on devices [0,1]
      : (!arr, !arr) -> (!arr, !arr)

  func.func @main(%x0: !arr, %x1: !arr, %x2: !arr)
      attributes {ifrt.function} {
    %o0, %o1, %ctrl0 = ifrt.Call @atom_a(%x0, %x1, %x2) on devices [0,1]
        <donated_input_indices = [0], io_aliases = [array<i32: 2, 1>]>
        : (!arr, !arr, !arr) -> (!arr, !arr)
    %l0, %l1, %ctrl1 = ifrt.CallLoadedExecutable @exec_a(%o0, %o1)
        <io_aliases = [array<i32: 0, 1>]>
        : (!arr, !arr) -> (!arr, !arr)
    %r0, %ctrl2 = ifrt.Reshard(%l0) : (!arr) -> !arr
    return
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_NE(main_func, nullptr);

  EXPECT_TRUE(GetIoAliases(/*io_aliases=*/nullptr).empty());

  int call_iface_count = 0;
  for (mlir::Operation& op : main_func.getBody().front()) {
    if (IfrtCallOpInterface call_iface =
            llvm::dyn_cast<IfrtCallOpInterface>(&op);
        call_iface != nullptr) {
      ++call_iface_count;
      IfrtArrayOpInterface as_array = call_iface;
      EXPECT_EQ(as_array.getOperation(), &op);
    } else {
      EXPECT_TRUE((llvm::isa<ReshardOp, mlir::func::ReturnOp>(&op)));
    }
  }
  EXPECT_EQ(call_iface_count, 2);

  CallOp call_op = *main_func.getOps<CallOp>().begin();
  IfrtCallOpInterface call_iface = call_op;
  EXPECT_EQ(call_iface.getCalleeAttr().getRootReference(), "atom_a");
  call_iface.setCalleeAttr(
      mlir::FlatSymbolRefAttr::get(&mlir_context_, "atom_b"));
  EXPECT_EQ(call_op.getCalleeAttr().getRootReference(), "atom_b");
  EXPECT_EQ(call_iface.getArgAttrsAttr(), nullptr);
  EXPECT_EQ(call_iface.getResAttrsAttr(), nullptr);

  llvm::SmallVector<IoAlias, 4> call_aliases = call_iface.getIoAliasPairs();
  ASSERT_EQ(call_aliases.size(), 1u);
  EXPECT_EQ(call_aliases[0], (IoAlias{2, 1}));
  EXPECT_NE(call_aliases[0], (IoAlias{1, 2}));

  llvm::SmallDenseSet<int32_t, 8> donated_or_aliased_inputs =
      GetDonatedOrAliasedInputIndices(call_iface);
  EXPECT_EQ(donated_or_aliased_inputs.size(), 2u);
  EXPECT_TRUE(donated_or_aliased_inputs.contains(0));
  EXPECT_FALSE(donated_or_aliased_inputs.contains(1));
  EXPECT_TRUE(donated_or_aliased_inputs.contains(2));

  llvm::SmallDenseSet<int32_t, 8> aliased_outputs =
      GetAliasedOutputIndices(call_iface);
  EXPECT_EQ(aliased_outputs.size(), 1u);
  EXPECT_FALSE(aliased_outputs.contains(0));
  EXPECT_TRUE(aliased_outputs.contains(1));

  CallLoadedExecutableOp call_exec_op =
      *main_func.getOps<CallLoadedExecutableOp>().begin();
  IfrtCallOpInterface exec_iface = call_exec_op;
  llvm::SmallVector<IoAlias, 4> exec_aliases = exec_iface.getIoAliasPairs();
  ASSERT_EQ(exec_aliases.size(), 1u);
  EXPECT_EQ(exec_aliases[0], (IoAlias{0, 1}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(IfrtInterfacesTest, OperandDonationAndAliasingQueries) {
  constexpr const char* kModuleStr = R"mlir(
!arr = !ifrt.array<tensor<2x2xi32>,
                   #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!arr_3d = !ifrt.array<tensor<1x2x2xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 2>, [0,1]>
!arr_d0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!arr_d1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!arr_sharded = !ifrt.array<tensor<2x4xi32>,
                           #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
module {
  func.func private @atom(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>,
                          %c: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %a : tensor<2x2xi32>
  }
  ifrt.LoadedExecutable @loaded_exec on devices [0,1]
      : (!arr, !arr, !arr) -> !arr

  func.func @main(%x_don: !arr, %x_alias: !arr, %x_plain: !arr,
                  %x_le_don: !arr, %x_le_alias: !arr, %x_le_plain: !arr,
                  %x_copy_plain: !arr, %x_copy_reuse: !arr, %x_copy_don: !arr,
                  %x_reshard_plain: !arr, %x_reshard_don: !arr,
                  %x_bitcast_alias: !arr, %x_bitcast_don: !arr,
                  %s_remap_alias: !arr_d0, %s_remap_don: !arr_d0,
                  %s0_alias: !arr_d0, %s1_alias: !arr_d1,
                  %s0_don: !arr_d0, %s1_don: !arr_d1, %ctrl: !ifrt.control)
      attributes {ifrt.function} {
    %c_out, %c_ctrl = ifrt.Call @atom(%x_don, %x_alias, %x_plain) after %ctrl
        on devices [0,1]
        <donated_input_indices = [0], io_aliases = [array<i32: 1, 0>]>
        : (!arr, !arr, !arr) -> !arr
    %le_out, %le_ctrl = ifrt.CallLoadedExecutable @loaded_exec(
        %x_le_don, %x_le_alias, %x_le_plain) after %ctrl
        <donated_input_indices = [0], io_aliases = [array<i32: 1, 0>]>
        : (!arr, !arr, !arr) -> !arr
    %cp0, %cp_ctrl0 = ifrt.CopyArrays(%x_copy_plain) : (!arr) -> !arr
    %cp1, %cp_ctrl1 = ifrt.CopyArrays(%x_copy_reuse) <reuse = true>
        : (!arr) -> !arr
    %cp2, %cp_ctrl2 = ifrt.CopyArrays(%x_copy_don) <donated = true>
        : (!arr) -> !arr
    %r0, %r_ctrl0 = ifrt.Reshard(%x_reshard_plain) : (!arr) -> !arr
    %r1, %r_ctrl1 = ifrt.Reshard(%x_reshard_don) <donated = true>
        : (!arr) -> !arr
    %b0, %b_ctrl0 = ifrt.BitcastArrays(%x_bitcast_alias) : (!arr) -> !arr_3d
    %b1, %b_ctrl1 = ifrt.BitcastArrays(%x_bitcast_don) <donated = true>
        : (!arr) -> !arr_3d
    %m0, %m_ctrl0 = ifrt.RemapArrays(%s_remap_alias)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        after %ctrl : (!arr_d0) -> !arr_d0
    %m1, %m_ctrl1 = ifrt.RemapArrays(%s_remap_don)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        after %ctrl <donated = true> : (!arr_d0) -> !arr_d0
    %asm0, %asm_ctrl0 = ifrt.Assemble(%s0_alias, %s1_alias) after %ctrl
        : (!arr_d0, !arr_d1) -> !arr_sharded
    %asm1, %asm_ctrl1 = ifrt.Assemble(%s0_don, %s1_don) after %ctrl
        <donated = true> : (!arr_d0, !arr_d1) -> !arr_sharded
    %d0_alias, %d1_alias, %dis_ctrl0 = ifrt.Disassemble(%asm0) after %ctrl
        : (!arr_sharded) -> (!arr_d0, !arr_d1)
    %d0_don, %d1_don, %dis_ctrl1 = ifrt.Disassemble(%asm1) after %ctrl
        <donated = true> : (!arr_sharded) -> (!arr_d0, !arr_d1)
    %after_ctrl = "ifrt.After"(%x_plain) : (!arr) -> !ifrt.control
    return
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_NE(main_func, nullptr);

  // `CallOp` operand 0: donated; operand 1: aliased; operand 2: plain;
  // operand 3: control input.
  CallOp call_op = *main_func.getOps<CallOp>().begin();
  EXPECT_TRUE(IsDonatedOperand(call_op->getOpOperand(0)));
  EXPECT_FALSE(IsAliasedOperand(call_op->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedOperand(call_op->getOpOperand(0)));

  EXPECT_FALSE(IsDonatedOperand(call_op->getOpOperand(1)));
  EXPECT_TRUE(IsAliasedOperand(call_op->getOpOperand(1)));
  EXPECT_TRUE(IsDonatedOrAliasedOperand(call_op->getOpOperand(1)));

  EXPECT_FALSE(IsDonatedOperand(call_op->getOpOperand(2)));
  EXPECT_FALSE(IsAliasedOperand(call_op->getOpOperand(2)));
  EXPECT_FALSE(IsDonatedOrAliasedOperand(call_op->getOpOperand(2)));

  EXPECT_FALSE(IsDonatedOperand(call_op->getOpOperand(3)));
  EXPECT_FALSE(IsAliasedOperand(call_op->getOpOperand(3)));

  EXPECT_TRUE(IsDonatedOrAliasedByUser(call_op.getInputs()[0]));
  EXPECT_TRUE(IsDonatedOrAliasedByUser(call_op.getInputs()[1]));
  EXPECT_FALSE(IsDonatedOrAliasedByUser(call_op.getInputs()[2]));

  // `CallLoadedExecutableOp` operand 0: donated; operand 1: aliased;
  // operand 2: plain; operand 3: control input.
  CallLoadedExecutableOp call_exec_op =
      *main_func.getOps<CallLoadedExecutableOp>().begin();
  EXPECT_TRUE(IsDonatedOperand(call_exec_op->getOpOperand(0)));
  EXPECT_FALSE(IsAliasedOperand(call_exec_op->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedOperand(call_exec_op->getOpOperand(0)));

  EXPECT_FALSE(IsDonatedOperand(call_exec_op->getOpOperand(1)));
  EXPECT_TRUE(IsAliasedOperand(call_exec_op->getOpOperand(1)));
  EXPECT_TRUE(IsDonatedOrAliasedOperand(call_exec_op->getOpOperand(1)));

  EXPECT_FALSE(IsDonatedOperand(call_exec_op->getOpOperand(2)));
  EXPECT_FALSE(IsAliasedOperand(call_exec_op->getOpOperand(2)));
  EXPECT_FALSE(IsDonatedOrAliasedOperand(call_exec_op->getOpOperand(2)));

  EXPECT_FALSE(IsDonatedOperand(call_exec_op->getOpOperand(3)));
  EXPECT_FALSE(IsAliasedOperand(call_exec_op->getOpOperand(3)));

  EXPECT_TRUE(IsDonatedOrAliasedByUser(call_exec_op.getInputs()[0]));
  EXPECT_TRUE(IsDonatedOrAliasedByUser(call_exec_op.getInputs()[1]));
  EXPECT_FALSE(IsDonatedOrAliasedByUser(call_exec_op.getInputs()[2]));

  // `CopyArraysOp`: plain vs. `reuse = true` (aliased) vs. `donated = true`.
  llvm::SmallVector<CopyArraysOp> copy_ops(main_func.getOps<CopyArraysOp>());
  ASSERT_EQ(copy_ops.size(), 3u);
  EXPECT_FALSE(IsDonatedOperand(copy_ops[0]->getOpOperand(0)));
  EXPECT_FALSE(IsAliasedOperand(copy_ops[0]->getOpOperand(0)));
  EXPECT_FALSE(IsDonatedOrAliasedByUser(copy_ops[0].getInputs()[0]));
  EXPECT_FALSE(IsDonatedOperand(copy_ops[1]->getOpOperand(0)));
  EXPECT_TRUE(IsAliasedOperand(copy_ops[1]->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedByUser(copy_ops[1].getInputs()[0]));
  EXPECT_TRUE(IsDonatedOperand(copy_ops[2]->getOpOperand(0)));
  EXPECT_FALSE(IsAliasedOperand(copy_ops[2]->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedByUser(copy_ops[2].getInputs()[0]));

  // `ReshardOp`: plain vs. `donated = true`.
  llvm::SmallVector<ReshardOp> reshard_ops(main_func.getOps<ReshardOp>());
  ASSERT_EQ(reshard_ops.size(), 2u);
  EXPECT_FALSE(IsDonatedOperand(reshard_ops[0]->getOpOperand(0)));
  EXPECT_FALSE(IsAliasedOperand(reshard_ops[0]->getOpOperand(0)));
  EXPECT_FALSE(IsDonatedOrAliasedByUser(reshard_ops[0].getInputs()[0]));
  EXPECT_TRUE(IsDonatedOperand(reshard_ops[1]->getOpOperand(0)));
  EXPECT_FALSE(IsAliasedOperand(reshard_ops[1]->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedByUser(reshard_ops[1].getInputs()[0]));

  // `BitcastArraysOp`: `donated = false` (aliased) vs. `donated = true`
  // (donated).
  llvm::SmallVector<BitcastArraysOp> bitcasts(
      main_func.getOps<BitcastArraysOp>());
  ASSERT_EQ(bitcasts.size(), 2u);
  EXPECT_FALSE(IsDonatedOperand(bitcasts[0]->getOpOperand(0)));
  EXPECT_TRUE(IsAliasedOperand(bitcasts[0]->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedByUser(bitcasts[0].getInputs()[0]));
  EXPECT_TRUE(IsDonatedOperand(bitcasts[1]->getOpOperand(0)));
  EXPECT_FALSE(IsAliasedOperand(bitcasts[1]->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedByUser(bitcasts[1].getInputs()[0]));

  // `RemapArraysOp`: `donated = false` (aliased) vs. `donated = true`
  // (donated), plus control input operand.
  llvm::SmallVector<RemapArraysOp> remap_ops(main_func.getOps<RemapArraysOp>());
  ASSERT_EQ(remap_ops.size(), 2u);
  EXPECT_FALSE(IsDonatedOperand(remap_ops[0]->getOpOperand(0)));
  EXPECT_TRUE(IsAliasedOperand(remap_ops[0]->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedOperand(remap_ops[0]->getOpOperand(0)));
  EXPECT_FALSE(IsDonatedOperand(remap_ops[0]->getOpOperand(1)));
  EXPECT_FALSE(IsAliasedOperand(remap_ops[0]->getOpOperand(1)));
  EXPECT_TRUE(IsDonatedOrAliasedByUser(remap_ops[0].getInputs()[0]));

  EXPECT_TRUE(IsDonatedOperand(remap_ops[1]->getOpOperand(0)));
  EXPECT_FALSE(IsAliasedOperand(remap_ops[1]->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedOperand(remap_ops[1]->getOpOperand(0)));
  EXPECT_FALSE(IsDonatedOperand(remap_ops[1]->getOpOperand(1)));
  EXPECT_FALSE(IsAliasedOperand(remap_ops[1]->getOpOperand(1)));
  EXPECT_TRUE(IsDonatedOrAliasedByUser(remap_ops[1].getInputs()[0]));

  // `AssembleOp`: `donated = false` (aliased) vs. `donated = true` (donated),
  // plus control input operand.
  llvm::SmallVector<AssembleOp> assemble_ops(main_func.getOps<AssembleOp>());
  ASSERT_EQ(assemble_ops.size(), 2u);
  EXPECT_FALSE(IsDonatedOperand(assemble_ops[0]->getOpOperand(0)));
  EXPECT_TRUE(IsAliasedOperand(assemble_ops[0]->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedOperand(assemble_ops[0]->getOpOperand(0)));
  EXPECT_FALSE(IsDonatedOperand(assemble_ops[0]->getOpOperand(2)));
  EXPECT_FALSE(IsAliasedOperand(assemble_ops[0]->getOpOperand(2)));
  EXPECT_TRUE(IsDonatedOrAliasedByUser(assemble_ops[0].getInputs()[0]));

  EXPECT_TRUE(IsDonatedOperand(assemble_ops[1]->getOpOperand(0)));
  EXPECT_FALSE(IsAliasedOperand(assemble_ops[1]->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedOperand(assemble_ops[1]->getOpOperand(0)));
  EXPECT_FALSE(IsDonatedOperand(assemble_ops[1]->getOpOperand(2)));
  EXPECT_FALSE(IsAliasedOperand(assemble_ops[1]->getOpOperand(2)));
  EXPECT_TRUE(IsDonatedOrAliasedByUser(assemble_ops[1].getInputs()[0]));

  // `DisassembleOp`: `donated = false` (aliased) vs. `donated = true`
  // (donated), plus control input operand.
  llvm::SmallVector<DisassembleOp> disassemble_ops(
      main_func.getOps<DisassembleOp>());
  ASSERT_EQ(disassemble_ops.size(), 2u);
  EXPECT_FALSE(IsDonatedOperand(disassemble_ops[0]->getOpOperand(0)));
  EXPECT_TRUE(IsAliasedOperand(disassemble_ops[0]->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedOperand(disassemble_ops[0]->getOpOperand(0)));
  EXPECT_FALSE(IsDonatedOperand(disassemble_ops[0]->getOpOperand(1)));
  EXPECT_FALSE(IsAliasedOperand(disassemble_ops[0]->getOpOperand(1)));

  EXPECT_TRUE(IsDonatedOperand(disassemble_ops[1]->getOpOperand(0)));
  EXPECT_FALSE(IsAliasedOperand(disassemble_ops[1]->getOpOperand(0)));
  EXPECT_TRUE(IsDonatedOrAliasedOperand(disassemble_ops[1]->getOpOperand(0)));
  EXPECT_FALSE(IsDonatedOperand(disassemble_ops[1]->getOpOperand(1)));
  EXPECT_FALSE(IsAliasedOperand(disassemble_ops[1]->getOpOperand(1)));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
