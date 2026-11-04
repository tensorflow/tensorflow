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

#include <gtest/gtest.h>
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
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

}  // namespace
}  // namespace ifrt
}  // namespace xla
