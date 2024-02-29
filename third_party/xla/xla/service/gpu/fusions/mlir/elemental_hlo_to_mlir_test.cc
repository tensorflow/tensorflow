/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/status_macros.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace mlir_converter {
namespace {

class ElementalHloToMlirTest : public HloTestBase {
 public:
  ElementalHloToMlirTest() {
    context_.loadDialect<mlir::tensor::TensorDialect, mlir::func::FuncDialect,
                         mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                         mlir::math::MathDialect, mlir::scf::SCFDialect,
                         mlir::mhlo::MhloDialect, mlir::LLVM::LLVMDialect,
                         xla::gpu::XlaGpuDialect>();
  }

  // Converts the root subgraph of the entry function of the given hlo module to
  // MLIR.
  absl::Status Run(
      const std::string& hlo, const std::string& filecheck_str,
      std::function<bool(const HloInstruction*)> is_subgraph_root = nullptr,
      std::function<bool(const HloInstruction*, int)>
          operand_is_function_argument = nullptr) {
    auto hlo_module = ParseAndReturnVerifiedModule(hlo).value();

    mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(&context_),
                                       &context_);
    auto module = llvm_ir::CreateMlirModuleOp(builder.getLoc());
    builder.setInsertionPointToStart(module->getBody());
    auto* entry_computation = hlo_module->entry_computation();
    PartitionedComputations partitioned_computations(
        entry_computation, is_subgraph_root, operand_is_function_argument);
    auto fns = partitioned_computations.DeclareFunctions(module.get());
    auto entry_func = fns[&partitioned_computations
                               .FindPartitionedComputation(entry_computation)
                               .GetRootSubgraph()];
    auto& entry_pc =
        partitioned_computations.FindPartitionedComputation(entry_computation);
    TF_RETURN_IF_ERROR(SubgraphToMlirFunction(
        entry_pc, entry_pc.GetRootSubgraph(), entry_func,
        partitioned_computations.CreateCallTargetProvider(fns)));

    // Canonicalize and CSE for better readability of check tests.
    mlir::PassManager pm(&context_);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    TF_RET_CHECK(pm.run(module.get()).succeeded());

    std::string out;
    llvm::raw_string_ostream stream(out);
    stream << entry_func;

    TF_ASSIGN_OR_RETURN(auto filecheck_result,
                        RunFileCheck(out, filecheck_str));
    TF_RET_CHECK(filecheck_result);
    return absl::OkStatus();
  }

  mlir::MLIRContext context_;
};

TEST_F(ElementalHloToMlirTest, Reduce) {
  TF_EXPECT_OK(Run(R"(
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT sum = f32[] add(p0, p1)
    }

    ENTRY main {
      p0 = f32[10,20,30,40] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[10,30] reduce(p0, p1), dimensions={1,3},
                                          to_apply=add
    })",
                   R"(
    // CHECK:      @main_r(
    // CHECK-SAME:   %[[ARG0:.*]]: tensor<10x20x30x40xf32>
    // CHECK-SAME:   %[[ARG1:.*]]: tensor<f32>
    // CHECK-SAME:   %[[X:.*]]: index {{.*}}, %[[Y:.*]]: index {{.*}} -> f32
    // CHECK-DAG:  %[[C0:.*]] = arith.constant 0
    // CHECK-DAG:  %[[C1:.*]] = arith.constant 1
    // CHECK-DAG:  %[[C20:.*]] = arith.constant 20
    // CHECK-DAG:  %[[C40:.*]] = arith.constant 40
    // CHECK:      %[[INIT:.*]] = tensor.extract %[[ARG1]][]
    // CHECK:      %[[RET:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C20]]
    // CHECK-SAME:   step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT]])
    // CHECK:        %[[RET_INNER:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C40]]
    // CHECK-SAME:     iter_args(%[[ACC_INNER:.*]] = %[[ACC]])
    // CHECK:          %[[VAL:.*]] = tensor.extract %[[ARG0]]
    // CHECK-SAME:        [%[[X]], %[[I]], %[[Y]], %[[J]]]
    // CHECK:          %[[UPD:.*]] = func.call @add_sum(%[[ACC_INNER]],
    // CHECK-SAME:                                      %[[VAL]])
    // CHECK:          scf.yield %[[UPD]]
    // CHECK:        }
    // CHECK:        scf.yield %[[RET_INNER]]
    // CHECK:      }
    // CHECK:      return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, Concatenate) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = f32[10,20,30] parameter(0)
      p1 = f32[10,15,30] parameter(1)
      p2 = f32[10,3,30] parameter(2)
      ROOT r = f32[10,38,30] concatenate(p0, p1, p2), dimensions={1}
    })",
                   R"(
    // CHECK:      @main_r(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<10x20x30xf32>,
    // CHECK-SAME:     %[[ARG1:.*]]: tensor<10x15x30xf32>,
    // CHECK-SAME:     %[[ARG2:.*]]: tensor<10x3x30xf32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}},
    // CHECK-SAME:     %[[Z:.*]]: index {{{.*}}}
    // CHECK-DAG:    %[[C35:.*]] = arith.constant 35
    // CHECK-DAG:    %[[C20:.*]] = arith.constant 20
    // CHECK:        %[[IN_BOUNDS:.*]] = arith.cmpi ult, %[[Y]], %[[C20]]
    // CHECK:        %[[CONCAT:.*]] = scf.if %[[IN_BOUNDS]]
    // CHECK:          %[[P0_VAL:.*]] = xla_gpu.pure_call @main_p0
    // CHECK-SAME:         %[[X]], %[[Y]], %[[Z]]
    // CHECK:          scf.yield %[[P0_VAL]]
    // CHECK:        } else {
    // CHECK:          %[[IN_BOUNDS:.*]] = arith.cmpi ult, %[[Y]], %[[C35]]
    // CHECK:          %[[CONCAT2:.*]] = scf.if %[[IN_BOUNDS]]
    // CHECK:            %[[OFFSET:.*]] = arith.subi %[[Y]], %[[C20]]
    // CHECK:            %[[P1_VAL:.*]] = xla_gpu.pure_call @main_p1
    // CHECK-SAME:           %[[X]], %[[OFFSET]], %[[Z]]
    // CHECK:            scf.yield %[[P1_VAL]]
    // CHECK:          } else {
    // CHECK:            %[[OFFSET:.*]] = arith.subi %[[Y]], %[[C35]]
    // CHECK:            %[[P2_VAL:.*]] = xla_gpu.pure_call @main_p2
    // CHECK-SAME:           %[[X]], %[[OFFSET]], %[[Z]]
    // CHECK:            scf.yield %[[P2_VAL]]
    // CHECK:          }
    // CHECK:          scf.yield %[[CONCAT2]]
    // CHECK:        }
    // CHECK:        return %[[CONCAT]]
  )"));
}

TEST_F(ElementalHloToMlirTest, Gather) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      operand = f32[33,34] parameter(0)
      indices = s32[1806,1] parameter(1)
      ROOT r = f32[1806,7,8] gather(operand, indices), offset_dims={1,2},
                                 collapsed_slice_dims={}, start_index_map={0},
                                 index_vector_dim=1, slice_sizes={7,8}
    })",
                   R"(
    // CHECK:      @main_r(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<33x34xf32>,
    // CHECK-SAME:     %[[ARG1:.*]]: tensor<1806x1xi32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}},
    // CHECK-SAME:     %[[Z:.*]]: index {{{.*}}}
    // CHECK-DAG:    %[[C0:.*]] = arith.constant 0
    // CHECK-DAG:    %[[C26:.*]] = arith.constant 26
    // CHECK:        %[[IDX_I32:.*]] = tensor.extract %[[ARG1]][%[[X]], %[[C0]]]
    // CHECK:        %[[IDX:.*]] = arith.index_cast %[[IDX_I32]] : i32 to index
    // CHECK:        %[[CLAMP_HIGH:.*]] = arith.minsi %[[IDX]], %[[C26]]
    // CHECK:        %[[CLAMPED:.*]] = arith.maxsi %[[CLAMP_HIGH]], %[[C0]]
    // CHECK:        %[[X_IN:.*]] = arith.addi %[[CLAMPED]], %[[Y]]
    // CHECK:        %[[RET:.*]] = tensor.extract %[[ARG0]][%[[X_IN]], %[[Z]]]
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, Pad) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = f32[4, 4] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[12, 16] pad(p0, p1), padding=1_4_1x4_8_0
    })",
                   R"(
    // CHECK:      @main_pad(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<4x4xf32>,
    // CHECK-SAME:     %[[ARG1:.*]]: tensor<f32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}}
    // CHECK-DAG:    %[[C0:.*]] = arith.constant 0
    // CHECK-DAG:    %[[C1:.*]] = arith.constant 1
    // CHECK-DAG:    %[[C4:.*]] = arith.constant 4
    // CHECK-DAG:    %[[C7:.*]] = arith.constant 7
    // CHECK:        %[[CONSTRAINT_VAL:.*]] = affine.apply
    // CHECK-SAME:     <()[s0] -> (s0 - ((s0 - 1) floordiv 2) * 2 - 1)>
    // CHECK-SAME:     ()[%[[X]]]
    // CHECK:        %[[CONSTRAINT:.*]] = arith.cmpi eq, %[[CONSTRAINT_VAL]], %[[C0]]
    // CHECK:        %[[X_L:.*]] = arith.cmpi sge, %[[X]], %[[C1]]
    // CHECK:        %[[X_H:.*]] = arith.cmpi sle, %[[X]], %[[C7]]
    // CHECK:        %[[X_BOUNDS:.*]] = arith.andi %[[X_L]], %[[X_H]]
    // CHECK:        %[[X_AND_CONSTRAINT:.*]] = arith.andi %[[CONSTRAINT]], %[[X_BOUNDS]]
    // CHECK:        %[[Y_L:.*]] = arith.cmpi sge, %[[Y]], %[[C4]]
    // CHECK:        %[[Y_H:.*]] = arith.cmpi sle, %[[Y]], %[[C7]]
    // CHECK:        %[[Y_BOUNDS:.*]] = arith.andi %[[Y_L]], %[[Y_H]]
    // CHECK:        %[[FROM_INPUT:.*]] = arith.andi %[[X_AND_CONSTRAINT]], %[[Y_BOUNDS]]
    // CHECK:        %[[RET:.*]] = scf.if %[[FROM_INPUT]]
    // CHECK:          %[[X_IN:.*]] = affine.apply
    // CHECK-SAME:         <()[s0] -> ((s0 - 1) floordiv 2)>()[%[[X]]]
    // CHECK:          %[[Y_IN:.*]] = affine.apply
    // CHECK-SAME:         <()[s0] -> (s0 - 4)>()[%[[Y]]]
    // CHECK:          %[[VAL:.*]] = tensor.extract %[[ARG0]][%[[X_IN]], %[[Y_IN]]]
    // CHECK:          scf.yield %[[VAL]]
    // CHECK:        } else {
    // CHECK:          %[[PAD_VAL:.*]] = tensor.extract %[[ARG1]][]
    // CHECK:          scf.yield %[[PAD_VAL]]
    // CHECK:        }
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, Transpose) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = f32[4,5,6] parameter(0)
      ROOT transpose = f32[6,5,4] transpose(p0), dimensions={2,1,0}
    })",
                   R"(
    // CHECK:      @main_transpose(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<4x5x6xf32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}},
    // CHECK-SAME:     %[[Z:.*]]: index {{{.*}}}
    // CHECK:        %[[RET:.*]] = tensor.extract %[[ARG0]]
    // CHECK-SAME:     [%[[Z]], %[[Y]], %[[X]]]
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, Broadcast) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = f32[4,5] parameter(0)
      ROOT broadcast = f32[6,4,5] broadcast(p0), dimensions={1,2}
    })",
                   R"(
    // CHECK:      @main_broadcast(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<4x5xf32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}},
    // CHECK-SAME:     %[[Z:.*]]: index {{{.*}}}
    // CHECK:        %[[RET:.*]] = tensor.extract %[[ARG0]]
    // CHECK-SAME:     [%[[Y]], %[[Z]]]
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, Add) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      ROOT add = f32[4] add(p0, p1)
    })",
                   R"(
    // CHECK:      @main_add(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xf32>,
    // CHECK-SAME:     %[[X:.*]]: index {{.*}}
    // CHECK:        %[[A:.*]] = tensor.extract %[[ARG0]][%[[X]]]
    // CHECK:        %[[B:.*]] = tensor.extract %[[ARG1]][%[[X]]]
    // CHECK:        %[[RET:.*]] = arith.addf %[[A]], %[[B]]
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, Complex) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      ROOT add = c64[4] complex(p0, p1)
    })",
                   R"(
    // CHECK:      @main_add(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xf32>,
    // CHECK-SAME:     %[[X:.*]]: index {{.*}}
    // CHECK:        %[[A:.*]] = tensor.extract %[[ARG0]][%[[X]]]
    // CHECK:        %[[B:.*]] = tensor.extract %[[ARG1]][%[[X]]]
    // CHECK:        %[[RET:.*]] = complex.create %[[A]], %[[B]]
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, InjectedParameter) {
  TF_EXPECT_OK(Run(
      R"(
      ENTRY main {
        %p0 = f32[2,16,17] parameter(0)
        %log = f32[2,16,17] log(%p0)
        %transpose = f32[2,17,16] transpose(%log), dimensions={0,2,1}
        %p1 = f32[] parameter(1)
        %bc = f32[2,17,16] broadcast(%p1), dimensions={}
        ROOT %add = f32[2,17,16] add(%transpose, %bc)
      })",
      R"(
      // CHECK:      @main_add(
      // CHECK-SAME:     %[[ARG0:.*]]: tensor<2x16x17xf32>
      // CHECK-SAME:     %[[ARG1:.*]]: tensor<f32>
      // CHECK-SAME:     %[[X:.*]]: index {xla.range = [0 : index, 1 :
      // CHECK-SAME:     %[[Y:.*]]: index {xla.range = [0 : index, 16 :
      // CHECK-SAME:     %[[Z:.*]]: index {xla.range = [0 : index, 15 :
      // CHECK-SAME:     %[[TRANSPOSE:.*]]: f32) -> f32
      // CHECK:        %[[B:.*]] = tensor.extract %[[ARG1]][]
      // CHECK:        %[[RET:.*]] = arith.addf %[[TRANSPOSE]], %[[B]]
      // CHECK:        return %[[RET]])",
      [](const HloInstruction* instr) {
        // Make the transpose a new root.
        return instr->opcode() == HloOpcode::kTranspose;
      },
      [](const HloInstruction* instr, int operand_id) {
        // Inject the transpose argument.
        return instr->operand(operand_id)->opcode() == HloOpcode::kTranspose;
      }));
}

TEST_F(ElementalHloToMlirTest, ScalarConstant) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = f32[1,1] parameter(0)
      c1 = f32[1,1] constant({{1.0}})
      ROOT add = f32[1,1] add(p0, c1)
    })",
                   R"(
    // CHECK:      @main_add(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<1x1xf32>
    // CHECK-SAME:     %[[X:.*]]: index {{.*}}, %[[Y:.*]]: index {{.*}}
    // CHECK:        %[[C_1:.*]] = arith.constant 1
    // CHECK:        %[[A:.*]] = tensor.extract %[[ARG0]][%[[X]], %[[Y]]]
    // CHECK:        %[[RET:.*]] = arith.addf %[[A]], %[[C_1]]
    // CHECK:        return %[[RET]]
  })"));
}

}  // namespace
}  // namespace mlir_converter
}  // namespace gpu
}  // namespace xla
