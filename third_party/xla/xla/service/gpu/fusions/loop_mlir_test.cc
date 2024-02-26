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
#include "xla/service/gpu/fusions/loop_mlir.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class MlirLoopFusionTest : public HloTestBase {
 public:
  MlirLoopFusionTest() {
    context_.loadDialect<mlir::tensor::TensorDialect, mlir::func::FuncDialect,
                         mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                         mlir::math::MathDialect, mlir::scf::SCFDialect,
                         mlir::mhlo::MhloDialect, mlir::gpu::GPUDialect>();
    mlir::DialectRegistry registry;
    mlir::func::registerInlinerExtension(registry);
    context_.appendDialectRegistry(registry);
  }

  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  mlir::MLIRContext context_;
};

TEST_F(MlirLoopFusionTest, NoCodeDuplication) {
  // This test HLO is copied from
  // xla/service/fusion_node_indexing_evaluation_test.cc.
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module
%fused_computation (param: f32[6]) -> f32[2] {
  %param = f32[6]{0} parameter(0)
  %slice0.1 = f32[5]{0} slice(f32[6]{0} %param), slice={[0:5]}
  %slice0.2 = f32[5]{0} slice(f32[6]{0} %param), slice={[1:6]}
  %add0 = f32[5]{0} add(f32[5]{0} %slice0.1, f32[5]{0} %slice0.2)
  %slice1.1 = f32[4]{0} slice(f32[5]{0} %add0), slice={[0:4]}
  %slice1.2 = f32[4]{0} slice(f32[5]{0} %add0), slice={[1:5]}
  %add1 = f32[4]{0} add(f32[4]{0} %slice1.1, f32[4]{0} %slice1.2)
  %slice2.1 = f32[3]{0} slice(f32[4]{0} %add1), slice={[0:3]}
  %slice2.2 = f32[3]{0} slice(f32[4]{0} %add1), slice={[1:4]}
  %add2 = f32[3]{0} add(f32[3]{0} %slice2.1, f32[3]{0} %slice2.2)
  %slice3.1 = f32[2]{0} slice(f32[3]{0} %add2), slice={[0:2]}
  %slice3.2 = f32[2]{0} slice(f32[3]{0} %add2), slice={[1:3]}
  ROOT %add3 = f32[2]{0} add(f32[2]{0} %slice3.1, f32[2]{0} %slice3.2)
}

ENTRY entry_computation {
  p0 = f32[] parameter(0)
  add = f32[] add(p0, p0)
  broadcast = f32[6]{0} broadcast(add), dimensions={}
  ROOT %fusion = f32[2]{0} fusion(broadcast), kind=kLoop, calls=%fused_computation
})")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  MlirLoopFusion fusion(analysis);
  TF_ASSERT_OK_AND_ASSIGN(
      auto mlir_module,
      fusion.CreateMLIRModule(context_, *Cast<HloFusionInstruction>(root),
                              "fused_computation", nullptr));

  std::string out;
  llvm::raw_string_ostream os(out);
  mlir_module->print(os);
  ASSERT_TRUE(RunFileCheck(out, R"(
// CHECK-COUNT-4: arith.add
// CHECK-NOT: arith.add
)")
                  .value());
}

TEST_F(MlirLoopFusionTest, TwoUsersConsistentIndexing) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module
%fused_computation (param: f32[6]) -> f32[2] {
  %p0 = f32[2]{0} parameter(0)
  %p1 = f32[2]{0} parameter(1)
  %add = f32[2] add(%p0, %p1)
  %sub = f32[2] subtract(%p0, %p1)
  %mul = f32[2] multiply(%add, %sub)
  %div = f32[2] divide(%add, %sub)
  ROOT %atan2 = f32[2] atan2(%mul, %div)
}

ENTRY entry_computation {
  p0 = f32[2] parameter(0)
  p1 = f32[2] parameter(1)
  ROOT %fusion = f32[2] fusion(p0, p1), kind=kLoop, calls=%fused_computation
})")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  MlirLoopFusion fusion(analysis);
  TF_ASSERT_OK_AND_ASSIGN(
      auto mlir_module,
      fusion.CreateMLIRModule(context_, *Cast<HloFusionInstruction>(root),
                              "fused_computation", nullptr));

  std::string out;
  llvm::raw_string_ostream os(out);
  mlir_module->print(os);
  ASSERT_TRUE(RunFileCheck(out, R"(
    // CHECK: func.func @fused_computation
    // CHECK-NEXT: gpu.thread_id
    // CHECK-NEXT: call @fused_computation_atan2
    // CHECK-NEXT: tensor.insert
    // CHECK-NEXT: return

    // CHECK: func.func @fused_computation_atan2
    // CHECK-NEXT: tensor.extract
    // CHECK-NEXT: tensor.extract
    // CHECK-NEXT: addf
    // CHECK-NEXT: subf
    // CHECK-NEXT: mulf
    // CHECK-NEXT: divf
    // CHECK-NEXT: atan2
    // CHECK-NEXT: return
    )")
                  .value());
}

TEST_F(MlirLoopFusionTest, IotaCopyBitcastBroadcastReshapeReverseTranspose) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module
%fused_computation {
  %iota = f32[10,20,30] iota(), iota_dimension=2
  %copy = f32[10,20,30] copy(%iota)
  %bitcast = s32[10,20,30] bitcast(%copy)
  %broadcast = s32[2,10,3,20,5,30,7] broadcast(%bitcast), dimensions={1,3,5}
  %reshape = s32[20,60,150,7] reshape(%broadcast)
  %reverse = s32[20,60,150,7] reverse(%reshape), dimensions={2,3}
  ROOT %transpose = s32[60,20,7,150] transpose(%reverse), dimensions={1,0,3,2}
}

ENTRY entry_computation {
  ROOT %fusion = s32[60,20,7,150] fusion(), kind=kLoop, calls=%fused_computation
})")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  MlirLoopFusion fusion(analysis);
  TF_ASSERT_OK_AND_ASSIGN(
      auto mlir_module,
      fusion.CreateMLIRModule(context_, *Cast<HloFusionInstruction>(root),
                              "fused_computation", nullptr));

  std::string out;
  llvm::raw_string_ostream os(out);
  mlir_module->print(os);

  ASSERT_TRUE(RunFileCheck(out, R"(
    // CHECK-COUNT-2: func.func
    // CHECK-NOT:     func.func
  )")
                  .value());
}

TEST_F(MlirLoopFusionTest, VariadicReduce) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule Test, is_scheduled=true

Add {
  scalar_lhs.0 = f32[] parameter(0)
  scalar_rhs.0 = f32[] parameter(1)
  scalar_lhs.1 = f32[] parameter(2)
  scalar_rhs.1 = f32[] parameter(3)
  add.0 = f32[] add(scalar_lhs.0, scalar_lhs.1)
  add.1 = f32[] add(scalar_rhs.0, scalar_rhs.1)
  ROOT t = (f32[], f32[]) tuple(add.0, add.1)
}

fused_computation {
  param_0 = f32[5,200,300]{2,1,0} parameter(0)
  param_1 = f32[5,200,300]{2,1,0} parameter(1)
  param_2 = f32[] parameter(2)
  ROOT d.1 = (f32[200]{0}, f32[200]{0}) reduce(f32[5,200,300]{2,1,0} param_0, f32[5,200,300]{2,1,0} %param_1, f32[] param_2, f32[] param_2), dimensions={0,2}, to_apply=Add
}

ENTRY main {
  a = f32[5, 200, 300]{2,1,0} parameter(0)
  b = f32[5, 200, 300]{2,1,0} parameter(1)
  c = f32[] constant(0)
  ROOT fusion = (f32[200]{0}, f32[200]{0}) fusion(f32[5,200,300]{2,1,0} a, f32[5,200,300]{2,1,0} b, f32[] c), kind=kLoop, calls=fused_computation
}
    )")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  MlirLoopFusion fusion(analysis);
  TF_ASSERT_OK_AND_ASSIGN(
      auto mlir_module,
      fusion.CreateMLIRModule(context_, *Cast<HloFusionInstruction>(root),
                              "fused_computation", nullptr));

  std::string out;
  llvm::raw_string_ostream os(out);
  mlir_module->print(os);

  ASSERT_TRUE(RunFileCheck(out, R"(
    // CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> ((s0 + s1 * 128) mod 200)>
    // CHECK: func @fused_computation(
    // CHECK:   %[[TID_X:.*]] = gpu.thread_id x
    // CHECK:   %[[BID_X:.*]] = gpu.block_id x
    // CHECK:   %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[TID_X]], %[[BID_X]]]
    // CHECK:   %[[SCALARS:.*]]:2 = func.call @fused_computation_d_1
    // CHECK:   %[[INSERTED_1:.*]] = tensor.insert %[[SCALARS]]#0 into %{{.*}}[%[[IDX]]]
    // CHECK:   %[[INSERTED_2:.*]] = tensor.insert %[[SCALARS]]#1 into %{{.*}}[%[[IDX]]]
    // CHECK:   yield %[[INSERTED_1]], %[[INSERTED_2]]

    // CHECK: func @fused_computation_d_1
    // CHECK:   %[[RET:.*]]:2 = func.call @Add_t
    // CHECK:   yield %[[RET]]#0, %[[RET]]#1
)")
                  .value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
