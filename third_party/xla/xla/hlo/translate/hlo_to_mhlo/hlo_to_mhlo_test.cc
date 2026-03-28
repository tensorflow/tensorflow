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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class HloToMhloTest : public HloTestBase {
 protected:
  HloToMhloTest() : HloTestBase() {}
};

TEST_F(HloToMhloTest, ConvertsMeshAxesReplicaGroups) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder("entry_computation");

  auto shape = ShapeUtil::MakeShape(F32, {128});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));

  // Create a mesh
  std::vector<int64_t> axes_sizes = {2, 2};
  std::vector<std::string> axes_names_str = {"a", "b"};
  std::vector<absl::string_view> axes_names(axes_names_str.begin(),
                                            axes_names_str.end());
  // Drop device_ids to use iota constructor
  Mesh mesh(axes_sizes, axes_names);

  // Create AxisRefs
  std::vector<AxisRef> axis_refs;
  axis_refs.push_back(AxisRef(0));
  axis_refs.push_back(AxisRef(1));

  // Create MeshAxesReplicaGroupList
  auto replica_groups =
      std::make_shared<MeshAxesReplicaGroupList>(mesh, axis_refs);

  // Create reduction computation
  auto scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto reduction_builder = HloComputation::Builder("reduction_computation");
  auto x = reduction_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "x"));
  auto y = reduction_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "y"));
  reduction_builder.AddInstruction(
      HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, x, y));

  // Create AllReduce instruction with the mesh axes replica groups
  auto* reduction_computation =
      module->AddEmbeddedComputation(reduction_builder.Build());
  builder.AddInstruction(HloInstruction::CreateAllReduce(
      shape, {param}, reduction_computation, replica_groups,
      /*constrain_layout=*/false,
      /*channel_id=*/std::nullopt, /*use_global_device_ids=*/false));

  module->AddEntryComputation(builder.Build());

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(auto mlir_module,
                          ConvertHloToMlirHlo(context, module.get()));

  // Verify that the generated MLIR module contains the expected attributes
  std::string mlir_output;
  llvm::raw_string_ostream os(mlir_output);
  mlir_module->print(os);

  // For now, we expect this to NOT fail, but checks might need adjustment
  // depending on current behavior. The goal is to see sdy.mesh and
  // replica_group_mesh_axes.
  EXPECT_PRED2([](absl::string_view s,
                  absl::string_view sub) { return absl::StrContains(s, sub); },
               mlir_output, "sdy.mesh");
  EXPECT_PRED2([](absl::string_view s,
                  absl::string_view sub) { return absl::StrContains(s, sub); },
               mlir_output, "#mhlo.replica_group_mesh_axes<mesh = @mesh");
}

TEST_F(HloToMhloTest, ConvertsMeshAxesReplicaGroupsToStableHlo) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder("entry_computation");

  auto shape = ShapeUtil::MakeShape(F32, {128});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));

  // Create a mesh
  std::vector<int64_t> axes_sizes = {2, 2};
  std::vector<std::string> axes_names_str = {"a", "b"};
  std::vector<absl::string_view> axes_names(axes_names_str.begin(),
                                            axes_names_str.end());
  Mesh mesh(axes_sizes, axes_names);

  // Create AxisRefs
  std::vector<AxisRef> axis_refs;
  axis_refs.push_back(AxisRef(0));

  // Create MeshAxesReplicaGroupList
  auto replica_groups =
      std::make_shared<MeshAxesReplicaGroupList>(mesh, axis_refs);

  // Create reduction computation
  auto scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto reduction_builder = HloComputation::Builder("reduction_computation");
  auto x = reduction_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "x"));
  auto y = reduction_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "y"));
  reduction_builder.AddInstruction(
      HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, x, y));

  // Create AllReduce instruction with the mesh axes replica groups
  auto* reduction_computation =
      module->AddEmbeddedComputation(reduction_builder.Build());
  builder.AddInstruction(HloInstruction::CreateAllReduce(
      shape, {param}, reduction_computation, replica_groups,
      /*constrain_layout=*/false,
      /*channel_id=*/std::nullopt, /*use_global_device_ids=*/false));

  module->AddEntryComputation(builder.Build());

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      auto mlir_module,
      ConvertHloToMlirHlo(context, module.get(),
                          /*import_all_computations=*/false,
                          /*flatten_computation_args_result=*/false,
                          /*emit_stablehlo=*/true));

  // Verify that the generated MLIR module contains the expected attributes
  std::string mlir_output;
  llvm::raw_string_ostream os(mlir_output);
  mlir_module->print(os);

  EXPECT_PRED2(
      [](absl::string_view s, absl::string_view sub) {
        return s.find(sub) != std::string::npos;
      },
      mlir_output, "sdy.mesh");
  EXPECT_PRED2(
      [](absl::string_view s, absl::string_view sub) {
        return s.find(sub) != std::string::npos;
      },
      mlir_output, "#stablehlo.replica_group_mesh_axes<mesh = @mesh");
}

}  // namespace
}  // namespace xla
