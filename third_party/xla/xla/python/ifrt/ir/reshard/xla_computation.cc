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

#include "xla/python/ifrt/ir/reshard/xla_computation.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/memory.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace reshard {

namespace {

// Find donatable inputs from non-tuple parameter i to output tuple element i by
// checking their per-buffer sizes. In principle, we should be able to donate
// all inputs, but it triggers some runtime's limitations of implementing a
// buffer copy using aliasing, which will not be invalid when the copied buffer
// is donated.
std::vector<int64_t> FindDonatableInputs(
    const xla::Shape& old_xla_shape, const xla::HloSharding& old_hlo_sharding,
    const xla::Shape& new_xla_shape, const xla::HloSharding& new_hlo_sharding) {
  CHECK(old_xla_shape.IsTuple());
  CHECK(new_xla_shape.IsTuple());
  CHECK(old_hlo_sharding.IsTuple());
  CHECK(new_hlo_sharding.IsTuple());
  CHECK_EQ(old_xla_shape.tuple_shapes().size(),
           new_xla_shape.tuple_shapes().size());
  CHECK_EQ(old_hlo_sharding.tuple_elements().size(),
           old_xla_shape.tuple_shapes().size());
  CHECK_EQ(new_hlo_sharding.tuple_elements().size(),
           new_xla_shape.tuple_shapes().size());

  const int64_t num_arrays = old_xla_shape.tuple_shapes().size();
  std::vector<int64_t> donatable_input_indices;
  for (int64_t idx = 0; idx < num_arrays; ++idx) {
    xla::Shape old_buffer_shape =
        old_hlo_sharding.tuple_elements()[idx].TileShape(
            old_xla_shape.tuple_shapes(idx));
    int64_t old_buffer_size = xla::ShapeUtil::ByteSizeOf(old_buffer_shape);
    xla::Shape new_buffer_shape =
        new_hlo_sharding.tuple_elements()[idx].TileShape(
            new_xla_shape.tuple_shapes(idx));
    int64_t new_buffer_size = xla::ShapeUtil::ByteSizeOf(new_buffer_shape);
    if (old_buffer_size == new_buffer_size) {
      donatable_input_indices.push_back(idx);
    }
  }
  return donatable_input_indices;
}

// Converts an XLA computation to an `xla::ifrt::HloProgram`, and applies input
// donation and memory kind attributes to the input and output. The generated
// MLIR module will have flattened (non XLA tuple) parameters and results.
absl::StatusOr<std::unique_ptr<xla::ifrt::HloProgram>> ConvertToHloProgram(
    const xla::XlaComputation& xla_computation,
    absl::Span<const int64_t> donated_input_indices,
    absl::Span<const xla::ifrt::MemoryKind> arg_memory_kinds,
    absl::Span<const xla::ifrt::MemoryKind> result_memory_kinds) {
  const xla::HloModuleProto& hlo_module_proto = xla_computation.proto();
  TF_ASSIGN_OR_RETURN(
      auto host_program_shape,
      xla::ProgramShape::FromProto(hlo_module_proto.host_program_shape()));
  const xla::HloModuleConfig hlo_module_config(host_program_shape,
                                               /*ignore_layouts=*/false);
  TF_ASSIGN_OR_RETURN(
      const auto hlo_module,
      xla::HloModule::CreateFromProto(hlo_module_proto, hlo_module_config));

  auto mlir_context = std::make_unique<mlir::MLIRContext>();
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      xla::ConvertHloToStablehlo(*mlir_context, hlo_module.get()));
  auto program = std::make_unique<xla::ifrt::HloProgram>(
      std::move(mlir_context), std::move(mlir_module));

  mlir::func::FuncOp main =
      program->mlir_module().lookupSymbol<mlir::func::FuncOp>("main");
  if (main == nullptr) {
    return absl::InvalidArgumentError("module has no 'main' function");
  }
  // TODO(b/390732473): We should always donate inputs. If an input cannot be
  // donated at runtime, the execution can use `non_donatable_input_indices` to
  // create a copy of device buffers on the fly. However, several runtimes
  // ignore `non_donatable_input_indices`, so we need to be conservative at
  // input donation.
  for (int64_t idx : donated_input_indices) {
    main.setArgAttr(
        idx, "jax.buffer_donor",
        mlir::BoolAttr::get(program->mlir_module()->getContext(), true));
  }
  for (int64_t idx = 0; idx < arg_memory_kinds.size(); ++idx) {
    if (arg_memory_kinds[idx].memory_kind().has_value()) {
      main.setArgAttr(
          idx, "mhlo.memory_kind",
          mlir::StringAttr::get(program->mlir_module()->getContext(),
                                *arg_memory_kinds[idx].memory_kind()));
    }
  }
  for (int64_t idx = 0; idx < result_memory_kinds.size(); ++idx) {
    if (result_memory_kinds[idx].memory_kind().has_value()) {
      main.setResultAttr(
          idx, "mhlo.memory_kind",
          mlir::StringAttr::get(program->mlir_module()->getContext(),
                                *result_memory_kinds[idx].memory_kind()));
    }
  }
  return program;
}

// Specifies where the computation should be executed based on the memory kind.
// This should be used only for computational XLA operations. If it is applied
// incorrectly, it can make some XLA optimization passes (e.g., SPMD
// propagation) to behave in an unexpected way (e.g., not finishing resharding
// work by the time when the execution is considered complete).
absl::Status SetComputeType(xla::XlaBuilder& builder, xla::XlaOp op,
                            const xla::ifrt::MemoryKind& memory_kind) {
  if (memory_kind.memory_kind().has_value()) {
    if (*memory_kind.memory_kind() == "device") {
      // No need to set the attribute.
    } else if (*memory_kind.memory_kind() == "pinned_host") {
      TF_RETURN_IF_ERROR(builder.SetInstructionFrontendAttribute(
          op, "_xla_compute_type", "host"));
    } else {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported memory kind: ", memory_kind));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<xla::ifrt::HloProgram>>
XlaComputationBuilder::BuildXlaReshardComputation(
    const xla::Shape& xla_shape, const xla::HloSharding& old_hlo_sharding,
    const xla::HloSharding& new_hlo_sharding,
    absl::Span<const xla::ifrt::MemoryKind> memory_kinds, bool pre_resharding) {
  TF_RET_CHECK(xla_shape.IsTuple());
  TF_RET_CHECK(old_hlo_sharding.IsTuple());
  TF_RET_CHECK(new_hlo_sharding.IsTuple());
  TF_RET_CHECK(old_hlo_sharding.tuple_elements().size() ==
               xla_shape.tuple_shapes().size());
  TF_RET_CHECK(new_hlo_sharding.tuple_elements().size() ==
               xla_shape.tuple_shapes().size());
  const int64_t num_arrays = xla_shape.tuple_shapes().size();

  xla::XlaBuilder builder(pre_resharding ? "pre_reshard" : "post_reshard");
  std::vector<xla::XlaOp> params;
  params.reserve(num_arrays);
  for (int64_t idx = 0; idx < num_arrays; ++idx) {
    xla::XlaOp param;
    {
      xla::XlaScopedShardingAssignment sa(
          &builder, old_hlo_sharding.tuple_elements()[idx].ToProto());
      param = xla::Parameter(&builder, idx, xla_shape.tuple_shapes(idx),
                             absl::StrCat("parameter_", idx));
    }
    // TODO(b/391701945): Remove this custom call. In principle, we should not
    // need this extra annotation because we already set `mhlo.memory_kind`
    // attribute for the output in `ConvertToHloProgram()`.
    if (memory_kinds[idx].memory_kind().has_value() && device_kind_ != "cpu") {
      param = xla::CustomCall(&builder, "annotate_device_placement", {param},
                              xla_shape.tuple_shapes(idx),
                              /*opaque=*/"",
                              /*has_side_effect=*/true);
      TF_RETURN_IF_ERROR(builder.SetInstructionFrontendAttribute(
          param, "_xla_buffer_placement",
          std::string(*memory_kinds[idx].memory_kind())));
    }
    params.push_back(param);
  }
  xla::XlaOp root;
  {
    xla::XlaScopedShardingAssignment sa(&builder, new_hlo_sharding.ToProto());
    root = xla::Tuple(&builder, params);
  }
  TF_ASSIGN_OR_RETURN(auto hlo, builder.Build(root));

  TF_ASSIGN_OR_RETURN(auto program_shape, hlo.GetProgramShape());
  TF_RET_CHECK(program_shape.result() == xla_shape);

  std::vector<int64_t> donatable_input_indices = FindDonatableInputs(
      xla_shape, old_hlo_sharding, xla_shape, new_hlo_sharding);
  return ConvertToHloProgram(hlo, donatable_input_indices,
                             /*arg_memory_kinds=*/memory_kinds,
                             /*result_memory_kinds=*/memory_kinds);
}

absl::StatusOr<std::unique_ptr<xla::ifrt::HloProgram>>
XlaComputationBuilder::BuildXlaZerosComputation(
    const xla::Shape& xla_shape, const xla::HloSharding& new_hlo_sharding,
    absl::Span<const xla::ifrt::MemoryKind> memory_kinds) {
  TF_RET_CHECK(xla_shape.IsTuple());
  TF_RET_CHECK(new_hlo_sharding.IsTuple());
  TF_RET_CHECK(new_hlo_sharding.tuple_elements().size() ==
               xla_shape.tuple_shapes().size());
  TF_RET_CHECK(memory_kinds.size() == xla_shape.tuple_shapes().size());
  const int64_t num_arrays = xla_shape.tuple_shapes().size();
  std::vector<xla::Shape> new_buffer_shapes;
  new_buffer_shapes.reserve(num_arrays);
  for (int64_t idx = 0; idx < num_arrays; ++idx) {
    new_buffer_shapes.push_back(
        new_hlo_sharding.tuple_elements()[idx].TileShape(
            xla_shape.tuple_shapes(idx)));
  }

  xla::XlaBuilder builder("zeros");
  std::vector<xla::XlaOp> elems;
  elems.reserve(num_arrays);
  for (int64_t idx = 0; idx < num_arrays; ++idx) {
    xla::XlaOp zero =
        xla::Zero(&builder, new_buffer_shapes[idx].element_type());
    xla::XlaOp zeros =
        xla::Broadcast(zero, new_buffer_shapes[idx].dimensions());
    if (device_kind_ != "cpu") {
      TF_RETURN_IF_ERROR(SetComputeType(builder, zeros, memory_kinds[idx]));
      // TODO(b/391701945): Remove this custom call. In principle, we should not
      // need this extra annotation because we already set `mhlo.memory_kind`
      // attribute for the output in `ConvertToHloProgram()`.
      if (memory_kinds[idx].memory_kind().has_value()) {
        zeros = xla::CustomCall(&builder, "annotate_device_placement", {zeros},
                                new_buffer_shapes[idx],
                                /*opaque=*/"",
                                /*has_side_effect=*/true);
        TF_RETURN_IF_ERROR(builder.SetInstructionFrontendAttribute(
            zeros, "_xla_buffer_placement",
            std::string(*memory_kinds[idx].memory_kind())));
      }
    }
    elems.push_back(zeros);
  }
  xla::XlaOp root = xla::Tuple(&builder, elems);
  TF_ASSIGN_OR_RETURN(auto hlo, builder.Build(root));

  TF_ASSIGN_OR_RETURN(auto program_shape, hlo.GetProgramShape());
  TF_RET_CHECK(program_shape.result().IsTuple());
  TF_RET_CHECK(program_shape.result().tuple_shapes().size() == num_arrays);
  for (int64_t idx = 0; idx < num_arrays; ++idx) {
    TF_RET_CHECK(program_shape.result().tuple_shapes(idx) ==
                 new_buffer_shapes[idx]);
  }

  return ConvertToHloProgram(hlo,
                             /*donated_input_indices=*/{},
                             /*arg_memory_kinds=*/{},
                             /*result_memory_kinds=*/memory_kinds);
}

absl::StatusOr<std::unique_ptr<xla::ifrt::HloProgram>>
XlaComputationBuilder::BuildXlaReduceComputation(
    const xla::Shape& old_xla_shape, const xla::HloSharding& old_hlo_sharding,
    const xla::Shape& new_xla_shape, const xla::HloSharding& new_hlo_sharding,
    absl::Span<const xla::ifrt::MemoryKind> memory_kinds) {
  TF_RET_CHECK(old_xla_shape.IsTuple());
  TF_RET_CHECK(new_xla_shape.IsTuple());
  TF_RET_CHECK(old_xla_shape.tuple_shapes().size() ==
               new_xla_shape.tuple_shapes().size());
  TF_RET_CHECK(old_hlo_sharding.IsTuple());
  TF_RET_CHECK(new_hlo_sharding.IsTuple());
  TF_RET_CHECK(old_hlo_sharding.tuple_elements().size() ==
               old_xla_shape.tuple_shapes().size());
  TF_RET_CHECK(new_hlo_sharding.tuple_elements().size() ==
               new_xla_shape.tuple_shapes().size());
  const int64_t num_arrays = old_xla_shape.tuple_shapes().size();

  xla::XlaBuilder builder("reduce");
  std::vector<xla::XlaOp> params;
  params.reserve(num_arrays);
  for (int64_t idx = 0; idx < num_arrays; ++idx) {
    xla::XlaScopedShardingAssignment sa(
        &builder, old_hlo_sharding.tuple_elements()[idx].ToProto());
    params.push_back(xla::Parameter(&builder, idx,
                                    old_xla_shape.tuple_shapes(idx),
                                    absl::StrCat("parameter_", idx)));
  }
  std::vector<xla::XlaOp> elems;
  elems.reserve(num_arrays);
  for (int64_t idx = 0; idx < num_arrays; ++idx) {
    // Reshape to ensure that the first dimension is the dimension to reduce.
    std::vector<int64_t> reshape_dims;
    reshape_dims.reserve(1 +
                         new_xla_shape.tuple_shapes(idx).dimensions().size());
    int64_t num_replicas;
    if (old_xla_shape.tuple_shapes(idx).dimensions().size() ==
        new_xla_shape.tuple_shapes(idx).dimensions().size() + 1) {
      // Originally scalar; the first dimension is solely the number of
      // replicas.
      num_replicas = old_xla_shape.tuple_shapes(idx).dimensions(0);
    } else {
      // Originally non-scalar; the first dimension was multiplied by the number
      // of replicas.
      num_replicas = old_xla_shape.tuple_shapes(idx).dimensions(0) /
                     new_xla_shape.tuple_shapes(idx).dimensions(0);
    }
    reshape_dims.push_back(num_replicas);
    absl::c_copy(new_xla_shape.tuple_shapes(idx).dimensions(),
                 std::back_inserter(reshape_dims));
    xla::Shape reshaped_shape = xla::ShapeUtil::MakeShape(
        old_xla_shape.tuple_shapes(idx).element_type(), reshape_dims);
    xla::LayoutUtil::SetToDefaultLayout(&reshaped_shape);
    xla::XlaOp reshaped = xla::Reshape(reshaped_shape, params[idx]);
    if (device_kind_ != "cpu") {
      TF_RETURN_IF_ERROR(SetComputeType(builder, reshaped, memory_kinds[idx]));
    }

    // Reduce-sum over the first dimension.
    xla::XlaOp reduced = xla::Reduce(
        reshaped, xla::Zero(&builder, reshaped_shape.element_type()),
        xla::CreateScalarAddComputation(reshaped_shape.element_type(),
                                        &builder),
        {0});
    if (device_kind_ != "cpu") {
      TF_RETURN_IF_ERROR(SetComputeType(builder, reduced, memory_kinds[idx]));
      // TODO(b/391701945): Remove this custom call. In principle, we should not
      // need this extra annotation because we already set `mhlo.memory_kind`
      // attribute for the output in `ConvertToHloProgram()`.
      if (memory_kinds[idx].memory_kind().has_value()) {
        reduced = xla::CustomCall(&builder, "annotate_device_placement",
                                  {reduced}, new_xla_shape.tuple_shapes(idx),
                                  /*opaque=*/"",
                                  /*has_side_effect=*/true);
        TF_RETURN_IF_ERROR(builder.SetInstructionFrontendAttribute(
            reduced, "_xla_buffer_placement",
            std::string(*memory_kinds[idx].memory_kind())));
      }
    }
    elems.push_back(reduced);
  }
  xla::XlaOp root;
  {
    xla::XlaScopedShardingAssignment sa(&builder, new_hlo_sharding.ToProto());
    root = xla::Tuple(&builder, elems);
  }
  TF_ASSIGN_OR_RETURN(auto hlo, builder.Build(root));

  TF_ASSIGN_OR_RETURN(auto program_shape, hlo.GetProgramShape());
  TF_RET_CHECK(program_shape.result() == new_xla_shape);

  std::vector<int64_t> donatable_input_indices = FindDonatableInputs(
      old_xla_shape, old_hlo_sharding, new_xla_shape, new_hlo_sharding);
  return ConvertToHloProgram(hlo, donatable_input_indices,
                             /*arg_memory_kinds=*/memory_kinds,
                             /*result_memory_kinds=*/memory_kinds);
}

std::string DumpHloProgram(xla::ifrt::HloProgram& program) {
  std::string s;
  llvm::raw_string_ostream os(s);
  program.mlir_module().print(os);
  return s;
}

}  // namespace reshard
}  // namespace ifrt
}  // namespace xla
