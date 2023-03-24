/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/python/custom_call_sharding.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/python/status_casters.h"
#include "tensorflow/compiler/xla/service/custom_call_sharding_helper.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner_util.h"

namespace xla {

namespace py = ::pybind11;

std::vector<Shape> GetArgShapes(const HloInstruction* instruction) {
  std::vector<Shape> result;
  result.reserve(instruction->operand_count());
  for (HloInstruction* operand : instruction->operands()) {
    result.push_back(operand->shape());
  }
  return result;
}

std::vector<std::optional<HloSharding>> GetArgShardings(
    const HloInstruction* instruction) {
  std::vector<std::optional<HloSharding>> result;
  result.reserve(instruction->operand_count());
  for (HloInstruction* operand : instruction->operands()) {
    if (operand->has_sharding()) {
      result.push_back(operand->sharding());
    } else {
      result.push_back(std::nullopt);
    }
  }
  return result;
}

HloInstruction* InlineHloComputation(HloInstruction* instruction,
                                     HloComputation* computation,
                                     HloComputation::Builder* builder,
                                     std::vector<HloInstruction*> operands,
                                     const std::string& suffix) {
  HloCloneContext context(instruction->GetModule(), suffix);

  absl::flat_hash_map<HloInstruction*, HloInstruction*> replacements;
  auto resolve = [&](HloInstruction* inst) {
    auto it = replacements.find(inst);
    if (it == replacements.end()) {
      throw py::key_error(
          absl::StrCat("Could not find mapping for: ", inst->ToString()));
    }
    return it->second;
  };

  for (auto* inst : computation->MakeInstructionPostOrder()) {
    if (inst->opcode() == HloOpcode::kParameter) {
      replacements.emplace(inst, operands[inst->parameter_number()]);
    } else {
      std::vector<HloInstruction*> new_operands;
      new_operands.reserve(inst->operand_count());
      for (HloInstruction* operand : inst->mutable_operands()) {
        new_operands.push_back(resolve(operand));
      }
      replacements.emplace(inst,
                           builder->AddInstruction(inst->CloneWithNewOperands(
                               inst->shape(), new_operands, &context)));
    }
  }
  return resolve(computation->root_instruction());
}

class PyCustomCallPartitioner : public CustomCallPartitioner {
 public:
  PyCustomCallPartitioner(py::object prop_user_sharding, py::object partition,
                          py::object infer_sharding_from_operands,
                          bool can_side_effecting_have_replicated_sharding)
      : prop_user_sharding_(prop_user_sharding),
        partition_(partition),
        infer_sharding_from_operands_(infer_sharding_from_operands),
        can_side_effecting_have_replicated_sharding_(
            can_side_effecting_have_replicated_sharding) {}
  xla::Status Partition(spmd::SpmdPartitioningVisitor* partitioner,
                        HloInstruction* instruction) const override {
    py::gil_scoped_acquire gil;
    auto py_result =
        partition_(GetArgShapes(instruction), GetArgShardings(instruction),
                   instruction->shape(), instruction->sharding(),
                   instruction->raw_backend_config_string());

    const XlaComputation* computation = nullptr;  // Kept alive by py_result.
    std::vector<HloSharding> arg_shardings;
    std::optional<HloSharding> result_sharding;
    try {
      std::tie(computation, arg_shardings, result_sharding) =
          py::cast<std::tuple<const XlaComputation*, std::vector<HloSharding>,
                              HloSharding>>(py_result);
    } catch (const py::cast_error& e) {
      return xla::InternalError(
          "Shardings returned from partitioning %s: expected "
          "Tuple[XlaComputation, List[HloSharding], HloSharding] got: %s",
          instruction->ToString(), py::repr(py_result));
    }
    auto hlo_module_config =
        xla::HloModule::CreateModuleConfigFromProto(
            computation->proto(), xla::DefaultDebugOptionsIgnoringFlags())
            .value();
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                        xla::HloModule::CreateFromProto(computation->proto(),
                                                        hlo_module_config));
    std::vector<HloInstruction*> operands;
    operands.reserve(instruction->operand_count());
    if (arg_shardings.size() != instruction->operand_count()) {
      return xla::InternalError(
          "Shardings returned from partitioning %s must match: %d vs %d",
          instruction->ToString(), arg_shardings.size(),
          instruction->operand_count());
    }
    for (size_t i = 0; i < instruction->operand_count(); ++i) {
      operands.push_back(
          partitioner->GetPartitionedHlo(instruction->mutable_operand(i))
              .Reshard(arg_shardings[i])
              .hlo());
    }

    auto* partitioned_hlo = InlineHloComputation(
        instruction, hlo_module->entry_computation(), partitioner->builder(),
        operands, "_custom_call_lowering_rule");
    partitioned_hlo->set_sharding(result_sharding.value());

    spmd::PartitionedHlo result_partitioned =
        spmd::PartitionedHlo(partitioned_hlo, instruction->shape(),
                             partitioner->MakePartitioningState())
            .Reshard(instruction->sharding());

    partitioner->SetPartitionedHlo(instruction, result_partitioned);
    return xla::OkStatus();
  }
  HloSharding PropagateUserSharding(
      const HloInstruction* instruction, const HloInstruction* user,
      const HloSharding& sharding) const override {
    py::gil_scoped_acquire gil;
    // TODO(parkers): expand this API to handle the `user` sharding.
    // The user is used when the custom call returns a Tuple and
    // the user is a get-tuple-element. In this case we must update only
    // part of the sharding spec.
    auto result = py::cast<HloSharding>(
        prop_user_sharding_(sharding, instruction->shape(),
                            instruction->raw_backend_config_string()));
    return result;
  }
  std::optional<HloSharding> InferShardingFromOperands(
      const HloInstruction* instruction) const override {
    std::vector<Shape> arg_shapes = GetArgShapes(instruction);
    auto arg_shardings = GetArgShardings(instruction);

    py::gil_scoped_acquire gil;
    auto py_result = infer_sharding_from_operands_(
        arg_shapes, arg_shardings, instruction->shape(),
        instruction->raw_backend_config_string());
    if (py_result.is_none()) {
      return std::nullopt;
    }
    return py::cast<HloSharding>(py_result);
  }
  bool IsCustomCallShardable(const HloInstruction* instruction) const override {
    return true;
  }
  bool CanSideEffectingHaveReplicatedSharding() const override {
    return can_side_effecting_have_replicated_sharding_;
  }

  py::object prop_user_sharding_;
  py::object partition_;
  py::object infer_sharding_from_operands_;
  bool can_side_effecting_have_replicated_sharding_;
};

void BuildCustomCallShardingPybindAPI(pybind11::module& m) {
  m.def(
      "register_custom_call_partitioner",
      [](std::string name, py::object prop_user_sharding, py::object partition,
         py::object infer_sharding_from_operands,
         bool can_side_effecting_have_replicated_sharding) {
        RegisterCustomCallPartitioner(
            name,
            std::make_unique<PyCustomCallPartitioner>(
                prop_user_sharding, partition, infer_sharding_from_operands,
                can_side_effecting_have_replicated_sharding));
      },
      R"(Registers a partitioner for a custom-call operation.

Args:
  name: custom_call_target to match.
  prop_user_sharding: Custom backwards sharding propagation rule.
     Takes result sharding and returns the instruction sharding.
  partition: Lowering rule. Takes operand and result shardings and returns
     a generated HLO and sharding specs. The spmd lowerer first reshards
     to match the returned sharding specs and then inserts the generated hlo.
  infer_sharding_from_operands: Custom forwards sharding propagation rule.
     Takes operand sharding and returns the instruction sharding.
  can_side_effecting_have_replicated_sharding: Side effecting ops are not
     allowed to have replicated sharding. Pass true to disable this check.
)",
      py::arg("name"), py::arg("prop_user_sharding"), py::arg("partition"),
      py::arg("infer_sharding_from_operands"),
      py::arg("can_side_effecting_have_replicated_sharding") = false);

  py::module hlo_sharding_util_m = m.def_submodule(
      "hlo_sharding_util", "Utilities for manipulating HloSharding.");
  hlo_sharding_util_m.def(
      "PartiallyReplicateTiledShardingOnDims",
      [](const HloSharding& sharding, std::vector<int64_t> dims) {
        return hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
            sharding, dims);
      });
}

}  // namespace xla
