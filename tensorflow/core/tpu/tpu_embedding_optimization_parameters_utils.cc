/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace tpu {

std::string GetOptimizationAlgorithmName(OptimizationAlgorithm alg) {
  switch (alg) {
    case OptimizationAlgorithm::kAdagrad:
      return "Adagrad";
    case OptimizationAlgorithm::kAdagradMomentum:
      return "AdagradMomentum";
    case OptimizationAlgorithm::kBoundedAdagrad:
      return "BoundedAdagrad";
    case OptimizationAlgorithm::kStochasticGradientDescent:
      return "StochasticGradientDescent";
    case OptimizationAlgorithm::kFtrl:
      return "FTRL";
    case OptimizationAlgorithm::kAdam:
      return "ADAM";
    case OptimizationAlgorithm::kMomentum:
      return "Momentum";
    case OptimizationAlgorithm::kRmsProp:
      return "RMSProp";
    case OptimizationAlgorithm::kCenteredRmsProp:
      return "CenteredRMSProp";
    case OptimizationAlgorithm::kMdlAdagradLight:
      return "MDLAdagradLight";
    case OptimizationAlgorithm::kAdadelta:
      return "Adadelta";
    case OptimizationAlgorithm::kProximalAdagrad:
      return "ProximalAdagrad";
    case OptimizationAlgorithm::kOnlineYogi:
      return "OnlineYogi";
    case OptimizationAlgorithm::kProximalYogi:
      return "ProximalYogi";
    case OptimizationAlgorithm::kFrequencyEstimator:
      return "FrequencyEstimator";
    case OptimizationAlgorithm::kUserDefinedProgram:
      return "UserDefinedProgram";
    case OptimizationAlgorithm::kAssign:
      return "Assign";
    case OptimizationAlgorithm::PARAMETERS_NOT_SET:
      return "*** Not set ***";
  }
  return "*** Not set ***";
}

std::string GetOptimizationAlgorithmFriendlyName(OptimizationAlgorithm alg) {
  switch (alg) {
    case OptimizationAlgorithm::kAdagrad:
      return "Adagrad";
    case OptimizationAlgorithm::kAdagradMomentum:
      return "Adagrad with Momentum";
    case OptimizationAlgorithm::kBoundedAdagrad:
      return "Bounded Adagrad";
    case OptimizationAlgorithm::kStochasticGradientDescent:
      return "stochastic gradient descent";
    case OptimizationAlgorithm::kFtrl:
      return "FTRL";
    case OptimizationAlgorithm::kAdam:
      return "ADAM";
    case OptimizationAlgorithm::kMomentum:
      return "Momentum";
    case OptimizationAlgorithm::kRmsProp:
      return "RMSProp";
    case OptimizationAlgorithm::kCenteredRmsProp:
      return "centered RMSProp";
    case OptimizationAlgorithm::kMdlAdagradLight:
      return "MDL Adagrad Light";
    case OptimizationAlgorithm::kAdadelta:
      return "Adadelta";
    case OptimizationAlgorithm::kProximalAdagrad:
      return "proximal Adagrad";
    case OptimizationAlgorithm::kOnlineYogi:
      return "online Yogi";
    case OptimizationAlgorithm::kProximalYogi:
      return "proximal Yogi";
    case OptimizationAlgorithm::kFrequencyEstimator:
      return "frequency estimator";
    case OptimizationAlgorithm::kUserDefinedProgram:
      return "UserDefinedProgram";
    case OptimizationAlgorithm::kAssign:
      return "Assign";
    case OptimizationAlgorithm::PARAMETERS_NOT_SET:
      return "unknown (not specified)";
  }
  return "unknown (not specified)";
}

// Returns the number of optimization parameter vectors used by the optimization
// algorithm, excluding the weights themselves and assuming no gradient
// accumulation.
Status GetBaseAuxiliaryParameterCount(const OptimizationParameters& params,
                                      int* count) {
  switch (params.parameters_case()) {
    case OptimizationAlgorithm::kAdagrad:
      *count = 1;
      return OkStatus();
    case OptimizationAlgorithm::kAdagradMomentum:
      *count = 2;
      return OkStatus();
    case OptimizationAlgorithm::kBoundedAdagrad:
      *count = 1;
      return OkStatus();
    case OptimizationAlgorithm::kStochasticGradientDescent:
      *count = 0;
      return OkStatus();
    case OptimizationAlgorithm::kFtrl:
      *count = 2;
      return OkStatus();
    case OptimizationAlgorithm::kAdam:
      *count = 2;
      return OkStatus();
    case OptimizationAlgorithm::kMomentum:
      *count = 1;
      return OkStatus();
    case OptimizationAlgorithm::kRmsProp:
      *count = 2;
      return OkStatus();
    case OptimizationAlgorithm::kCenteredRmsProp:
      *count = 3;
      return OkStatus();
    case OptimizationAlgorithm::kMdlAdagradLight:
      *count = 3;
      return OkStatus();
    case OptimizationAlgorithm::kAdadelta:
      *count = 2;
      return OkStatus();
    case OptimizationAlgorithm::kProximalAdagrad:
      *count = 1;
      return OkStatus();
    case OptimizationAlgorithm::kOnlineYogi:
      *count = 2;
      return OkStatus();
    case OptimizationAlgorithm::kProximalYogi:
      *count = 2;
      return OkStatus();
    case OptimizationAlgorithm::kFrequencyEstimator:
      *count = 1;
      return OkStatus();
    case OptimizationAlgorithm::kUserDefinedProgram: {
      const xla::ProgramShapeProto& program_shape =
          params.user_defined_program().program().host_program_shape();

      const int num_inputs = program_shape.parameters_size();
      const int num_outputs = program_shape.result().tuple_shapes_size();

      if ((num_inputs < 2) || ((num_inputs != num_outputs + 1) &&
                               (num_inputs != num_outputs + 2))) {
        return errors::InvalidArgument(
            "User-defined TPU embedding optimizer program must have at least "
            "two inputs and the number of outputs must be 1 or 2 less than the "
            "number of inputs. Received ",
            num_inputs, " input(s) and ", num_outputs, "output(s).");
      }

      *count = num_outputs - 1;

      return OkStatus();
    }
    case OptimizationAlgorithm::kAssign:
      *count = 0;
      return OkStatus();
    case OptimizationAlgorithm::PARAMETERS_NOT_SET:
      return errors::InvalidArgument("No optimization algorithm specified");
  }
  return errors::InvalidArgument("No optimization algorithm specified");
}

Status GetGradientAccumulationSupport(const OptimizationParameters& params,
                                      GradientAccumulationSupport* support) {
  int auxiliary_parameter_count;
  TF_RETURN_IF_ERROR(
      GetBaseAuxiliaryParameterCount(params, &auxiliary_parameter_count));
  *support = auxiliary_parameter_count + 1 <= kMaxAuxiliaryParameterCount
                 ? GradientAccumulationSupport::kSupported
                 : GradientAccumulationSupport::kNotSupported;
  return OkStatus();
}

Status UseGradientAccumulation(const OptimizationParameters& params,
                               bool* use_gradient_accumulation) {
  GradientAccumulationSupport support;
  TF_RETURN_IF_ERROR(GetGradientAccumulationSupport(params, &support));
  bool raw_gradient_accumulation_status = false;
  switch (params.gradient_accumulation_status()) {
    case GradientAccumulationStatus::UNSPECIFIED: {
      // Default is now to turn gradient accumulation on by default.
      raw_gradient_accumulation_status = true;
      break;
    }
    case GradientAccumulationStatus::DISABLED: {
      raw_gradient_accumulation_status = false;
      break;
    }
    case GradientAccumulationStatus::ENABLED: {
      raw_gradient_accumulation_status = true;
      break;
    }
    default:
      return errors::Internal(
          absl::StrCat("Unsupported gradient accumulation status ",
                       GradientAccumulationStatus_Status_Name(
                           params.gradient_accumulation_status())));
  }
  switch (support) {
    case GradientAccumulationSupport::kSupported: {
      *use_gradient_accumulation = raw_gradient_accumulation_status;
      break;
    }
    case GradientAccumulationSupport::kNotSupported: {
      if (raw_gradient_accumulation_status) {
        return errors::InvalidArgument(strings::Printf(
            "Optimization algorithm %s does not support gradient accumulation "
            "but parameters specify it.",
            GetOptimizationAlgorithmName(params.parameters_case()).c_str()));
      }
      *use_gradient_accumulation = false;
      break;
    }
  }
  return OkStatus();
}

Status GetOptimizationAlgorithmStateVariables(
    const OptimizationParameters& params,
    std::vector<StateVariableSpecification>* state_variables) {
  // The parameter set for the weights themselves is required to be named
  // "parameters". The rest should stay stable for compatibility. There is an
  // internal function, GetOptimizationAlgorithmStateVariableInternalIndices,
  // that needs to be updated along with this one.
  bool use_gradient_accumulation;
  TF_RETURN_IF_ERROR(
      UseGradientAccumulation(params, &use_gradient_accumulation));

  auto add_state_variable = [&](const std::string& name) {
    StateVariableSpecification spec;
    spec.set_name(name);
    (void)spec.mutable_user_defined();
    state_variables->push_back(spec);
  };

  switch (params.parameters_case()) {
    case OptimizationAlgorithm::kAdagrad: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      break;
    }
    case OptimizationAlgorithm::kAdagradMomentum: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      add_state_variable("momenta");
      break;
    }
    case OptimizationAlgorithm::kBoundedAdagrad: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      break;
    }
    case OptimizationAlgorithm::kStochasticGradientDescent: {
      add_state_variable("parameters");
      break;
    }
    case OptimizationAlgorithm::kFtrl: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      add_state_variable("linears");
      break;
    }
    case OptimizationAlgorithm::kAdam: {
      add_state_variable("parameters");
      add_state_variable("momenta");
      add_state_variable("velocities");
      break;
    }
    case OptimizationAlgorithm::kMomentum: {
      add_state_variable("parameters");
      add_state_variable("momenta");
      break;
    }
    case OptimizationAlgorithm::kRmsProp: {
      add_state_variable("parameters");
      add_state_variable("ms");
      add_state_variable("mom");
      break;
    }
    case OptimizationAlgorithm::kCenteredRmsProp: {
      add_state_variable("parameters");
      add_state_variable("ms");
      add_state_variable("mom");
      add_state_variable("mg");
      break;
    }
    case OptimizationAlgorithm::kMdlAdagradLight: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      add_state_variable("weights");
      add_state_variable("benefits");
      break;
    }
    case OptimizationAlgorithm::kAdadelta: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      add_state_variable("updates");
      break;
    }
    case OptimizationAlgorithm::kProximalAdagrad: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      break;
    }
    case OptimizationAlgorithm::kOnlineYogi: {
      add_state_variable("parameters");
      add_state_variable("vs");
      add_state_variable("linears");
      break;
    }
    case OptimizationAlgorithm::kProximalYogi: {
      add_state_variable("parameters");
      add_state_variable("v");
      add_state_variable("m");
      break;
    }
    case OptimizationAlgorithm::kFrequencyEstimator: {
      add_state_variable("parameters");
      add_state_variable("last_hit_step");
      break;
    }
    case OptimizationAlgorithm::kUserDefinedProgram: {
      add_state_variable("parameters");
      int num_slots = -1;
      TF_RETURN_IF_ERROR(GetBaseAuxiliaryParameterCount(params, &num_slots));
      for (int i = 0; i < num_slots; ++i) {
        add_state_variable(absl::StrCat("Slot_", i));
      }
      break;
    }
    case OptimizationAlgorithm::kAssign: {
      add_state_variable("parameters");
      break;
    }
    case OptimizationAlgorithm::PARAMETERS_NOT_SET: {
      return errors::InvalidArgument("No optimization algorithm specified");
    }
  }

  // This needs to be last for compatibility.
  if (use_gradient_accumulation) {
    StateVariableSpecification gradient_acc;
    gradient_acc.set_name("gradient_accumulators");
    gradient_acc.mutable_fill_with_constant()->set_initial_value(
        GradientAccumulatorInitialValue());
    state_variables->push_back(std::move(gradient_acc));
  }

  if (state_variables->size() > kMaxAuxiliaryParameterCount + 1) {
    return errors::InvalidArgument(
        "Optimization algorithm",
        GetOptimizationAlgorithmName(params.parameters_case()),
        "does not support gradient accumulation because it "
        "already has too many other accumulators");
  }
  return OkStatus();
}

std::vector<OptimizationAlgorithm> GetOptimizationAlgorithms() {
  return {
      OptimizationAlgorithm::kAdagrad,
      OptimizationAlgorithm::kAdagradMomentum,
      OptimizationAlgorithm::kBoundedAdagrad,
      OptimizationAlgorithm::kStochasticGradientDescent,
      OptimizationAlgorithm::kFtrl,
      OptimizationAlgorithm::kAdam,
      OptimizationAlgorithm::kMomentum,
      OptimizationAlgorithm::kRmsProp,
      OptimizationAlgorithm::kCenteredRmsProp,
      OptimizationAlgorithm::kMdlAdagradLight,
      OptimizationAlgorithm::kAdadelta,
      OptimizationAlgorithm::kProximalAdagrad,
      OptimizationAlgorithm::kOnlineYogi,
      OptimizationAlgorithm::kProximalYogi,
      OptimizationAlgorithm::kFrequencyEstimator,
      OptimizationAlgorithm::kUserDefinedProgram,
      OptimizationAlgorithm::kAssign,
  };
}

Status LoadOpShapeFunction::operator()(
    shape_inference::InferenceContext* c) const {
  int table_id;
  TF_RETURN_IF_ERROR(c->GetAttr("table_id", &table_id));
  string table_name;
  TF_RETURN_IF_ERROR(c->GetAttr("table_name", &table_name));
  // Exactly one must be non-default.
  if ((table_id >= 0) == (!table_name.empty())) {
    return errors::InvalidArgument(
        "exactly one of table_id or table_name must be non-default");
  }
  int num_shards;
  TF_RETURN_IF_ERROR(c->GetAttr("num_shards", &num_shards));
  int shard_id;
  TF_RETURN_IF_ERROR(c->GetAttr("shard_id", &shard_id));

  // Verify shapes have rank 2 and are compatible when they are
  // required to be valid.
  shape_inference::ShapeHandle parameter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &parameter_shape));
  for (int j = 1; j < c->num_inputs(); ++j) {
    shape_inference::ShapeHandle accumulator_j_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(j), 2, &accumulator_j_shape));
    shape_inference::ShapeHandle merged;
    TF_RETURN_IF_ERROR(c->Merge(parameter_shape, accumulator_j_shape, &merged));
  }

  return OkStatus();
}

Status RetrieveOpShapeFunction::operator()(
    shape_inference::InferenceContext* c) const {
  int table_id;
  TF_RETURN_IF_ERROR(c->GetAttr("table_id", &table_id));
  string table_name;
  TF_RETURN_IF_ERROR(c->GetAttr("table_name", &table_name));
  // Exactly one must be non-default.
  if ((table_id >= 0) == (!table_name.empty())) {
    return errors::InvalidArgument(
        "exactly one of table_id or table_name must be non-default");
  }
  int num_shards;
  TF_RETURN_IF_ERROR(c->GetAttr("num_shards", &num_shards));
  int shard_id;
  TF_RETURN_IF_ERROR(c->GetAttr("shard_id", &shard_id));
  for (int j = 0; j < c->num_outputs(); ++j) {
    c->set_output(j, c->MakeShape(std::vector<shape_inference::DimensionHandle>(
                         2, c->UnknownDim())));
  }
  return OkStatus();
}

}  // namespace tpu
}  // namespace tensorflow
