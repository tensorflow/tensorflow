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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace tpu {

string GetOptimizationAlgorithmName(OptimizationAlgorithm alg) {
  switch (alg) {
    case OptimizationAlgorithm::kAdagrad:
      return "Adagrad";
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
    case OptimizationAlgorithm::PARAMETERS_NOT_SET:
      return "*** Not set ***";
  }
  return "*** Not set ***";
}

string GetOptimizationAlgorithmFriendlyName(OptimizationAlgorithm alg) {
  switch (alg) {
    case OptimizationAlgorithm::kAdagrad:
      return "Adagrad";
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
    case OptimizationAlgorithm::PARAMETERS_NOT_SET:
      return "unknown (not specified)";
  }
  return "unknown (not specified)";
}

// Returns the number of optimization parameter vectors used by the optimization
// algorithm, excluding the weights themselves and assuming no gradient
// accumulation.
Status GetBaseAuxiliaryParameterCount(OptimizationAlgorithm alg, int* count) {
  switch (alg) {
    case OptimizationAlgorithm::kAdagrad:
      *count = 1;
      return Status::OK();
    case OptimizationAlgorithm::kBoundedAdagrad:
      *count = 1;
      return Status::OK();
    case OptimizationAlgorithm::kStochasticGradientDescent:
      *count = 0;
      return Status::OK();
    case OptimizationAlgorithm::kFtrl:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::kAdam:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::kMomentum:
      *count = 1;
      return Status::OK();
    case OptimizationAlgorithm::kRmsProp:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::kCenteredRmsProp:
      *count = 3;
      return Status::OK();
    case OptimizationAlgorithm::kMdlAdagradLight:
      *count = 3;
      return Status::OK();
    case OptimizationAlgorithm::kAdadelta:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::kProximalAdagrad:
      *count = 1;
      return Status::OK();
    case OptimizationAlgorithm::kOnlineYogi:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::kProximalYogi:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::PARAMETERS_NOT_SET:
      return errors::InvalidArgument("No optimization algorithm specified");
  }
  return errors::InvalidArgument("No optimization algorithm specified");
}

Status GetGradientAccumulationSupport(OptimizationAlgorithm alg,
                                      GradientAccumulationSupport* support) {
  int auxiliary_parameter_count;
  TF_RETURN_IF_ERROR(
      GetBaseAuxiliaryParameterCount(alg, &auxiliary_parameter_count));
  *support = auxiliary_parameter_count + 1 <= kMaxAuxiliaryParameterCount
                 ? GradientAccumulationSupport::kSupported
                 : GradientAccumulationSupport::kNotSupported;
  return Status::OK();
}

namespace {
// Make a normal state variable specification. Please refer to
// //tensorflow/core/protobuf/tpu/optimization_parameters.proto
// (StateVariableSpecification message) for instructions on how to set the
// padding_initial_value field.
StateVariableSpecification MakeStandardStateVariableSpecification(
    const string& name, double padding_initial_value) {
  StateVariableSpecification result;
  result.set_name(name);
  result.mutable_user_defined()->set_padding_initial_value(
      padding_initial_value);
  return result;
}
}  // namespace

Status GetOptimizationAlgorithmStateVariables(
    OptimizationAlgorithm alg, bool use_gradient_accumulation,
    std::vector<StateVariableSpecification>* state_variables) {
  // The first parameter set is always the weights themselves.
  state_variables->push_back(
      MakeStandardStateVariableSpecification("parameters", 0.0));
  // The order of the returned parameters needs to match the offsets used by
  // the algorithm implementations in test_util.cc and
  // address_handler_program_creator.cc.
  switch (alg) {
    case OptimizationAlgorithm::kAdagrad: {
      state_variables->push_back(
          MakeStandardStateVariableSpecification("accumulators", 0.1));
      break;
    }
    case OptimizationAlgorithm::kBoundedAdagrad: {
      state_variables->push_back(
          MakeStandardStateVariableSpecification("accumulators", 0.1));
      break;
    }
    case OptimizationAlgorithm::kStochasticGradientDescent: {
      // None.
      break;
    }
    case OptimizationAlgorithm::kFtrl: {
      state_variables->push_back(
          MakeStandardStateVariableSpecification("accumulators", 0.1));
      state_variables->push_back(
          MakeStandardStateVariableSpecification("linears", 0.0));
      break;
    }
    case OptimizationAlgorithm::kAdam: {
      state_variables->push_back(
          MakeStandardStateVariableSpecification("momenta", 0.0));
      state_variables->push_back(
          MakeStandardStateVariableSpecification("velocities", 0.0));
      break;
    }
    case OptimizationAlgorithm::kMomentum: {
      state_variables->push_back(
          MakeStandardStateVariableSpecification("momenta", 0.0));
      break;
    }
    case OptimizationAlgorithm::kRmsProp: {
      state_variables->push_back(
          MakeStandardStateVariableSpecification("ms", 1.0));
      state_variables->push_back(
          MakeStandardStateVariableSpecification("mom", 0.0));
      break;
    }
    case OptimizationAlgorithm::kCenteredRmsProp: {
      state_variables->push_back(
          MakeStandardStateVariableSpecification("ms", 1.0));
      state_variables->push_back(
          MakeStandardStateVariableSpecification("mom", 0.0));
      state_variables->push_back(
          MakeStandardStateVariableSpecification("mg", 0.0));
      break;
    }
    case OptimizationAlgorithm::kMdlAdagradLight: {
      state_variables->push_back(
          MakeStandardStateVariableSpecification("accumulators", 0.1));
      state_variables->push_back(
          MakeStandardStateVariableSpecification("weights", 0.0));
      state_variables->push_back(
          MakeStandardStateVariableSpecification("benefits", 0.0));
      break;
    }
    case OptimizationAlgorithm::kAdadelta: {
      state_variables->push_back(
          MakeStandardStateVariableSpecification("accumulators", 0.0));
      state_variables->push_back(
          MakeStandardStateVariableSpecification("updates", 0.0));
      break;
    }
    case OptimizationAlgorithm::kProximalAdagrad: {
      state_variables->push_back(
          MakeStandardStateVariableSpecification("accumulators", 0.1));
      break;
    }
    case OptimizationAlgorithm::kOnlineYogi: {
      state_variables->push_back(
          MakeStandardStateVariableSpecification("vs", 0.1));
      state_variables->push_back(
          MakeStandardStateVariableSpecification("linears", 0.0));
      break;
    }
    case OptimizationAlgorithm::kProximalYogi: {
      state_variables->push_back(
          MakeStandardStateVariableSpecification("v", 0.1));
      state_variables->push_back(
          MakeStandardStateVariableSpecification("m", 0.0));
      break;
    }
    case OptimizationAlgorithm::PARAMETERS_NOT_SET: {
      return errors::InvalidArgument("No optimization algorithm specified");
    }
  }
  // This needs to be last so that the save/restore ops do not need to know
  // about gradient accumulation.
  if (use_gradient_accumulation) {
    StateVariableSpecification gradient_acc;
    gradient_acc.set_name("gradient_accumulators");
    gradient_acc.mutable_fill_with_constant()->set_initial_value(
        GradientAccumulatorInitialValue());
    state_variables->push_back(std::move(gradient_acc));
  }
  if (state_variables->size() > kMaxAuxiliaryParameterCount + 1) {
    return errors::InvalidArgument(
        "Optimization algorithm", GetOptimizationAlgorithmName(alg),
        "does not support gradient accumulation because it "
        "already has too many other accumulators");
  }
  return Status::OK();
}  // namespace tpu

std::vector<OptimizationAlgorithm> GetOptimizationAlgorithms() {
  return {
      OptimizationAlgorithm::kAdagrad,
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
  };
}

LoadOpShapeFunction::LoadOpShapeFunction(OptimizationAlgorithm alg,
                                         bool is_debug_op)
    : alg_(alg), is_debug_op_(is_debug_op) {}

Status LoadOpShapeFunction::operator()(
    shape_inference::InferenceContext* c) const {
  GradientAccumulationSupport grad_accum_support;
  TF_CHECK_OK(GetGradientAccumulationSupport(alg_, &grad_accum_support));

  std::vector<StateVariableSpecification> state_variable_specs;
  TF_CHECK_OK(GetOptimizationAlgorithmStateVariables(
      alg_,
      grad_accum_support == GradientAccumulationSupport::kSupported &&
          is_debug_op_,
      &state_variable_specs));
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
  const int user_param_count =
      std::count_if(state_variable_specs.begin(), state_variable_specs.end(),
                    [&](const StateVariableSpecification& sv) {
                      return sv.has_user_defined() || is_debug_op_;
                    });
  std::vector<shape_inference::ShapeHandle> inputs(user_param_count);
  int input_index = 0;
  for (int i = 0; i < state_variable_specs.size(); ++i) {
    if (state_variable_specs[i].has_user_defined() || is_debug_op_) {
      std::vector<shape_inference::ShapeHandle> input_temp;
      TF_RETURN_IF_ERROR(c->input(state_variable_specs[i].name(), &input_temp));
      if (input_temp.size() != 1) {
        return errors::InvalidArgument("each input to be rank 1");
      }
      inputs[input_index] = input_temp[0];
      ++input_index;
    }
  }
  // Verify shapes have rank 2 and are compatible when they are
  // required to be valid.
  shape_inference::ShapeHandle parameter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(inputs[0], 2, &parameter_shape));
  for (int j = 1; j < user_param_count; ++j) {
    shape_inference::ShapeHandle accumulator_j_shape;
    TF_RETURN_IF_ERROR(c->WithRank(inputs[j], 2, &accumulator_j_shape));
    shape_inference::ShapeHandle merged;
    TF_RETURN_IF_ERROR(c->Merge(parameter_shape, accumulator_j_shape, &merged));
  }
  return Status::OK();
}

RetrieveOpShapeFunction::RetrieveOpShapeFunction(OptimizationAlgorithm alg,
                                                 bool is_debug_op)
    : alg_(alg), is_debug_op_(is_debug_op) {}

Status RetrieveOpShapeFunction::operator()(
    shape_inference::InferenceContext* c) const {
  GradientAccumulationSupport grad_accum_support;
  TF_CHECK_OK(GetGradientAccumulationSupport(alg_, &grad_accum_support));

  std::vector<StateVariableSpecification> state_variable_specs;
  TF_CHECK_OK(GetOptimizationAlgorithmStateVariables(
      alg_,
      grad_accum_support == GradientAccumulationSupport::kSupported &&
          is_debug_op_,
      &state_variable_specs));
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
  for (int j = 0; j < state_variable_specs.size(); ++j) {
    if (state_variable_specs[j].has_user_defined() || is_debug_op_) {
      auto shape = c->MakeShape(
          std::vector<shape_inference::DimensionHandle>(2, c->UnknownDim()));
      TF_RETURN_IF_ERROR(
          c->set_output(state_variable_specs[j].name(),
                        std::vector<shape_inference::ShapeHandle>(1, shape)));
    }
  }
  return Status::OK();
}

Status IsOptimizationAlgorithmInternal(OptimizationAlgorithm alg,
                                       bool* internal) {
  switch (alg) {
    case OptimizationAlgorithm::kAdagrad:
    case OptimizationAlgorithm::kStochasticGradientDescent:
    case OptimizationAlgorithm::kFtrl:
    case OptimizationAlgorithm::kAdam:
    case OptimizationAlgorithm::kMomentum:
    case OptimizationAlgorithm::kRmsProp:
    case OptimizationAlgorithm::kCenteredRmsProp:
    case OptimizationAlgorithm::kMdlAdagradLight:
    case OptimizationAlgorithm::kAdadelta:
    case OptimizationAlgorithm::kProximalAdagrad:
    case OptimizationAlgorithm::kProximalYogi: {
      *internal = false;
      return Status::OK();
    }
    case OptimizationAlgorithm::kBoundedAdagrad:
    case OptimizationAlgorithm::kOnlineYogi: {
      *internal = true;
      return Status::OK();
    }
    case OptimizationAlgorithm::PARAMETERS_NOT_SET:
      return errors::InvalidArgument("No optimization algorithm specified");
  }
}

}  // namespace tpu
}  // namespace tensorflow
