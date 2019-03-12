/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/norm_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

namespace xla {
namespace poplarplugin {
namespace {
StatusOr<std::tuple<uint32, uint32, float>> GetNormOpts(
    const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
  TF_ASSIGN_OR_RETURN(int32 feature_index_int32,
                      attribute_map.GetAttributeAsInt("feature_index"));
  auto optional_feature_index = convert_scalar<uint32>(feature_index_int32);
  if (!optional_feature_index) {
    return xla::FailedPrecondition("Norm - Feature index can't be casted.");
  }
  const auto feature_index = *optional_feature_index;

  TF_ASSIGN_OR_RETURN(int32 num_groups_int32,
                      attribute_map.GetAttributeAsInt("num_groups"));
  auto optional_num_groups = convert_scalar<uint32>(num_groups_int32);
  if (!optional_num_groups) {
    return xla::FailedPrecondition("Norm - Num groups can't be casted.");
  }
  const auto num_groups = *optional_num_groups;

  TF_ASSIGN_OR_RETURN(float epsilon,
                      attribute_map.GetAttributeAsFloat("epsilon"));
  return std::make_tuple(feature_index, num_groups, epsilon);
};

class NormInferenceAndTrainingOp : public PoplibsOpDef {
  StatusOr<poplar::Tensor> Allocator(
      poplar::Graph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target,
      const IPUCustomKernelsUtil::AttributeMap& attribute_map,
      const TensorMap& tensor_map) override {
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;
    std::vector<const HloInstruction*> forward_path =
        tensor_target.forward_path;
    absl::optional<const HloInstruction*> layout = tensor_target.layout;
    absl::optional<int64> layout_output_idx = tensor_target.layout_output_idx;

    TF_ASSIGN_OR_RETURN(int32 feature_index_int32,
                        attribute_map.GetAttributeAsInt("feature_index"));
    auto optional_feature_index = convert_scalar<uint32>(feature_index_int32);
    if (!optional_feature_index) {
      return xla::FailedPrecondition("Norm - Feature index can't be casted.");
    }
    const auto feature_index = *optional_feature_index;

    switch (input_index) {
      case 1: {
        return AddNormScaleTensor(graph, name, *layout, *layout_output_idx,
                                  feature_index, forward_path, tensor_map);
      }
      case 2: {
        return AddNormOffsetTensor(graph, name, *layout, *layout_output_idx,
                                   feature_index, forward_path, tensor_map);
      }
      default: {
        return xla::FailedPrecondition(
            "NormInferenceTraining op %s should not be allocating on index "
            "%lld.",
            inst->name().c_str(), input_index);
      }
    }
  }
};

class GroupNormInferenceOp : public NormInferenceAndTrainingOp {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const IPUCustomKernelsUtil::AttributeMap& attribute_map) override {
    uint32 feature_index;
    uint32 num_groups;
    float epsilon;
    TF_ASSIGN_OR_RETURN(std::tie(feature_index, num_groups, epsilon),
                        GetNormOpts(attribute_map));
    return CreateNormInference(NormType::GroupNorm, graph, res, inst, epsilon,
                               feature_index, num_groups, tensor_map);
  }
};
REGISTER_POPLIBS_OP(Popnn, GroupNormInference, GroupNormInferenceOp);

class GroupNormTrainingOp : public NormInferenceAndTrainingOp {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
    uint32 feature_index;
    uint32 num_groups;
    float epsilon;
    TF_ASSIGN_OR_RETURN(std::tie(feature_index, num_groups, epsilon),
                        GetNormOpts(attribute_map));
    return CreateNormTraining(NormType::GroupNorm, graph, res, inst, epsilon,
                              feature_index, num_groups, tensor_map);
  }
};
REGISTER_POPLIBS_OP(Popnn, GroupNormTraining, GroupNormTrainingOp);

class GroupNormGradOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
    uint32 feature_index;
    uint32 num_groups;
    float epsilon;
    TF_ASSIGN_OR_RETURN(std::tie(feature_index, num_groups, epsilon),
                        GetNormOpts(attribute_map));
    return CreateNormGrad(NormType::GroupNorm, graph, res, inst, epsilon,
                          feature_index, num_groups, tensor_map);
  }
};
REGISTER_POPLIBS_OP(Popnn, GroupNormGrad, GroupNormGradOp);

class GroupNormStatisticsOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
    uint32 feature_index;
    uint32 num_groups;
    float epsilon;
    TF_ASSIGN_OR_RETURN(std::tie(feature_index, num_groups, epsilon),
                        GetNormOpts(attribute_map));
    return CreateNormStatistics(NormType::GroupNorm, graph, res, inst, epsilon,
                                feature_index, num_groups, tensor_map);
  }
};
REGISTER_POPLIBS_OP(Popnn, GroupNormStatistics, GroupNormStatisticsOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla