/*
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

#include "tensorflow/lite/delegates/gpu/common/transformations/group_norm_mean_var_addition.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {

class AddGroupNormMeanVar : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    
        // transformation only for elementwise layers
        if (node->operation.type != ToString(OperationType::GROUP_NORMALIZATION)){
            return {TransformStatus::SKIPPED, ""};
        }

        // checking the input shape and checking for broadcasting condition
        auto inputs = graph->FindInputs(node->id);
        bool transform_applied = false;

        auto input_value = inputs[0];

        // IF there is no consumers for this input, then there is error in
        // graph generation (a node with no input connections)
        auto input_consumers = graph->FindConsumers(input_value->id);
        if(input_consumers.empty()) {
            return {TransformStatus::INVALID, 
                    "There are no consumers for the input to the node"};
        }

        Node* group_norm_mean;
        absl::Status status = graph->InsertNodeBefore(node->id, &group_norm_mean);
        if(!status.ok())
            return {TransformStatus::INVALID, "Could not insert the group_norm_mean node"};
        group_norm_mean->operation.type = ToString(OperationType::GROUP_NORM_MEAN);


        Value* mean_output_value = graph->NewValue();
        mean_output_value->tensor = inputs[0]->tensor;

        BHWC new_shape;
        new_shape.b = 1;
        new_shape.h = 1;
        new_shape.w = 1;
        new_shape.c = inputs[0]->tensor.shape.c;
        mean_output_value->tensor.shape = new_shape;

        status = graph->AddConsumer(group_norm_mean->id, input_value->id);
        if(!status.ok()) {
            return {TransformStatus::INVALID,
                    absl::StrCat(
                        "Could not add input as consumer to group norm mean kernel",
                            status.message())};
        }

        status = graph->AddConsumer(node->id, mean_output_value->id);
        if(!status.ok()) {
            return {TransformStatus::INVALID,
                    absl::StrCat(
                        "Could not add input as new input to group norm",
                            status.message())};
        }

        status = graph->SetProducer(group_norm_mean->id, mean_output_value->id);
        if(!status.ok()){
            return {TransformStatus::INVALID, 
                    "Could not set producer for group_norm_mean to mean input tensor"};
        }


        // doing for variance kernel;
        Node* group_norm_var;
        status = graph->InsertNodeBefore(node->id, &group_norm_var);

        if(!status.ok())
            return {TransformStatus::INVALID, "Could not insert the group_norm_var node"};
        group_norm_var->operation.type = ToString(OperationType::GROUP_NORM_VAR);

        GroupNormalizationAttributes gn_attr = absl::any_cast<GroupNormalizationAttributes>(node->operation.attributes);
        
        GroupNormVarAttributes gnv_attr;
        gnv_attr.num_groups = gn_attr.groups;
        group_norm_var->operation.attributes = gnv_attr;

        Value* var_output_value = graph->NewValue();
        var_output_value->tensor = inputs[0]->tensor;
        var_output_value->tensor.shape = new_shape;

        //manage variance connections
        status = graph->AddConsumer(group_norm_var->id, input_value->id);
        if(!status.ok()) {
            return {TransformStatus::INVALID,
                    absl::StrCat(
                        "Could not add input as consumer to group norm var kernel",
                            status.message())};
        }

        status = graph->AddConsumer(group_norm_var->id, mean_output_value->id);
        if(!status.ok()) {
            return {TransformStatus::INVALID,
                    absl::StrCat(
                        "Could not add mean value as consumer to group norm var kernel",
                            status.message())};
        }

        status = graph->AddConsumer(node->id, var_output_value->id);
        if(!status.ok()) {
            return {TransformStatus::INVALID,
                    absl::StrCat(
                        "Could not add input as new input to group norm",
                            status.message())};
        }


        status = graph->SetProducer(group_norm_var->id, var_output_value->id);
        if(!status.ok()){
            return {TransformStatus::INVALID, 
                    "Could not set producer for group_norm_var to var tensor"};
        }

        transform_applied = true;
        if (transform_applied) {
            return {TransformStatus::APPLIED, ""};
        }

        return {TransformStatus::SKIPPED, ""};

  }
};

}  // namespace

std::unique_ptr<NodeTransformation> NewGroupNormMeanVarAddition() {
  return absl::make_unique<AddGroupNormMeanVar>();
}

}  // namespace gpu
}  // namespace tflite