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
#include "tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate_with_control_flow.h"

#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace tflite {
namespace example {

// This is not an actual subgraph index, but a placeholder index to store
// the node inputs/outputs/builtin_code information for the initial entry
// subgraph layer. For the entry subgraph, the subgraph index is not used
// for computation, so this index is safe to use.
static const int kTopLevelSubgraphIndex = -1;

namespace {

class SampleStableDelegateKernel : public SimpleOpaqueDelegateKernelInterface {
  bool IsExternalTensor(const TfLiteOpaqueTensor* opaque_tensor) const {
    return external_tensors_.count(opaque_tensor) != 0;
  }

  void DeriveExternalTensors() {
    for (const TfLiteOpaqueTensor* tensor : node_input_tensors_set_) {
      if (node_output_tensors_set_.count(tensor) == 0) {
        external_tensors_.insert(tensor);
      }
    }

    for (const TfLiteOpaqueTensor* tensor : node_output_tensors_set_) {
      if (node_input_tensors_set_.count(tensor) == 0) {
        external_tensors_.insert(tensor);
      }
    }
  }

 public:
  TfLiteStatus Init(TfLiteOpaqueContext* context,
                    const TfLiteOpaqueDelegateParams* params) override {
    if (params->delegate == nullptr) return kTfLiteDelegateError;

    context_ = context;
    std::vector<int> callee_subgraph_indices;
    TfLiteStatus status =
        InitSubgraphNodes(context, kTopLevelSubgraphIndex,
                          params->nodes_to_replace, callee_subgraph_indices);
    if (status != kTfLiteOk) return status;

    // Determine which tensors are external (the TFLite runtime takes care
    // of them) so that we know which tensors are 'internal' to this delegate.
    // For the internal tensors we need to ensure they have memory allocated to
    // store their data, and take care of re-sizing etc.
    DeriveExternalTensors();

    return kTfLiteOk;
  }
  TfLiteStatus InitSubgraphNodes(TfLiteOpaqueContext* context,
                                 int subgraph_index,
                                 const TfLiteIntArray* nodes_to_execute,
                                 std::vector<int>& callee_subgraph_indices) {
    node_input_tensors_[subgraph_index].resize(nodes_to_execute->size);
    node_output_tensors_[subgraph_index].resize(nodes_to_execute->size);
    builtin_codes_[subgraph_index].resize(nodes_to_execute->size);

    for (int i = 0; i < nodes_to_execute->size; ++i) {
      const int node_index = nodes_to_execute->data[i];

      TfLiteOpaqueNode* delegated_node = nullptr;
      TfLiteOperator* delegated_node_registration = nullptr;
      TfLiteOpaqueContextGetNodeAndRegistration(
          context, node_index, &delegated_node, &delegated_node_registration);

      builtin_codes_[subgraph_index][i] =
          TfLiteOperatorGetBuiltInCode(delegated_node_registration);

      for (int n = 0; n < TfLiteOpaqueNodeNumberOfInputs(delegated_node); ++n) {
        auto input_tensor =
            TfLiteOpaqueNodeGetInput(context, delegated_node, n);
        node_input_tensors_[subgraph_index][i].push_back(input_tensor);
        if (subgraph_index == kTopLevelSubgraphIndex) {
          // node_input_tensors_set_ is used for deriving external tensors. For
          // this sample delegate, we only derive external tensors for the
          // top-level subgraph, i.e. for any children subgraphs we handle
          // tensor memory allocation on our own.
          node_input_tensors_set_.insert(input_tensor);
        }
      }

      for (int n = 0; n < TfLiteOpaqueNodeNumberOfOutputs(delegated_node);
           ++n) {
        auto output_tensor =
            TfLiteOpaqueNodeGetOutput(context, delegated_node, n);
        node_output_tensors_[subgraph_index][i].push_back(output_tensor);
        if (subgraph_index == kTopLevelSubgraphIndex) {
          // node_output_tensors_set_ is used for deriving external tensors. For
          // this sample delegate, we only derive external tensors for the
          // top-level subgraph, i.e. for any children subgraphs we handle
          // tensor memory allocation on our own.
          node_output_tensors_set_.insert(output_tensor);
        }
      }

      if (builtin_codes_[subgraph_index][i] == kTfLiteBuiltinWhile) {
        void* builtin_data = TfLiteOpaqueNodeGetBuiltinData(delegated_node);
        TfLiteWhileParams* params =
            reinterpret_cast<TfLiteWhileParams*>(builtin_data);

        control_flow_branch_indices_[subgraph_index][i] = {
            params->cond_subgraph_index, params->body_subgraph_index};

        for (int branch_index :
             control_flow_branch_indices_[subgraph_index][i]) {
          callee_subgraph_indices.push_back(branch_index);

          TfLiteStatus status;
          TfLiteIntArray* execution_plan;
          TfLiteOpaqueContext* branch_context;
          status = TfLiteOpaqueContextAcquireSubgraphContext(
              context, branch_index, &branch_context);
          if (status != kTfLiteOk) return status;
          status = TfLiteOpaqueContextGetExecutionPlan(branch_context,
                                                       &execution_plan);
          if (status != kTfLiteOk) return status;
          status = InitSubgraphNodes(branch_context, branch_index,
                                     execution_plan, callee_subgraph_indices);
          if (status != kTfLiteOk) return status;

          // Release the acquired subgraph context.
          status =
              TfLiteOpaqueContextReleaseSubgraphContext(context, branch_index);
          if (status != kTfLiteOk) return status;
        }
      }
    }

    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                       TfLiteOpaqueNode* delegated_node) override {
    if (external_tensors_.empty()) return kTfLiteOk;

    const int kTheInputTensorSize =
        helpers::CalculateNumElements((*external_tensors_.begin()));
    // For each subgraph
    for (auto [_, node_input_tensors] : node_input_tensors_) {
      // For each node in the subgraph
      for (std::vector<const TfLiteOpaqueTensor*>& vecs : node_input_tensors) {
        // For each tensor in the node
        for (const TfLiteOpaqueTensor* tensor : vecs) {
          if (IsExternalTensor(tensor)) continue;

          std::vector<float>& vec_memory =
              internal_float_tensors_memory_[tensor];
          vec_memory.resize(kTheInputTensorSize);
        }
      }
    }
    // For each subgraph
    for (auto [subgraph_index, node_output_tensors] : node_output_tensors_) {
      // For each node in the subgraph
      for (int i = 0; i < node_output_tensors.size(); ++i) {
        std::vector<const TfLiteOpaqueTensor*>& vecs = node_output_tensors[i];
        // For each tensor in the node
        for (int j = 0; j < vecs.size(); ++j) {
          const TfLiteOpaqueTensor* tensor = vecs[j];
          if (IsExternalTensor(tensor)) break;

          if (builtin_codes_[subgraph_index][i] == kTfLiteBuiltinEqual) {
            std::vector<int>& vec_memory = internal_int_tensors_memory_[tensor];
            vec_memory.resize(kTheInputTensorSize);
          } else {
            std::vector<float>& vec_memory =
                internal_float_tensors_memory_[tensor];
            vec_memory.resize(kTheInputTensorSize);
          }
        }
      }
    }

    return kTfLiteOk;
  }

  int* GetIntRawDataSource(const TfLiteOpaqueTensor* tensor) {
    if (IsExternalTensor(tensor)) {
      return reinterpret_cast<int*>(TfLiteOpaqueTensorData(tensor));
    } else {
      return internal_int_tensors_memory_[tensor].data();
    }
  }

  float* GetFloatRawDataSource(const TfLiteOpaqueTensor* tensor) {
    if (IsExternalTensor(tensor)) {
      return reinterpret_cast<float*>(TfLiteOpaqueTensorData(tensor));
    } else {
      return internal_float_tensors_memory_[tensor].data();
    }
  }

  void CopyRawDataSource(const TfLiteOpaqueTensor* from_tensor,
                         const TfLiteOpaqueTensor* to_tensor) {
    float* from_data = GetFloatRawDataSource(from_tensor);
    float* to_data = GetFloatRawDataSource(to_tensor);
    int number_of_elements = helpers::CalculateNumElements(to_tensor);
    memcpy(to_data, from_data, number_of_elements * sizeof(float));
  }

  TfLiteStatus EvalArithmeticOp(int subgraph_index, int node_index) {
    auto node_input_tensors = node_input_tensors_[subgraph_index];
    auto node_output_tensors = node_output_tensors_[subgraph_index];
    auto builtin_codes = builtin_codes_[subgraph_index];

    float* input1 = GetFloatRawDataSource(node_input_tensors[node_index][0]);
    float* input2 = GetFloatRawDataSource(node_input_tensors[node_index][1]);
    float* output = GetFloatRawDataSource(node_output_tensors[node_index][0]);
    int number_of_elements =
        helpers::CalculateNumElements(node_output_tensors[node_index][0]);
    // We assume that all input, output and intermediate tensors of the
    // delegated subgraph have the same size.
    for (int i = 0; i < number_of_elements; ++i) {
      switch (builtin_codes[node_index]) {
        case kTfLiteBuiltinAdd:
          output[i] = input1[i] + input2[i];
          break;
        case kTfLiteBuiltinSub:
          output[i] = input1[i] - input2[i];
          break;
        case kTfLiteBuiltinMul:
          output[i] = input1[i] * input2[i];
          break;
        default:
          return kTfLiteDelegateError;
      }
    }
    return kTfLiteOk;
  }

  TfLiteStatus EvalComparisonOp(int subgraph_index, int node_index) {
    auto node_input_tensors = node_input_tensors_[subgraph_index];
    auto node_output_tensors = node_output_tensors_[subgraph_index];
    auto builtin_codes = builtin_codes_[subgraph_index];

    float* input1 = GetFloatRawDataSource(node_input_tensors[node_index][0]);
    float* input2 = GetFloatRawDataSource(node_input_tensors[node_index][1]);
    int* output = GetIntRawDataSource(node_output_tensors[node_index][0]);
    int number_of_elements =
        helpers::CalculateNumElements(node_output_tensors[node_index][0]);
    // We assume that all input, output and intermediate tensors of the
    // delegated subgraph have the same size.
    for (int i = 0; i < number_of_elements; ++i) {
      switch (builtin_codes[node_index]) {
        case kTfLiteBuiltinEqual:
          output[i] = input1[i] == input2[i];
          break;
        default:
          return kTfLiteDelegateError;
      }
    }
    return kTfLiteOk;
  }

  TfLiteStatus EvalWhileOp(int while_subgraph_index, int while_node_index) {
    auto branch_indices =
        control_flow_branch_indices_[while_subgraph_index][while_node_index];
    int cond_subgraph_index = branch_indices[0];
    int body_subgraph_index = branch_indices[1];
    int last_cond_node_index =
        node_output_tensors_[cond_subgraph_index].size() - 1;
    int last_body_node_index =
        node_output_tensors_[body_subgraph_index].size() - 1;
    // 1. Copy while input to cond input.
    CopyRawDataSource(
        node_input_tensors_[while_subgraph_index][while_node_index][0],
        node_input_tensors_[cond_subgraph_index][0][0]);

    TfLiteStatus status;
    while (true) {
      status = EvalSubgraph(cond_subgraph_index);
      if (status != kTfLiteOk) return status;

      int* cond_output = GetIntRawDataSource(
          node_output_tensors_[cond_subgraph_index][last_cond_node_index][0]);
      int number_of_elements = helpers::CalculateNumElements(
          node_output_tensors_[cond_subgraph_index][last_cond_node_index][0]);
      // We assume that all input, output and intermediate tensors of the
      // delegated subgraph have the same size.
      bool condition = true;
      for (int i = 0; i < number_of_elements; ++i) {
        if (cond_output[i] == 0) {
          condition = false;
          break;
        }
      }
      if (!condition) {
        // 4. Copy body output to while output.
        CopyRawDataSource(
            node_output_tensors_[body_subgraph_index][last_body_node_index][0],
            node_output_tensors_[while_subgraph_index][while_node_index][0]);
        break;
      }

      // 2. Copy cond input to body input.
      CopyRawDataSource(node_input_tensors_[cond_subgraph_index][0][0],
                        node_input_tensors_[body_subgraph_index][0][0]);

      status = EvalSubgraph(body_subgraph_index);
      if (status != kTfLiteOk) return status;

      // 3. Copy body output to cond input.
      CopyRawDataSource(
          node_output_tensors_[body_subgraph_index][last_body_node_index][0],
          node_input_tensors_[cond_subgraph_index][0][0]);
    }

    return kTfLiteOk;
  }

  TfLiteStatus EvalSubgraph(int subgraph_index) {
    TfLiteStatus status;
    for (int i = 0; i < node_input_tensors_[subgraph_index].size(); ++i) {
      status = EvalNode(subgraph_index, i);
      if (status != kTfLiteOk) return status;
    }
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteOpaqueContext* context,
                    TfLiteOpaqueNode* delegated_node) override {
    return EvalSubgraph(kTopLevelSubgraphIndex);
  }

  TfLiteStatus EvalNode(int subgraph_index, int node_index) {
    TfLiteStatus status;
    switch (builtin_codes_[subgraph_index][node_index]) {
      case kTfLiteBuiltinAdd:
      case kTfLiteBuiltinSub:
      case kTfLiteBuiltinMul:
        status = EvalArithmeticOp(subgraph_index, node_index);
        break;
      case kTfLiteBuiltinEqual:
        status = EvalComparisonOp(subgraph_index, node_index);
        break;
      case kTfLiteBuiltinWhile:
        status = EvalWhileOp(subgraph_index, node_index);
        break;
      default:
        return kTfLiteDelegateError;
    }
    if (status != kTfLiteOk) {
      return status;
    }

    return kTfLiteOk;
  }

 private:
  absl::flat_hash_map<int, absl::flat_hash_map<int, std::vector<int>>>
      control_flow_branch_indices_;
  absl::flat_hash_map<int, std::vector<std::vector<const TfLiteOpaqueTensor*>>>
      node_input_tensors_;
  absl::flat_hash_set<const TfLiteOpaqueTensor*> node_input_tensors_set_;
  absl::flat_hash_map<int, std::vector<std::vector<const TfLiteOpaqueTensor*>>>
      node_output_tensors_;
  absl::flat_hash_set<const TfLiteOpaqueTensor*> node_output_tensors_set_;
  absl::flat_hash_set<const TfLiteOpaqueTensor*> external_tensors_;
  absl::flat_hash_map<const TfLiteOpaqueTensor*, std::vector<float>>
      internal_float_tensors_memory_;
  absl::flat_hash_map<const TfLiteOpaqueTensor*, std::vector<int>>
      internal_int_tensors_memory_;
  TfLiteOpaqueContext* context_;
  // Holds the builtin code of the ops per subgraph.
  // builtin_code_[i] is the type of node at index 'i'
  absl::flat_hash_map<int, std::vector<int>> builtin_codes_;
};
}  // namespace

TfLiteStatus SampleStableDelegate::ComputeCompatibleCalleeSubgraphs(
    TfLiteOpaqueContext* opaque_context, int subgraph_index) {
  TfLiteStatus status;
  TfLiteOpaqueContext* current_context;
  status = TfLiteOpaqueContextAcquireSubgraphContext(
      opaque_context, subgraph_index, &current_context);
  if (status != kTfLiteOk) {
    return status;
  }

  TfLiteIntArray* execution_plan;
  status =
      TfLiteOpaqueContextGetExecutionPlan(current_context, &execution_plan);
  if (status != kTfLiteOk) {
    return status;
  }

  bool is_compatible_subgraph = true;
  // The list of TfLite nodes currently in the model.
  for (int i = 0; i < execution_plan->size; ++i) {
    int node_index = execution_plan->data[i];
    TfLiteOpaqueNode* node = nullptr;
    TfLiteOperator* registration = nullptr;
    status = TfLiteOpaqueContextGetNodeAndRegistration(
        current_context, node_index, &node, &registration);
    if (status != kTfLiteOk) {
      return status;
    }

    TfLiteBuiltinOperator builtin_operator =
        TfLiteOperatorGetBuiltInCode(registration);
    if (builtin_operator == kTfLiteBuiltinWhile) {
      void* builtin_data = TfLiteOpaqueNodeGetBuiltinData(node);
      const auto* op_data =
          reinterpret_cast<const TfLiteWhileParams*>(builtin_data);
      // Handle WHILE's cond subgraph.
      AddCalleeSubgraphToCallerSubgraph(op_data->cond_subgraph_index,
                                        subgraph_index);
      ComputeCompatibleCalleeSubgraphs(opaque_context,
                                       op_data->cond_subgraph_index);
      // Handle WHILE's body subgraph.
      AddCalleeSubgraphToCallerSubgraph(op_data->body_subgraph_index,
                                        subgraph_index);
      ComputeCompatibleCalleeSubgraphs(opaque_context,
                                       op_data->body_subgraph_index);
    }
    if (!IsNodeSupportedByDelegate(registration, node, current_context)) {
      is_compatible_subgraph = false;
    }
  }
  if (is_compatible_subgraph) {
    AddCompatibleCalleeSubgraph(subgraph_index);
  }

  // Release the acquired subgraph context.
  status =
      TfLiteOpaqueContextReleaseSubgraphContext(opaque_context, subgraph_index);
  if (status != kTfLiteOk) {
    return status;
  }

  return kTfLiteOk;
}

TfLiteStatus SampleStableDelegate::PrepareControlFlow(
    TfLiteOpaqueContext* opaque_context) {
  // Walk through all the subgraphs, and collect all the subgraphs called by
  // control flow ops (e.g. WHILE or IF) to `callee_subgraph_indices` in the
  // topological order.
  constexpr int kPrimarySubgraphIndex = 0;
  ComputeCompatibleCalleeSubgraphs(opaque_context, kPrimarySubgraphIndex);

  // Mark callee subgraphs as "delegation-skippable" that should be skipped by
  // interpreter->ModifyGraphWithDelegate().
  for (const auto& [caller_subgraph_index, callee_subgraph_indices] :
       control_flow_subgraph_tree_) {
    if (callee_subgraph_indices.empty()) {
      continue;
    }
    bool callee_subgraphs_all_delegatable = true;
    for (int callee_subgraph_index : callee_subgraph_indices) {
      if (!IsCompatibleCalleeSubgraph(callee_subgraph_index)) {
        callee_subgraphs_all_delegatable = false;
      }
    }
    if (!callee_subgraphs_all_delegatable) {
      continue;
    }
    for (int callee_subgraph_index : callee_subgraph_indices) {
      TfLiteOpaqueContextMarkSubgraphAsDelegationSkippable(
          opaque_context, callee_subgraph_index);
    }
  }

  return kTfLiteOk;
}

int helpers::CalculateNumElements(const TfLiteOpaqueTensor* opaque_tensor) {
  int total_num_elements = 1;
  for (int i = 0; i < TfLiteOpaqueTensorNumDims(opaque_tensor); ++i) {
    total_num_elements *= TfLiteOpaqueTensorDim(opaque_tensor, i);
  }
  return total_num_elements;
}

bool SampleStableDelegate::IsNodeSupportedByDelegate(
    const TfLiteOperator* registration_external, const TfLiteOpaqueNode* node,
    TfLiteOpaqueContext* context) const {
  TfLiteBuiltinOperator builtin_operator =
      TfLiteOperatorGetBuiltInCode(registration_external);
  void* builtin_data = TfLiteOpaqueNodeGetBuiltinData(node);
  // List of supported / unsupported ops.
  switch (builtin_operator) {
    case kTfLiteBuiltinAdd: {
      TfLiteAddParams* params =
          reinterpret_cast<TfLiteAddParams*>(builtin_data);
      if (!params || params->activation != kTfLiteActNone) return false;
      break;
    }
    case kTfLiteBuiltinSub: {
      TfLiteSubParams* params =
          reinterpret_cast<TfLiteSubParams*>(builtin_data);
      if (!params || params->activation != kTfLiteActNone) return false;
      break;
    }
    case kTfLiteBuiltinMul: {
      TfLiteMulParams* params =
          reinterpret_cast<TfLiteMulParams*>(builtin_data);
      if (!params || params->activation != kTfLiteActNone) return false;
      break;
    }
    case kTfLiteBuiltinEqual:
      break;
    case kTfLiteBuiltinWhile: {
      TfLiteWhileParams* params =
          reinterpret_cast<TfLiteWhileParams*>(builtin_data);
      if (!params || !IsCompatibleCalleeSubgraph(params->cond_subgraph_index) ||
          !IsCompatibleCalleeSubgraph(params->body_subgraph_index)) {
        return false;
      }
      break;
    }
    default:
      // Any unlisted ops are unsupported by default without further check.
      return false;
  }

  // The delegate only accepts two inputs with float32 type for binary ops and
  // one input with float32 type for control flow ops.
  if (builtin_operator == kTfLiteBuiltinWhile) {
    if (TfLiteOpaqueNodeNumberOfInputs(node) != 1) return false;
    const TfLiteOpaqueTensor* tensor =
        TfLiteOpaqueNodeGetInput(context, node, 0);
    if (!tensor || TfLiteOpaqueTensorType(tensor) != kTfLiteFloat32)
      return false;
  } else {
    if (TfLiteOpaqueNodeNumberOfInputs(node) != 2) return false;
    const TfLiteOpaqueTensor* tensor_1 =
        TfLiteOpaqueNodeGetInput(context, node, 0);
    const TfLiteOpaqueTensor* tensor_2 =
        TfLiteOpaqueNodeGetInput(context, node, 1);
    if (!tensor_1 || TfLiteOpaqueTensorType(tensor_1) != kTfLiteFloat32)
      return false;
    if (!tensor_2 || TfLiteOpaqueTensorType(tensor_2) != kTfLiteFloat32)
      return false;
    // The delegate doesn't support broadcasting; it requires both inputs to
    // have the same shape.
    if (TfLiteOpaqueTensorNumDims(tensor_1) !=
        TfLiteOpaqueTensorNumDims(tensor_2))
      return false;
    for (int i = 0; i < TfLiteOpaqueTensorNumDims(tensor_1); ++i) {
      if (TfLiteOpaqueTensorDim(tensor_1, i) !=
          TfLiteOpaqueTensorDim(tensor_2, i)) {
        return false;
      }
    }
  }

  return true;
}

TfLiteStatus SampleStableDelegate::Initialize(TfLiteOpaqueContext* context) {
  if (!has_been_initialized_) {
    PrepareControlFlow(context);
    has_been_initialized_ = true;
  }
  return kTfLiteOk;
}

const char* SampleStableDelegate::Name() const {
  return kSampleStableDelegateName;
}

std::unique_ptr<SimpleOpaqueDelegateKernelInterface>
SampleStableDelegate::CreateDelegateKernelInterface() {
  return std::make_unique<SampleStableDelegateKernel>();
}

}  // namespace example
}  // namespace tflite
