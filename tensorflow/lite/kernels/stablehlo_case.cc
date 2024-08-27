#include <numeric>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/control_flow_common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/tensor_slice_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_case {

struct OpData {
  std::vector<int> subgraph_indices;
  bool subgraph_has_dynamic_output_tensors;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  const auto* params =
      reinterpret_cast<const TfLiteStablehloCaseParams*>(buffer);
  op_data->subgraph_indices.assign(
      params->branch_subgraph_indices,
      params->branch_subgraph_indices + params->num_branches);
  op_data->subgraph_has_dynamic_output_tensors = false;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const auto* params =
      reinterpret_cast<const TfLiteStablehloCaseParams*>(node->builtin_data);
  TF_LITE_ENSURE(context, params->num_branches > 0);
  const TfLiteTensor* index;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &index));
  TF_LITE_ENSURE_EQ(context, index->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumElements(index), 1);

  int num_inputs = node->inputs->size - 1;
  int num_outputs = node->outputs->size;

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  for (size_t i = 1; i < subgraphs->size(); ++i) {
    Subgraph* subgraph = (*subgraphs)[i].get();
    TF_LITE_ENSURE_EQ(context, num_inputs, subgraph->inputs().size());
    TF_LITE_ENSURE_EQ(context, num_outputs, subgraph->outputs().size());
  }

  for (size_t i = 1; i < subgraphs->size(); ++i) {
    Subgraph* subgraph = (*subgraphs)[i].get();
    subgraph->RemoveUnusedInputs();
  }

  const int* const start = node->inputs->data + 1;
  std::vector<int> node_inputs(start, start + num_inputs);
  // Prepare and check the subgraphs.
  for (size_t i = 1; i < subgraphs->size(); ++i) {
    Subgraph* subgraph = (*subgraphs)[i].get();
    TF_LITE_ENSURE_OK(
        context, CopyTensorsShapeAndType(context, this_subgraph, node_inputs,
                                         subgraph, subgraph->inputs(), true));
  }
  for (size_t k = 1; k < subgraphs->size(); ++k) {
    Subgraph* subgraph = (*subgraphs)[k].get();
    for (int i = 0; i < num_inputs; ++i) {
      int input_idx = subgraph->inputs()[i];
      if (input_idx == kTfLiteOptionalTensor) continue;
      TfLiteTensor* subgraph_input = subgraph->tensor(input_idx);
      if (!IsResourceOrVariant(subgraph_input)) {
        // Set the allocation type to custom to prevent memory allocation.
        subgraph_input->allocation_type = kTfLiteCustom;
      }
    }
    TF_LITE_ENSURE_OK(context, subgraph->AllocateTensors());
    op_data->subgraph_has_dynamic_output_tensors |=
        subgraph->HasDynamicTensors();
  }

  // Check if any subgraph outputs have dynamic shapes
  if (!op_data->subgraph_has_dynamic_output_tensors) {
    // Iterate over all subgraphs to compare output shapes
    for (size_t j = 1; j < subgraphs->size() - 1; ++j) {
      Subgraph* branch_subgraph =
          (*subgraphs)[op_data->subgraph_indices[j]].get();

      for (int i = 0; i < num_outputs; ++i) {
        TfLiteTensor* branch_output =
            branch_subgraph->tensor(branch_subgraph->outputs()[i]);

        // Check against the first subgraph (reference)
        TfLiteTensor* reference_output =
            (*subgraphs)[op_data->subgraph_indices[0]].get()->tensor(
                (*subgraphs)[op_data->subgraph_indices[0]]->outputs()[i]);

        if (!TfLiteIntArrayEqual(reference_output->dims, branch_output->dims)) {
          op_data->subgraph_has_dynamic_output_tensors = true;
          break;
        }
      }

      if (op_data->subgraph_has_dynamic_output_tensors) {
        break;
      }
    }
  }

  // Resize the output tensors based on whether dynamic shapes are present
  for (int i = 0; i < num_outputs; ++i) {
    if (node->outputs->data[i] == kTfLiteOptionalTensor) continue;

    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));

    if (op_data->subgraph_has_dynamic_output_tensors) {
      SetTensorToDynamic(output);
    } else {
      // Use the dimensions from the reference subgraph
      TfLiteTensor* reference_output =
          (*subgraphs)[op_data->subgraph_indices[0]].get()->tensor(
              (*subgraphs)[op_data->subgraph_indices[0]]->outputs()[i]);
      TfLiteIntArray* output_size = TfLiteIntArrayCopy(reference_output->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output, output_size));
    }
  }

  return kTfLiteOk;
}

// Evaluate CASE op when subgraphs have dynamic outputs.
TfLiteStatus Eval_dynamic(TfLiteContext* context, TfLiteNode* node,
                          Subgraph* active_branch_subgraph) {
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);

  TF_LITE_ENSURE_OK(context, active_branch_subgraph->AllocateTensors());
  const int num_inputs = node->inputs->size - 1;
  const int num_outputs = node->outputs->size;
  const int* const start = node->inputs->data + 1;
  std::vector<int> node_inputs(start, start + num_inputs);
  // node->inputs -> subgraph->inputs
  TF_LITE_ENSURE_OK(
      context, DeepOrShallowCopyTensorsShapeTypeData(
                   context, node, this_subgraph, node_inputs,
                   active_branch_subgraph, active_branch_subgraph->inputs()));

  // Invoke active_branch_subgraph subgraph
  TF_LITE_ENSURE_OK(context, active_branch_subgraph->Invoke());
  for (int tensor_index : active_branch_subgraph->outputs()) {
    active_branch_subgraph->EnsureTensorDataIsReadable(tensor_index);
  }

  // subgraph->outputs -> node->outputs
  TF_LITE_ENSURE_OK(context,
                    DeepCopyTensorsShapeTypeData(
                        context, node, active_branch_subgraph,
                        active_branch_subgraph->outputs(), this_subgraph,
                        TfLiteIntArrayView(node->outputs), true));

  for (int i = 0; i < num_outputs; ++i) {
    const int input_pos = OutputIsInput(active_branch_subgraph->outputs()[i],
                                        active_branch_subgraph->inputs());
    if (input_pos != -1) {
      TfLiteTensor* this_input =
          this_subgraph->tensor(node->inputs->data[input_pos + 1]);
      TfLiteTensor* this_output = this_subgraph->tensor(node->outputs->data[i]);
      TfLiteTensorCopy(this_input, this_output);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval_static(TfLiteContext* context, TfLiteNode* node,
                         Subgraph* active_branch_subgraph) {
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);

  const int num_inputs = node->inputs->size - 1;
  const int num_outputs = node->outputs->size;
  const int* const start = node->inputs->data + 1;
  std::vector<int> node_inputs(start, start + num_inputs);
  for (int i = 0; i < num_outputs; ++i) {
    int output_idx = active_branch_subgraph->outputs()[i];
    if (output_idx == kTfLiteOptionalTensor) continue;
    TfLiteTensor* subgraph_output = active_branch_subgraph->tensor(output_idx);
    if (!IsResourceOrVariant(subgraph_output) &&
        !IsConstantTensor(subgraph_output)) {
      subgraph_output->allocation_type = kTfLiteCustom;
    }
  }
  // node->inputs -> subgraph->inputs
  TF_LITE_ENSURE_OK(
      context, DeepOrShallowCopyTensorsShapeTypeData(
                   context, node, this_subgraph, node_inputs,
                   active_branch_subgraph, active_branch_subgraph->inputs()));

  TF_LITE_ENSURE_OK(
      context,
      CopyTensorsShapeAndType(context, active_branch_subgraph,
                              active_branch_subgraph->outputs(), this_subgraph,
                              TfLiteIntArrayView(node->outputs), false));
  for (int i = 0; i < num_outputs; ++i) {
    TfLiteTensor* this_output = this_subgraph->tensor(node->outputs->data[i]);
    TfLiteTensor* subgraph_output =
        active_branch_subgraph->tensor(active_branch_subgraph->outputs()[i]);
    if (active_branch_subgraph->outputs()[i] == kTfLiteOptionalTensor) {
      TfLiteTensor* this_input =
          this_subgraph->tensor(node->inputs->data[i + 1]);
      TfLiteTensorResizeMaybeCopy(this_input->bytes, this_output, false);
      TfLiteTensorCopy(this_input, this_output);
    } else {
      const int input_pos = OutputIsInput(active_branch_subgraph->outputs()[i],
                                          active_branch_subgraph->inputs());
      if (input_pos != -1) {
        TfLiteTensor* this_input =
            this_subgraph->tensor(node->inputs->data[input_pos + 1]);
        TfLiteTensorResizeMaybeCopy(this_input->bytes, this_output, false);
        TfLiteTensorCopy(this_input, this_output);
      } else if (IsConstantTensor(subgraph_output)) {
        TfLiteTensorCopy(subgraph_output, this_output);
      } else {
        subgraph_output->data = this_output->data;
      }
    }
  }

  // Invoke subgraph
  TF_LITE_ENSURE_OK(context, active_branch_subgraph->Invoke());
  for (int tensor_index : active_branch_subgraph->outputs()) {
    active_branch_subgraph->EnsureTensorDataIsReadable(tensor_index);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();

  const TfLiteTensor* index_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &index_tensor));
  TF_LITE_ENSURE_EQ(context, index_tensor->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumElements(index_tensor), 1);

  int32_t index_value = index_tensor->data.i32[0];
  if (index_value < 0 || index_value >= op_data->subgraph_indices.size()) {
    index_value = op_data->subgraph_indices.size() - 1;
  }

  int selected_subgraph_index = op_data->subgraph_indices[index_value];
  TF_LITE_ENSURE(context, selected_subgraph_index < subgraphs->size());
  Subgraph& selected_subgraph = *(*subgraphs)[selected_subgraph_index].get();
  TF_LITE_ENSURE_OK(context, selected_subgraph.AllocateTensors());
  if (op_data->subgraph_has_dynamic_output_tensors) {
    TF_LITE_ENSURE_OK(context, Eval_dynamic(context, node, &selected_subgraph));
  } else {
    TF_LITE_ENSURE_OK(context, Eval_static(context, node, &selected_subgraph));
  }
  if (!this_subgraph->ShouldPreserveAllTensors()) {
    TF_LITE_ENSURE_OK(context, selected_subgraph.ReleaseMemory());
  }
  return kTfLiteOk;
}

}  // namespace stablehlo_case

TfLiteRegistration* Register_STABLEHLO_CASE() {
  static TfLiteRegistration r = {stablehlo_case::Init, stablehlo_case::Free,
                                 stablehlo_case::Prepare, stablehlo_case::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
