#include "tensorflow/lite/delegates/utils/mw_delegates/fully_connected_delegate/fully_connected_delegate.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
// #include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/compiler/mlir/lite/core/c/builtin_op_data.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"


#include "tensorflow/lite/delegates/utils/mw_delegates/fully_connected_delegate/fully_connected_drivers/fully_connected_bram_driver.h"
#include "tensorflow/lite/delegates/utils/mw_delegates/fully_connected_delegate/fully_connected_drivers/fully_connected_ip_driver.h"



namespace tflite {
namespace fully_connected {

// FullyConnectedDelegateKernel implements the interface of SimpleDelegateKernelInterface.
// This holds the Delegate capabilities.
class FullyConnectedDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit FullyConnectedDelegateKernel(const FullyConnectedDelegateOptions& options)
      : options_(options), fpga_ip_driver_(std::make_unique<FpgaIpDriver>()), fpga_bram_driver_(std::make_unique<FullyConnectedBRAMDriver>()) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode* delegated_node = nullptr;
      TfLiteRegistration* delegated_node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
          context,
          context->GetNodeAndRegistration(context, node_index, &delegated_node,
                                          &delegated_node_registration),
          kTfLiteOk);
      inputs_[i].push_back(delegated_node->inputs->data[0]);
      inputs_[i].push_back(delegated_node->inputs->data[1]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
    }
    return !options_.error_during_init ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
  // Check inputs
  for (int i = 0; i < node->inputs->size; ++i) {
    TfLiteTensor& tensor = context->tensors[node->inputs->data[i]];
    if (tensor.allocation_type == kTfLiteDynamic) {
      TF_LITE_KERNEL_LOG(context, "Prepare failed: dynamic input tensor #%d", node->inputs->data[i]);
      return kTfLiteError;
    }
  }

  // Check outputs
  for (int i = 0; i < node->outputs->size; ++i) {
    TfLiteTensor& tensor = context->tensors[node->outputs->data[i]];
    if (tensor.allocation_type == kTfLiteDynamic) {
      TF_LITE_KERNEL_LOG(context, "Prepare failed: dynamic output tensor #%d", node->outputs->data[i]);
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
} 

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    // Evaluate the delegated graph.
    // Here we loop over all the delegated nodes.
    // We know that all the nodes are either ADD or SUB operations and the
    // number of nodes equals ''inputs_.size()'' and inputs[i] is a list of
    // tensor indices for inputs to node ''i'', while outputs_[i] is the list of
    // outputs for node
    // ''i''. Note, that it is intentional we have simple implementation as this
    // is for demonstration.

    for (int i = 0; i < inputs_.size(); ++i) {
      // Get the node input tensors.
      // Add/Sub operation accepts 2 inputs.
      auto& input_tensor_1 = context->tensors[inputs_[i][0]];
      auto& input_tensor_2 = context->tensors[inputs_[i][1]];
      auto& output_tensor = context->tensors[outputs_[i][0]];
      TF_LITE_ENSURE_EQ(
          context,
          ComputeResult(context, builtin_code_[i], &input_tensor_1,
                        &input_tensor_2, &output_tensor),
          kTfLiteOk);
    }

    return !options_.error_during_invoke ? kTfLiteOk : kTfLiteError;
  }

 private:
  const FullyConnectedDelegateOptions options_;
  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_;
  std::unique_ptr<FpgaIpDriver> fpga_driver_;  // FPGA driver instance

  TfLiteStatus ComputeResult(TfLiteContext* context, int builtin_code,
                             const TfLiteTensor* input_tensor_1,
                             const TfLiteTensor* input_tensor_2,
                             TfLiteTensor* output_tensor) {
    if (NumElements(input_tensor_1->dims) != NumElements(input_tensor_2->dims) ||
        NumElements(input_tensor_1->dims) != NumElements(output_tensor->dims)) {
      TF_LITE_KERNEL_LOG(context, "Input and output tensors must have the same size. In FullyConnectedDelegateKernel::ComputeResult");   
      return kTfLiteDelegateError;
    }
    // This code assumes no activation, and no broadcasting needed (both inputs
    // have the same size).
  
    auto* input_1 = GetTensorData<int32>(input_tensor_1);
    auto* input_2 = GetTensorData<int32>(input_tensor_2);
    auto* output = GetTensorData<int32>(output_tensor);
    
    if (!fpga_driver_) {
      return kTfLiteDelegateError;
    }
    
    int num_elements = NumElements(input_tensor_1->dims);
    
    // Simple example assumes 1 element per call; 
    // You may need to extend this for batch processing by looping or modifying your FPGA IP.
    for (int i = 0; i < num_elements; ++i) {
      bool add_flag = (builtin_code == kTfLiteBuiltinAdd);
      int32_t result = fpga_driver_->fpga_compute(input_1[i], input_2[i], add_flag);
      output[i] = result;
    }
    return kTfLiteOk;
  }

  int NumElements(const TfLiteIntArray* dims) {
    int count = 1;
    for (int i = 0; i < dims->size; ++i) {
      count *= dims->data[i];
    }
    return count;
  }
};

class FullyConnectedDelegate : public SimpleDelegateInterface {
 public:
  explicit FullyConnectedDelegate(const FullyConnectedDelegateOptions& options)
      : options_(options) {}

  // This method is called by the TFLite runtime to check if the delegate
  // supports a specific node. The delegate can return true if it supports, false otherwise.
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {

    TF_LITE_KERNEL_LOG(context, "Registering op with builtin_code: %d", registration->builtin_code);

    // This delegate supports only FULLY_CONNECTED operations.
    if (registration->builtin_code != kTfLiteBuiltinFullyConnected)
      return false;
    
    // only supports RELU and NONE activations.
    const TfLiteFullyConnectedParams* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
    if (params->activation != kTfLiteActNone && params->activation != kTfLiteActRelu) return false;
    
    // Debug logging to show the activation type.
    TF_LITE_KERNEL_LOG(context, "Activation type: %d", params->activation);

    // kTFLiteBuiltinFullyConnected input index are fixed:
    // 0 - input tensor, 1 - weights tensor, 2 - bias tensor (Bias is optional).
    // Output tensor is always at index 0.
    const TfLiteTensor* input = GetInput(context, node, 0);
    const TfLiteTensor* weights = GetInput(context, node, 1);
    const TfLiteTensor* bias = node->inputs->size > 2 ? GetInput(context, node, 2) : nullptr;
    const TfLiteTensor* output = GetOutput(context, node, 0);
    
    // Debug logging to show the shapes of input, weights, and output tensors.
    TF_LITE_KERNEL_LOG(context, 
    "FullyConnectedDelegate: Input shape = [%d x %d], Weights shape = [%d x %d], Output shape = [%d x %d]",
    input->dims->data[0], input->dims->data[1],
    weights->dims->data[0], weights->dims->data[1],
    output->dims->data[0], output->dims->data[1]);

    // Check if input, weights, and output tensors are of rank 2.
    if (input->dims->size != 2 || weights->dims->size != 2 || output->dims->size != 2) {
      return false;
    }

    // Check if input, weights, and output tensors are of max size 32x32.
    // FPGA IP is designed to handle max 32x32 matrices.
    if (input->dims->data[1] > 32 || weights->dims->data[0] > 32 || 
      weights->dims->data[1] > 32 || output->dims->data[1] > 32) return false;

    // Debug logging to show the types of input, weights, bias, and output tensors.
    TF_LITE_KERNEL_LOG(context,
    "Input type: %d, Weights type: %d, Bias type: %d, Output type: %d",
    input->type, weights->type, bias ? bias->type : -1, output->type);
    
    // Check if input, weights, and output tensors are of type int32.
    return input->type == kTfLiteInt32 && weights->type == kTfLiteInt32 &&
         (!bias || bias->type == kTfLiteInt32) && output->type == kTfLiteInt32;
}

  // Initialises the delegate. For FPGA, loading the FPGA bitstream can be triggered here.
  // This method is called by the TFLite runtime before any node evaluation.
  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "FullyConnectedDelegate";
    // This name is used for debugging/logging/profiling.
    // It is important to use a unique name for each delegate.
    // If multiple delegates have the same name, it can lead to confusion in
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<FullyConnectedDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const FullyConnectedDelegateOptions options_;

};

}  // namespace fully_connected
}  // namespace tflite

FullyConnectedDelegateOptions TfLiteFullyConnectedDelegateOptionsDefault() {
  FullyConnectedDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this dummy test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteFullyConnectedDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteFullyConnectedDelegateCreate(const FullyConnectedDelegateOptions* options) {
  std::unique_ptr<tflite::fully_connected::FullyConnectedDelegate> fully_connected_delegate(
      new tflite::fully_connected::FullyConnectedDelegate(
          options ? *options : TfLiteFullyConnectedDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(fully_connected_delegate));
}

// Destroys a delegate created with `TfLiteFullyConnectedDelegateCreate` call.
void TfLiteFullyConnectedDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}



