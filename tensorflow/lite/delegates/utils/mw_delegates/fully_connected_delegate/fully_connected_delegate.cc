#include "tensorflow/lite/delegates/utils/mw_delegates/fully_connected_delegate/fully_connected_delegate.h"

#include <algorithm>
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
      : options_(options), fpga_ip_driver_(std::make_unique<FpgaIpDriver>()), fpga_bram_driver_(std::make_unique<FpgaBramDriver>()) {}


  // Lifecycle methods for the delegate kernel.
  // Init is called once per delegate, and is used to initialize the delegate
  // with the parameters provided in TfLiteDelegateParams.
  // It is called during model initialization and delegate planning.
  // It should not be used for any heavy computation or resource allocation.
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {

  // Currently assuming one node per delegate. So only one FullyConnected node
  // is replaced by the delegate. 
  //TODO: Support multiple nodes in the future. By looping over params->nodes_to_replace->data
  const int node_index = params->nodes_to_replace->data[0];

  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  TF_LITE_ENSURE_EQ(context,
                    context->GetNodeAndRegistration(context, node_index, &node, &registration),
                    kTfLiteOk);

  input_index_ = node->inputs->data[0];
  weights_index_ = node->inputs->data[1];
  has_bias_ = node->inputs->size > 2;
  if (has_bias_) bias_index_ = node->inputs->data[2];
  output_index_ = node->outputs->data[0];

  activation_type_ =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data)->activation;

  try {
    bool clear_bram = false;
      fpga_bram_driver_->initialize_bram(clear_bram);
  } catch (const std::exception& e) {
      TF_LITE_KERNEL_LOG(context, "BRAM initialization failed: %s", e.what());
      return kTfLiteError;
  }

  return kTfLiteOk;
}

  // Prepare is called before the delegate is invoked.
  // It is called once per node(one instance of the delegate, i.e one FullyConnected operation).
  // Used to validate input and output memory, Allocate BRAMS and Pre-load weights and biases as they are static for the given node.
  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
  const TfLiteTensor& input_tensor = context->tensors[input_index_];
  const TfLiteTensor& weights_tensor = context->tensors[weights_index_];
  const TfLiteTensor& output_tensor = context->tensors[output_index_];

  // Additional safety checks for dynamic tensors
  if (input_tensor.dims == nullptr || weights_tensor.dims == nullptr || output_tensor.dims == nullptr) {
    TF_LITE_KERNEL_LOG(context, "Prepare failed: null tensor dimensions detected.");
    return kTfLiteError;
  }

  // Check for dynamic dimensions
  for (int i = 0; i < input_tensor.dims->size; ++i) {
    if (input_tensor.dims->data[i] < 0) {
      TF_LITE_KERNEL_LOG(context, "Prepare failed: dynamic input tensor dimension detected.");
      return kTfLiteError;
    }
  }
  for (int i = 0; i < weights_tensor.dims->size; ++i) {
    if (weights_tensor.dims->data[i] < 0) {
      TF_LITE_KERNEL_LOG(context, "Prepare failed: dynamic weights tensor dimension detected.");
      return kTfLiteError;
    }
  }
  for (int i = 0; i < output_tensor.dims->size; ++i) {
    if (output_tensor.dims->data[i] < 0) {
      TF_LITE_KERNEL_LOG(context, "Prepare failed: dynamic output tensor dimension detected.");
      return kTfLiteError;
    }
  }

  if (input_tensor.allocation_type == kTfLiteDynamic ||
      weights_tensor.allocation_type == kTfLiteDynamic ||
      output_tensor.allocation_type == kTfLiteDynamic) {
    TF_LITE_KERNEL_LOG(context, "Prepare failed: dynamic tensors are not supported.");
    return kTfLiteError;
  }

  // Write weights to BRAM (INT32 only)
  if (weights_tensor.type == kTfLiteInt32) {
    if (fpga_bram_driver_->write_weights_to_bram(weights_tensor.data.i32, NumElements(weights_tensor.dims))) {
      TF_LITE_KERNEL_LOG(context, "Prepare failed: failed to write INT32 weights to BRAM.");
      return kTfLiteError;
    }
  } else {
    TF_LITE_KERNEL_LOG(context, "Prepare failed: only INT32 weights supported. Detected type: %d", weights_tensor.type);
    return kTfLiteError;
  }

  // Write bias if available
  if (has_bias_) {
    const TfLiteTensor& bias_tensor = context->tensors[bias_index_];
    if (bias_tensor.allocation_type == kTfLiteDynamic) {
      TF_LITE_KERNEL_LOG(context, "Prepare failed: dynamic bias tensor.");
      return kTfLiteError;
    }

    // Handle INT32 bias tensors only
    if (bias_tensor.type == kTfLiteInt32) {
      if (fpga_bram_driver_->write_bias_to_bram(bias_tensor.data.i32, NumElements(bias_tensor.dims))) {
        TF_LITE_KERNEL_LOG(context, "Prepare failed: failed to write INT32 bias to BRAM.");
        return kTfLiteError;
      }
    } else {
      TF_LITE_KERNEL_LOG(context, "Prepare failed: only INT32 bias supported. Detected type: %d", bias_tensor.type);
      return kTfLiteError;
    }

    TF_LITE_KERNEL_LOG(context, "Weights and bias successfully written to BRAM.");

  }

  // // Optional: clear output BRAM
  // fpga_driver_->clear_output_bram();

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

    const TfLiteTensor& input_tensor = context->tensors[input_index_];
    TfLiteTensor& output_tensor = context->tensors[output_index_];

    // Safety check for dynamic tensors during evaluation
    if (input_tensor.dims == nullptr || output_tensor.dims == nullptr) {
      TF_LITE_KERNEL_LOG(context, "Eval failed: null tensor dimensions detected.");
      return kTfLiteError;
    }

    // Check for dynamic dimensions
    for (int i = 0; i < input_tensor.dims->size; ++i) {
      if (input_tensor.dims->data[i] < 0) {
        TF_LITE_KERNEL_LOG(context, "Eval failed: dynamic input tensor dimension detected.");
        return kTfLiteError;
      }
    }
    for (int i = 0; i < output_tensor.dims->size; ++i) {
      if (output_tensor.dims->data[i] < 0) {
        TF_LITE_KERNEL_LOG(context, "Eval failed: dynamic output tensor dimension detected.");
        return kTfLiteError;
      }
    }

    const int input_size = NumElements(input_tensor.dims);
    const int output_size = NumElements(output_tensor.dims);

    // Additional safety check for zero-sized tensors
    if (input_size <= 0 || output_size <= 0) {
      TF_LITE_KERNEL_LOG(context, "Eval failed: invalid tensor sizes (input: %d, output: %d).", input_size, output_size);
      return kTfLiteError;
    }

    // Write input tensor to input BRAM (INT32 only)
    if (input_tensor.type == kTfLiteInt32) {
      if (fpga_bram_driver_->write_input_to_bram(input_tensor.data.i32, input_size)) {
        TF_LITE_KERNEL_LOG(context, "Eval failed: failed to write INT32 input to BRAM.");
        return kTfLiteError;
      }
    } else {
      TF_LITE_KERNEL_LOG(context, "Eval failed: only INT32 input supported. Detected type: %d", input_tensor.type);
      return kTfLiteError;
    }

    // Trigger FPGA execution
    if (fpga_ip_driver_->fpga_compute(input_size, output_size)) {
      TF_LITE_KERNEL_LOG(context, "Eval failed: FPGA inference trigger failed.");
      return kTfLiteError;
    }

    // Read output from output BRAM into output tensor (INT32 only)
    if (output_tensor.type == kTfLiteInt32) {
      if (fpga_bram_driver_->read_output_from_bram(output_tensor.data.i32, output_size)) {
        TF_LITE_KERNEL_LOG(context, "Eval failed: failed to read INT32 output from BRAM.");
        return kTfLiteError;
      }
    } else {
      TF_LITE_KERNEL_LOG(context, "Eval failed: only INT32 output supported. Detected type: %d", output_tensor.type);
      return kTfLiteError;
    }

    return kTfLiteOk;
}

 private:
  const FullyConnectedDelegateOptions options_;
  int input_index_, weights_index_, bias_index_, output_index_;
  bool has_bias_ = false;
  TfLiteFusedActivation activation_type_ = kTfLiteActNone;


  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_;
  std::unique_ptr<FpgaIpDriver> fpga_ip_driver_;  // FPGA driver instance
  std::unique_ptr<FpgaBramDriver> fpga_bram_driver_;  // BRAM driver instance

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

    TF_LITE_KERNEL_LOG(context, "=== FullyConnectedDelegate: Checking node support ===");
    TF_LITE_KERNEL_LOG(context, "Node builtin_code: %d (kTfLiteBuiltinFullyConnected = %d)", 
                       registration->builtin_code, kTfLiteBuiltinFullyConnected);

    // This delegate supports only FULLY_CONNECTED operations.
    if (registration->builtin_code != kTfLiteBuiltinFullyConnected) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: not a fully connected operation.");
      return false;
    }
    
    // only supports RELU and NONE activations.
    const TfLiteFullyConnectedParams* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
    if (params->activation != kTfLiteActNone && params->activation != kTfLiteActRelu) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: unsupported activation type %d.", params->activation);
      return false;
    }
    
    // Debug logging to show the activation type.
    TF_LITE_KERNEL_LOG(context, "Activation type: %d", params->activation);

    // kTFLiteBuiltinFullyConnected input index are fixed:
    // 0 - input tensor, 1 - weights tensor, 2 - bias tensor (Bias is optional).
    // Output tensor is always at index 0.
    const TfLiteTensor* input = GetInput(context, node, 0);
    const TfLiteTensor* weights = GetInput(context, node, 1);
    const TfLiteTensor* bias = node->inputs->size > 2 ? GetInput(context, node, 2) : nullptr;
    const TfLiteTensor* output = GetOutput(context, node, 0);
    
    if (!input || !weights || !output) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: null tensor pointers detected.");
      return false;
    }

    // Log tensor information for debugging
    TF_LITE_KERNEL_LOG(context, "Tensor info - Input: dims=%p, allocation=%d", 
                       input->dims, input->allocation_type);
    TF_LITE_KERNEL_LOG(context, "Tensor info - Weights: dims=%p, allocation=%d", 
                       weights->dims, weights->allocation_type);
    TF_LITE_KERNEL_LOG(context, "Tensor info - Output: dims=%p, allocation=%d", 
                       output->dims, output->allocation_type);
    if (bias) {
      TF_LITE_KERNEL_LOG(context, "Tensor info - Bias: dims=%p, allocation=%d", 
                         bias->dims, bias->allocation_type);
    }
    
    // Reject dynamic tensors - check for unknown dimensions
    if (input->dims == nullptr || weights->dims == nullptr || output->dims == nullptr ||
        (bias && bias->dims == nullptr)) {
      TF_LITE_KERNEL_LOG(context, "Null tensor dimensions detected — rejecting.");
      return false;
    }

    // Check for dynamic dimensions (negative values indicate dynamic dimensions)
    bool has_dynamic_dims = false;
    for (int i = 0; i < input->dims->size; ++i) {
      if (input->dims->data[i] < 0) {
        TF_LITE_KERNEL_LOG(context, "Dynamic input tensor dimension detected at index %d: %d", 
                           i, input->dims->data[i]);
        has_dynamic_dims = true;
      }
    }
    for (int i = 0; i < weights->dims->size; ++i) {
      if (weights->dims->data[i] < 0) {
        TF_LITE_KERNEL_LOG(context, "Dynamic weights tensor dimension detected at index %d: %d", 
                           i, weights->dims->data[i]);
        has_dynamic_dims = true;
      }
    }
    for (int i = 0; i < output->dims->size; ++i) {
      if (output->dims->data[i] < 0) {
        TF_LITE_KERNEL_LOG(context, "Dynamic output tensor dimension detected at index %d: %d", 
                           i, output->dims->data[i]);
        has_dynamic_dims = true;
      }
    }
    if (bias) {
      for (int i = 0; i < bias->dims->size; ++i) {
        if (bias->dims->data[i] < 0) {
          TF_LITE_KERNEL_LOG(context, "Dynamic bias tensor dimension detected at index %d: %d", 
                             i, bias->dims->data[i]);
          has_dynamic_dims = true;
        }
      }
    }

    if (has_dynamic_dims) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: dynamic tensor dimensions detected.");
      return false;
    }

    // Also check allocation types as additional safety
    if (input->allocation_type == kTfLiteDynamic ||
        weights->allocation_type == kTfLiteDynamic ||
        output->allocation_type == kTfLiteDynamic ||
        (bias && bias->allocation_type == kTfLiteDynamic)) {
      TF_LITE_KERNEL_LOG(context, "Dynamic tensor allocation detected — rejecting.");
      return false;
    }


    // Debug logging to show the shapes of input, weights, and output tensors.
    TF_LITE_KERNEL_LOG(context, 
    "FullyConnectedDelegate: Input shape = [%d x %d], Weights shape = [%d x %d], Output shape = [%d x %d]",
    input->dims->data[0], input->dims->data[1],
    weights->dims->data[0], weights->dims->data[1],
    output->dims->data[0], output->dims->data[1]);

    // Check if input, weights, and output tensors are of rank 2.
    if (input->dims->size != 2 || weights->dims->size != 2 || output->dims->size != 2) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: tensor rank != 2 (input: %d, weights: %d, output: %d).",
                        input->dims->size, weights->dims->size, output->dims->size);
      return false;
    }

    // Check if input, weights, and output tensors are of max size 32x32.
    // FPGA IP is designed to handle max 32x32 matrices.
    if (input->dims->data[1] > 32 || weights->dims->data[0] > 32 || 
      weights->dims->data[1] > 32 || output->dims->data[1] > 32) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: tensor dimensions exceed 32x32 limit.");
      return false;
    }

    // Debug logging to show the types of input, weights, bias, and output tensors.
    TF_LITE_KERNEL_LOG(context,
    "Input type: %d, Weights type: %d, Bias type: %d, Output type: %d",
    input->type, weights->type, bias ? bias->type : -1, output->type);
    
    // Check if input, weights, and output tensors are of type INT32 only.
    // This delegate only supports INT32 tensors for now.
    bool type_check = (input->type == kTfLiteInt32 && weights->type == kTfLiteInt32 &&
                       (!bias || bias->type == kTfLiteInt32) && output->type == kTfLiteInt32);
    
    if (type_check) {
      TF_LITE_KERNEL_LOG(context, "Tensor types: INT32 (supported)");
    } else {
      TF_LITE_KERNEL_LOG(context, "Tensor types: Only INT32 tensors supported. Detected types - Input: %d, Weights: %d, Bias: %d, Output: %d",
                         input->type, weights->type, bias ? bias->type : -1, output->type);
      TF_LITE_KERNEL_LOG(context, "  INT8 support will be added later when low-level drivers are ready.");
    }
    
    if (!type_check) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: unsupported tensor types.");
      return false;
    }
    
    TF_LITE_KERNEL_LOG(context, "Node accepted: fully connected operation meets all requirements.");
    TF_LITE_KERNEL_LOG(context, "=== FullyConnectedDelegate: Node support check complete ===");
    return true;
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
    // Configure delegate to handle graphs with dynamic tensors
    // This allows the delegate to be applied even when the graph contains
    // dynamic tensors (e.g., from LSTM operations), enabling selective
    // node support for static fully connected layers
    SimpleDelegateInterface::Options options;
    options.max_delegated_partitions = 100;  // Allow multiple partitions
    options.min_nodes_per_partition = 1;     // Allow single-node partitions
    // This is the key setting - allow dynamic tensors
    options.allow_dynamic_tensors = true;
    return options;
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



