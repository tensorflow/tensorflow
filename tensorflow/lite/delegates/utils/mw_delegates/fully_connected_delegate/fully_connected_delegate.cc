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

  TF_LITE_KERNEL_LOG(context, "======== IN FullyConnectedDelegateKernel::Init =========\n");
  
  inputs_.resize(params->nodes_to_replace->size);
  outputs_.resize(params->nodes_to_replace->size);
  weights_.resize(params->nodes_to_replace->size);
  biases_.resize(params->nodes_to_replace->size);
  builtin_code_.resize(params->nodes_to_replace->size);

  for(int i = 0; i < params->nodes_to_replace->size; ++i) {
    const int node_index = params->nodes_to_replace->data[i];
    //Get this node information
    TfLiteNode* delegate_node = nullptr;
    TfLiteRegistration* delegate_node_registration = nullptr;
    TF_LITE_ENSURE_EQ(context,
                      context->GetNodeAndRegistration(context, node_index, &delegate_node, &delegate_node_registration),
                      kTfLiteOk);
    inputs_[i].push_back(delegate_node->inputs->data[0]);
    weights_[i].push_back(delegate_node->inputs->data[1]);
  
    if (delegate_node->inputs->size > 2) {
      biases_[i].push_back(delegate_node->inputs->data[2]);
    } else {
      biases_[i].push_back(-1);  // No bias for this node
    }
    outputs_[i].push_back(delegate_node->outputs->data[0]);
    builtin_code_[i] = delegate_node_registration->builtin_code;
    
    // Add debug logging
    TF_LITE_KERNEL_LOG(context, "Node %d Input_Tidx=%d; Weights_Tidx=%d; Bias_Tidx=%d; output_Tidx=%d\n",
                       i, inputs_[i][0], weights_[i][0], biases_[i][0], outputs_[i][0]);
  }
  // Initialize FPGA drivers once for all nodes
  try {
    // Note: FPGA IP driver is already initialized in constructor
    bool clear_bram = false;
    fpga_bram_driver_->initialize_bram(clear_bram);
  } catch (const std::exception& e) {
    TF_LITE_KERNEL_LOG(context, "BRAM initialization failed: %s\n", e.what());
    return kTfLiteError;
  }
  
  TF_LITE_KERNEL_LOG(context, "======== FullyConnectedDelegateKernel::Init completed successfully =========\n\n\n");
  return kTfLiteOk;
}

  // Prepare is called before Eval, and is used to prepare the delegate for
  // execution. It is called once per node in the partition.
  // It should not be used for any heavy computation or resource allocation.
  // It should only be used to prepare the delegate for execution.
  // It is called after Init and before Eval.
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
  
  TF_LITE_KERNEL_LOG(context, "======== IN FullyConnectedDelegateKernel::Prepare =========\n");
  TF_LITE_KERNEL_LOG(context, "Preparing %d nodes in partition\n\n", inputs_.size());
  
  for(int i = 0; i < inputs_.size(); i++){
    // Safety check for tensor indices
    if (inputs_[i].empty() || weights_[i].empty() || outputs_[i].empty()) {
      TF_LITE_KERNEL_LOG(context, "Error: Empty tensor indices for node %d\n", i);
      return kTfLiteError;
    }

    TF_LITE_KERNEL_LOG(context, "Processing node %d with Input_Tidx=%d, Weights_Tidx=%d, Output_Tidx=%d\n",
                       i, inputs_[i][0], weights_[i][0], outputs_[i][0]);

    // Get tensors for this node using standard TensorFlow Lite API
    const TfLiteTensor* input_tensor = &context->tensors[inputs_[i][0]];
    const TfLiteTensor* weights_tensor = &context->tensors[weights_[i][0]];
    TfLiteTensor* output_tensor = &context->tensors[outputs_[i][0]];
    
    // Check if this node has bias - safer approach
    bool node_has_bias = (!biases_[i].empty() && biases_[i][0] != -1);
    const TfLiteTensor* bias_tensor = nullptr;
    if (node_has_bias) {
      bias_tensor = &context->tensors[biases_[i][0]];
      TF_LITE_KERNEL_LOG(context, "Node %d has bias tensor at index %d\n", i, biases_[i][0]);
    }

    // Write weights to BRAM (FLOAT32 only)
    if (weights_tensor->type == kTfLiteFloat32) {
      if (fpga_bram_driver_->write_weights_to_bram(weights_tensor->data.f, NumElements(weights_tensor->dims))) {
        TF_LITE_KERNEL_LOG(context, "Prepare failed: failed to write FLOAT32 weights to BRAM for node %d.\n", i);
        return kTfLiteError;
      }
      TF_LITE_KERNEL_LOG(context, "FLOAT32 weights written to BRAM for node %d.\n", i);
    } else {
      TF_LITE_KERNEL_LOG(context, "Prepare failed: unsupported weights tensor type: %d. Only FLOAT32 supported.\n", weights_tensor->type);
      return kTfLiteError;
    }

    // Write bias to BRAM(FLOAT32) if available for this node
    if (node_has_bias && bias_tensor) {
      // Handle bias tensors (FLOAT32 only)
      if (bias_tensor->type == kTfLiteFloat32) {
        if (fpga_bram_driver_->write_bias_to_bram(bias_tensor->data.f, NumElements(bias_tensor->dims))) {
          TF_LITE_KERNEL_LOG(context, "Prepare failed: failed to write FLOAT32 bias to BRAM for node %d.\n", i);
          return kTfLiteError;
        }
        TF_LITE_KERNEL_LOG(context, "FLOAT32 bias written to BRAM for node %d.\n", i);
      } else {
        TF_LITE_KERNEL_LOG(context, "Prepare failed: unsupported bias tensor type: %d. Only FLOAT32 supported.\n", bias_tensor->type);
        return kTfLiteError;
      }

      TF_LITE_KERNEL_LOG(context, "Weights and bias successfully written to BRAM for node %d.\n", i);
    } else {
      TF_LITE_KERNEL_LOG(context, "Node %d has no bias tensor.\n", i);
    }

    // Ensure output tensor is properly set up for TensorFlow Lite to allocate
    // Log tensor allocation info
    TF_LITE_KERNEL_LOG(context, "Node %d - Output tensor allocation: data=%p, bytes=%d, allocation_type=%d\n", 
                       i, output_tensor->data.data, output_tensor->bytes, output_tensor->allocation_type);
    
    // // Resize output tensor to ensure TensorFlow Lite allocates it properly
    // TfLiteIntArray* output_shape = TfLiteIntArrayCopy(output_tensor->dims);
    // if (context->ResizeTensor(context, output_tensor, output_shape) != kTfLiteOk) {
    //   TF_LITE_KERNEL_LOG(context, "Failed to resize output tensor for node %d\n", i);
    //   return kTfLiteError;
    // }
    
    TF_LITE_KERNEL_LOG(context, "Node %d - Successfully prepared output tensor for allocation\n", i);

    // Log tensor shapes for debugging
    if (node_has_bias && bias_tensor){
      TF_LITE_KERNEL_LOG(context, "Node %d - Input_Shape: [%d, %d], Weights_Shape: [%d, %d], Bias_Shape: [%d], Output_Shape: [%d, %d]\n\n",
                         i, input_tensor->dims->data[0], input_tensor->dims->data[1],
                         weights_tensor->dims->data[0], weights_tensor->dims->data[1],
                         bias_tensor->dims->data[0],
                         output_tensor->dims->data[0], output_tensor->dims->data[1]);
    } else {
      TF_LITE_KERNEL_LOG(context, "Node %d - Input_Shape: [%d, %d], Weights_Shape: [%d, %d], Output_Shape: [%d, %d]\n\n",
                         i, input_tensor->dims->data[0], input_tensor->dims->data[1],
                         weights_tensor->dims->data[0], weights_tensor->dims->data[1],
                         output_tensor->dims->data[0], output_tensor->dims->data[1]);
    }
  }

  TF_LITE_KERNEL_LOG(context, "======== FullyConnectedDelegateKernel::Prepare completed successfully =========\n\n\n");
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
  TF_LITE_KERNEL_LOG(context, "======== IN FullyConnectedDelegateKernel::Eval =========\n");
  TF_LITE_KERNEL_LOG(context, "Evaluating %d nodes in partition\n", inputs_.size());


  // Process each node in the partition
  for(int i = 0; i < inputs_.size(); i++) {
    // Safety check for tensor indices
    if (inputs_[i].empty() || outputs_[i].empty()) {
      TF_LITE_KERNEL_LOG(context, "Error: Empty tensor indices for node %d\n", i);
      return kTfLiteError;
    }
    
    TF_LITE_KERNEL_LOG(context, "Evaluating node %d with input_Tidx=%d, output_Tidx=%d\n",
                       i, inputs_[i][0], outputs_[i][0]);

    const TfLiteTensor& input_tensor = context->tensors[inputs_[i][0]];
    TfLiteTensor& output_tensor = context->tensors[outputs_[i][0]];

    // Check if input and output tensors are valid
    if (input_tensor.data.raw == nullptr || output_tensor.data.raw == nullptr) {
      TF_LITE_KERNEL_LOG(context, "Eval failed: null tensor data detected for node %d.\n", i);
      return kTfLiteError;
    }

    const int input_size = NumElements(input_tensor.dims);
    const int output_size = NumElements(output_tensor.dims);

    // Additional safety check for zero-sized tensors
    if (input_size <= 0 || output_size <= 0) {
      TF_LITE_KERNEL_LOG(context, "Eval failed: invalid tensor sizes for node %d (input: %d, output: %d).\n", 
                         i, input_size, output_size);
      return kTfLiteError;
    }

    // Debug logging for tensor sizes
    TF_LITE_KERNEL_LOG(context, "Node %d - Tensor sizes - Input: %d, Output: %d\n", i, input_size, output_size);
    TF_LITE_KERNEL_LOG(context, "Node %d - Input tensor shape: [%d, %d]\n", i, input_tensor.dims->data[0], input_tensor.dims->data[1]);
    TF_LITE_KERNEL_LOG(context, "Node %d - Output tensor shape: [%d, %d]\n", i, output_tensor.dims->data[0], output_tensor.dims->data[1]);

    // Extract the actual feature dimensions (ignore batch dimension)
    const int input_features = input_tensor.dims->data[1];  // Features dimension
    const int output_features = output_tensor.dims->data[1]; // Output features dimension
    
    // DEBUG: Force print to console to see what's happening
    printf("[DELEGATE-DEBUG] Node %d - Input tensor shape: [%d, %d]\n", i, input_tensor.dims->data[0], input_tensor.dims->data[1]);
    printf("[DELEGATE-DEBUG] Node %d - Output tensor shape: [%d, %d]\n", i, output_tensor.dims->data[0], output_tensor.dims->data[1]);
    printf("[DELEGATE-DEBUG] Node %d - Feature dimensions - Input: %d, Output: %d\n", i, input_features, output_features);
    fflush(stdout);
    
    TF_LITE_KERNEL_LOG(context, "Node %d - Feature dimensions - Input: %d, Output: %d\n", i, input_features, output_features);

    // Write input tensor to input BRAM (FLOAT32 only)
    if (input_tensor.type == kTfLiteFloat32) {
      // Safety check: ensure input tensor data is allocated
      if (input_tensor.data.f == nullptr) {
        TF_LITE_KERNEL_LOG(context, "Eval failed: input tensor data is null for node %d\n", i);
        return kTfLiteError;
      }
      
      // For batched input, we need to handle each sample in the batch
      const int batch_size = input_tensor.dims->data[0];
      if (batch_size != 1) {
        TF_LITE_KERNEL_LOG(context, "Eval failed: batch size %d not supported for node %d. Only batch size 1 supported.\n", batch_size, i);
        return kTfLiteError;
      }
      
      TF_LITE_KERNEL_LOG(context, "Node %d - Input tensor data address: %p\n", i, input_tensor.data.f);

      
      printf("[DELEGATE-DEBUG] Node %d: input_tensor.data.f address: %p, input_features: %d\n", 
             i, input_tensor.data.f, input_features);
      if (fpga_bram_driver_->write_input_to_bram(input_tensor.data.f, input_features)) {
        TF_LITE_KERNEL_LOG(context, "Eval failed: failed to write FLOAT32 input to BRAM for node %d.\n", i);
        return kTfLiteError;
      }
      
      // DEBUG: Print what we're passing to BRAM
      printf("[DELEGATE-DEBUG] Node %d - Calling write_input_to_bram with input_features: %d\n", i, input_features);
      fflush(stdout);
      
      TF_LITE_KERNEL_LOG(context, "Node %d - Successfully wrote input to BRAM (features: %d)\n", i, input_features);
    } else {
      TF_LITE_KERNEL_LOG(context, "Eval failed: unsupported input tensor type: %d for node %d. Only FLOAT32 supported.\n", input_tensor.type, i);
      return kTfLiteError;
    }

    // Trigger FPGA execution for this node
    printf("[DELEGATE-DEBUG] Node %d - Calling fpga_compute with input_features: %d, output_features: %d\n", i, input_features, output_features);
    fflush(stdout);
    
    if (fpga_ip_driver_->fpga_compute(input_features, output_features)) {
      TF_LITE_KERNEL_LOG(context, "Eval failed: FPGA inference trigger failed for node %d.\n", i);
      return kTfLiteError;
    }
    TF_LITE_KERNEL_LOG(context, "Node %d - FPGA computation completed successfully\n", i);

    // Read output from output BRAM into output tensor (FLOAT32 only)
    if (output_tensor.type == kTfLiteFloat32) {
      // Safety check: ensure output tensor data is allocated
      if (output_tensor.data.f == nullptr) {
        TF_LITE_KERNEL_LOG(context, "Node %d - Output tensor info before read: data=%p, bytes=%d, allocation_type=%d\n", 
                           i, output_tensor.data.f, output_tensor.bytes, output_tensor.allocation_type);
        TF_LITE_KERNEL_LOG(context, "CRITICAL ERROR: Output tensor data is null for node %d - TensorFlow Lite runtime failed to allocate memory\n", i);
        TF_LITE_KERNEL_LOG(context, "This indicates a problem with the TensorFlow Lite runtime or delegate integration\n");
        TF_LITE_KERNEL_LOG(context, "Tensor info: bytes=%d, allocation_type=%d\n", output_tensor.bytes, output_tensor.allocation_type);
        return kTfLiteError;
      }
      
      TF_LITE_KERNEL_LOG(context, "Node %d - Output tensor data address: %p\n", i, output_tensor.data.f);
      
      // DEBUG: Print what we're passing to BRAM for output
      printf("[DELEGATE-DEBUG] Node %d: output_tensor.data.f address: %p, output_features: %d\n", 
             i, output_tensor.data.f, output_features);
      fflush(stdout);
      
      if (fpga_bram_driver_->read_output_from_bram(output_tensor.data.f, output_features)) {
        TF_LITE_KERNEL_LOG(context, "Eval failed: failed to read FLOAT32 output from BRAM for node %d.\n", i);
        return kTfLiteError;
      }
      TF_LITE_KERNEL_LOG(context, "Node %d - Successfully read output from BRAM (features: %d)\n", i, output_features);
    } else {
      TF_LITE_KERNEL_LOG(context, "Eval failed: unsupported output tensor type: %d for node %d. Only FLOAT32 supported.\n", output_tensor.type, i);
      return kTfLiteError;
    }
    
    TF_LITE_KERNEL_LOG(context, "Node %d - Processing completed successfully\n", i);
  } // End of for loop - this was missing!

  TF_LITE_KERNEL_LOG(context, "======== FullyConnectedDelegateKernel::Eval completed successfully =========\n");
  return kTfLiteOk;
}

//   TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
//     // Evaluate the delegated graph.
//     // Here we loop over all the delegated nodes.
//     // We know that all the nodes are either ADD or SUB operations and the
//     // number of nodes equals ''inputs_.size()'' and inputs[i] is a list of
//     // tensor indices for inputs to node ''i'', while outputs_[i] is the list of
//     // outputs for node
//     // ''i''. Note, that it is intentional we have simple implementation as this
//     // is for demonstration.
//     TF_LITE_KERNEL_LOG(context, "======== IN FullyConnectedDelegateKernel::Eval =========\n");

//     int node_count = inputs_.size();
//     for(int i =0; i< node_count; i++) {
//       const TfLiteTensor& input_tensor = context->tensors[inputs_[i][0]];
//       TfLiteTensor& output_tensor = context->tensors[outputs_[i][0]];

//       // Check if input and output tensors are valid
//       if (input_tensor.data.raw == nullptr || output_tensor.data.raw == nullptr) {
//         TF_LITE_KERNEL_LOG(context, "Eval failed: null tensor data detected for node %d.\n", i);
//         return kTfLiteError;
//       }
//     const int input_size = NumElements(input_tensor.dims);
//     const int output_size = NumElements(output_tensor.dims);

//     // Additional safety check for zero-sized tensors
//     if (input_size <= 0 || output_size <= 0) {
//       TF_LITE_KERNEL_LOG(context, "Eval failed: invalid tensor sizes (input: %d, output: %d).\n", input_size, output_size);
//       return kTfLiteError;
//     }

//     // Debug logging for tensor sizes
//     TF_LITE_KERNEL_LOG(context, "Tensor sizes - Input: %d, Output: %d\n", input_size, output_size);
//     TF_LITE_KERNEL_LOG(context, "Input tensor shape: [%d, %d]\n", input_tensor.dims->data[0], input_tensor.dims->data[1]);
//     TF_LITE_KERNEL_LOG(context, "Output tensor shape: [%d, %d]\n", output_tensor.dims->data[0], output_tensor.dims->data[1]);

//     // Extract the actual feature dimensions (ignore batch dimension)
//     const int input_features = input_tensor.dims->data[1];  // Features dimension
//     const int output_features = output_tensor.dims->data[1]; // Output features dimension
    
//     // DEBUG: Force print to console to see what's happening
//     printf("[DELEGATE-DEBUG] Input tensor shape: [%d, %d]\n", input_tensor.dims->data[0], input_tensor.dims->data[1]);
//     printf("[DELEGATE-DEBUG] Output tensor shape: [%d, %d]\n", output_tensor.dims->data[0], output_tensor.dims->data[1]);
//     printf("[DELEGATE-DEBUG] Feature dimensions - Input: %d, Output: %d\n", input_features, output_features);
//     fflush(stdout);
    
//     TF_LITE_KERNEL_LOG(context, "Feature dimensions - Input: %d, Output: %d\n", input_features, output_features);

//     // Write input tensor to input BRAM (FLOAT32 only)
//     if (input_tensor.type == kTfLiteFloat32) {
//       // Safety check: ensure input tensor data is allocated
//       if (input_tensor.data.f == nullptr) {
//         TF_LITE_KERNEL_LOG(context, "Eval failed: input tensor data is null\n");
//         return kTfLiteError;
//       }
      
//       // For batched input, we need to handle each sample in the batch
//       const int batch_size = input_tensor.dims->data[0];
//       if (batch_size != 1) {
//         TF_LITE_KERNEL_LOG(context, "Eval failed: batch size %d not supported. Only batch size 1 supported.\n", batch_size);
//         return kTfLiteError;
//       }
      
//       TF_LITE_KERNEL_LOG(context, "Input tensor data address: %p\n", input_tensor.data.f);
      
//       if (fpga_bram_driver_->write_input_to_bram(input_tensor.data.f, input_features)) {
//         TF_LITE_KERNEL_LOG(context, "Eval failed: failed to write FLOAT32 input to BRAM.\n");
//         return kTfLiteError;
//       }
      
//       // DEBUG: Print what we're passing to BRAM
//       printf("[DELEGATE-DEBUG] Calling write_input_to_bram with input_features: %d\n", input_features);
//       fflush(stdout);
      
//       TF_LITE_KERNEL_LOG(context, "Successfully wrote input to BRAM (features: %d)\n", input_features);
//     } else {
//       TF_LITE_KERNEL_LOG(context, "Eval failed: unsupported input tensor type: %d. Only FLOAT32 supported.\n", input_tensor.type);
//       return kTfLiteError;
//     }

//     // Trigger FPGA execution
//     printf("[DELEGATE-DEBUG] Calling fpga_compute with input_features: %d, output_features: %d\n", input_features, output_features);
//     fflush(stdout);
    
//     if (fpga_ip_driver_->fpga_compute(input_features, output_features)) {
//       TF_LITE_KERNEL_LOG(context, "Eval failed: FPGA inference trigger failed.\n");
//       return kTfLiteError;
//     }
//     TF_LITE_KERNEL_LOG(context, "FPGA computation completed successfully\n");

//     // Read output from output BRAM into output tensor (FLOAT32 only)
//     if (output_tensor.type == kTfLiteFloat32) {
//       // Safety check: ensure output tensor data is allocated
//       if (output_tensor.data.f == nullptr) {
//         TF_LITE_KERNEL_LOG(context, "Eval failed: output tensor data is null\n");
//         return kTfLiteError;
//       }
      
//       TF_LITE_KERNEL_LOG(context, "Output tensor data address: %p\n", output_tensor.data.f);
      
//       // DEBUG: Print what we're passing to BRAM for output
//       printf("[DELEGATE-DEBUG] Calling read_output_from_bram with output_features: %d\n", output_features);
//       fflush(stdout);
      
//       if (fpga_bram_driver_->read_output_from_bram(output_tensor.data.f, output_features)) {
//         TF_LITE_KERNEL_LOG(context, "Eval failed: failed to read FLOAT32 output from BRAM.\n");
//         return kTfLiteError;
//       }
//       TF_LITE_KERNEL_LOG(context, "Successfully read output from BRAM (features: %d)\n", output_features);
//     } else {
//       TF_LITE_KERNEL_LOG(context, "Eval failed: unsupported output tensor type: %d. Only FLOAT32 supported.\n", output_tensor.type);
//       return kTfLiteError;
//     }
//     TF_LITE_KERNEL_LOG(context, "======== FullyConnectedDelegateKernel::Eval completed successfully =========\n");
//     return kTfLiteOk;
// }

 private:
  const FullyConnectedDelegateOptions options_;
  std::vector<std::vector<int>> inputs_, outputs_, biases_, weights_;
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

    TF_LITE_KERNEL_LOG(context, "======== IN FullyConnectedDelegate: IsNodeSupportedByDelegate =========\n");
    TF_LITE_KERNEL_LOG(context, "FullyConnectedDelegate: Checking node support\n");
    TF_LITE_KERNEL_LOG(context, "Node builtin_code: %d (kTfLiteBuiltinFullyConnected = %d)\n", 
                       registration->builtin_code, kTfLiteBuiltinFullyConnected);

    // This delegate supports only FULLY_CONNECTED operations.
    if (registration->builtin_code != kTfLiteBuiltinFullyConnected) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: not a fully connected operation.\n");
      return false;
    }
    
    // only supports RELU and NONE activations.
    const TfLiteFullyConnectedParams* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
    if (params->activation != kTfLiteActNone && params->activation != kTfLiteActRelu) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: unsupported activation type %d.\n", params->activation);
      return false;
    }
    
    // Debug logging to show the activation type.
    TF_LITE_KERNEL_LOG(context, "Activation type: %d\n", params->activation);

    // kTFLiteBuiltinFullyConnected input index are fixed:
    // 0 - input tensor, 1 - weights tensor, 2 - bias tensor (Bias is optional).
    // Output tensor is always at index 0.
    const TfLiteTensor* input = GetInput(context, node, 0);
    const TfLiteTensor* weights = GetInput(context, node, 1);
    const TfLiteTensor* bias = node->inputs->size > 2 ? GetInput(context, node, 2) : nullptr;
    const TfLiteTensor* output = GetOutput(context, node, 0);
    
    if (!input || !weights || !output) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: null tensor pointers detected.\n");
      return false;
    }

    // Log tensor information for debugging
    TF_LITE_KERNEL_LOG(context, "Tensor info - Input: dims=%p, allocation=%d\n", 
                       input->dims, input->allocation_type);
    TF_LITE_KERNEL_LOG(context, "Tensor info - Weights: dims=%p, allocation=%d\n", 
                       weights->dims, weights->allocation_type);
    TF_LITE_KERNEL_LOG(context, "Tensor info - Output: dims=%p, allocation=%d\n", 
                       output->dims, output->allocation_type);
    if (bias) {
      TF_LITE_KERNEL_LOG(context, "Tensor info - Bias: dims=%p, allocation=%d\n", 
                         bias->dims, bias->allocation_type);
    }
    
    // Reject dynamic tensors - check for unknown dimensions
    if (input->dims == nullptr || weights->dims == nullptr || output->dims == nullptr ||
        (bias && bias->dims == nullptr)) {
      TF_LITE_KERNEL_LOG(context, "Null tensor dimensions detected — rejecting.\n");
      return false;
    }

    // Check for dynamic dimensions (negative values indicate dynamic dimensions)
    bool has_dynamic_dims = false;
    for (int i = 0; i < input->dims->size; ++i) {
      if (input->dims->data[i] < 0) {
        TF_LITE_KERNEL_LOG(context, "Dynamic input tensor dimension detected at index %d: %d\n", 
                           i, input->dims->data[i]);
        has_dynamic_dims = true;
      }
    }
    for (int i = 0; i < weights->dims->size; ++i) {
      if (weights->dims->data[i] < 0) {
        TF_LITE_KERNEL_LOG(context, "Dynamic weights tensor dimension detected at index %d: %d\n", 
                           i, weights->dims->data[i]);
        has_dynamic_dims = true;
      }
    }
    for (int i = 0; i < output->dims->size; ++i) {
      if (output->dims->data[i] < 0) {
        TF_LITE_KERNEL_LOG(context, "Dynamic output tensor dimension detected at index %d: %d\n", 
                           i, output->dims->data[i]);
        has_dynamic_dims = true;
      }
    }
    if (bias) {
      for (int i = 0; i < bias->dims->size; ++i) {
        if (bias->dims->data[i] < 0) {
          TF_LITE_KERNEL_LOG(context, "Dynamic bias tensor dimension detected at index %d: %d\n", 
                             i, bias->dims->data[i]);
          has_dynamic_dims = true;
        }
      }
    }

    if (has_dynamic_dims) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: dynamic tensor dimensions detected.\n");
      return false;
    }

    // Also check allocation types as additional safety
    if (input->allocation_type == kTfLiteDynamic ||
        weights->allocation_type == kTfLiteDynamic ||
        output->allocation_type == kTfLiteDynamic ||
        (bias && bias->allocation_type == kTfLiteDynamic)) {
      TF_LITE_KERNEL_LOG(context, "Dynamic tensor allocation detected — rejecting.\n");
      return false;
    }


    // Debug logging to show the shapes of input, weights, and output tensors.
    TF_LITE_KERNEL_LOG(context, 
    "FullyConnectedDelegate: Input shape = [%d x %d], Weights shape = [%d x %d], Output shape = [%d x %d]\n",
    input->dims->data[0], input->dims->data[1],
    weights->dims->data[0], weights->dims->data[1],
    output->dims->data[0], output->dims->data[1]);

    // Check if input, weights, and output tensors are of rank 2.
    if (input->dims->size != 2 || weights->dims->size != 2 || output->dims->size != 2) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: tensor rank != 2 (input: %d, weights: %d, output: %d).\n",
                        input->dims->size, weights->dims->size, output->dims->size);
      return false;
    }

    // Check if input, weights, and output tensors are of max size 32x32.
    // FPGA IP is designed to handle max 32x32 matrices.
    if (input->dims->data[1] > 32 || weights->dims->data[0] > 32 || 
      weights->dims->data[1] > 32 || output->dims->data[1] > 32) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: tensor dimensions exceed 32x32 limit.\n");
      return false;
    }

    // Debug logging to show the types of input, weights, bias, and output tensors.
    TF_LITE_KERNEL_LOG(context,
    "Input type: %d, Weights type: %d, Bias type: %d, Output type: %d\n",
    input->type, weights->type, bias ? bias->type : -1, output->type);
    
    // Check if input, weights, and output tensors are of supported types.
    // This delegate supports only FLOAT32 tensors.
    bool type_check = (input->type == kTfLiteFloat32 && 
                       weights->type == kTfLiteFloat32 &&
                       (!bias || bias->type == kTfLiteFloat32) && 
                       output->type == kTfLiteFloat32);
    
    if (type_check) {
      TF_LITE_KERNEL_LOG(context, "Tensor types: FLOAT32 (supported)\n");
    } else {
      TF_LITE_KERNEL_LOG(context, "Tensor types: Only FLOAT32 tensors supported. Detected types - Input: %d, Weights: %d, Bias: %d, Output: %d\n",
                         input->type, weights->type, bias ? bias->type : -1, output->type);
      TF_LITE_KERNEL_LOG(context, "  INT8 and INT32 support will be added later when low-level drivers are ready.\n");
    }
    
    if (!type_check) {
      TF_LITE_KERNEL_LOG(context, "Node rejected: unsupported tensor types.\n");
      return false;
    }
    
    TF_LITE_KERNEL_LOG(context, "Node accepted: fully connected operation meets all requirements.\n");
    TF_LITE_KERNEL_LOG(context, "======== FullyConnectedDelegate: Node support check complete ===========\n");
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

  // SimpleDelegateInterface::Options DelegateOptions() const override {
  //   // Configure delegate partitioning options
  //   // This allows the delegate to work with complex graphs by creating
  //   // multiple partitions and selecting only the nodes it can support
  //   SimpleDelegateInterface::Options options;
  //   options.max_delegated_partitions = 100;  // Allow many partitions
  //   options.min_nodes_per_partition = 1;     // Allow single-node partitions
  //   return options;
  // }
  SimpleDelegateInterface::Options DelegateOptions() const override {
  SimpleDelegateInterface::Options options;
  options.max_delegated_partitions = 1;  // ✅ Only allow 1 delegated partition
  options.min_nodes_per_partition = 2;   // ✅ Only delegate if 2+ supported nodes are together
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



