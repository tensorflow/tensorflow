#include "tensorflow/lite/delegates/utils/mw_delegates/add_fpga_test_delegate/add_fpga_test_delegate.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
// #include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/delegates/utils/mw_delegates/add_fpga_test_delegate/fpga_ip_driver.h"

namespace tflite {
namespace add_fpga_test {

// AddFpgaTestDelegateKernel implements the interface of SimpleDelegateKernelInterface.
// This holds the Delegate capabilities.
class AddFpgaTestDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit AddFpgaTestDelegateKernel(const AddFpgaTestDelegateOptions& options)
      : options_(options), fpga_driver_(std::make_unique<FpgaIpDriver>()) {}

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
    return !options_.error_during_prepare ? kTfLiteOk : kTfLiteError;
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
  const AddFpgaTestDelegateOptions options_;
  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_;
  std::unique_ptr<FpgaIpDriver> fpga_driver_;  // FPGA driver instance

  TfLiteStatus ComputeResult(TfLiteContext* context, int builtin_code,
                             const TfLiteTensor* input_tensor_1,
                             const TfLiteTensor* input_tensor_2,
                             TfLiteTensor* output_tensor) {
    if (NumElements(input_tensor_1->dims) != NumElements(input_tensor_2->dims) ||
        NumElements(input_tensor_1->dims) != NumElements(output_tensor->dims)) {
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

class AddFpgaTestDelegate : public SimpleDelegateInterface {
 public:
  explicit AddFpgaTestDelegate(const AddFpgaTestDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // This delegate supports only ADD and SUB operations.
    if (registration->builtin_code != kTfLiteBuiltinAdd &&
        registration->builtin_code != kTfLiteBuiltinSub)
      return false;
    // Only supports int32 type
    for (int i = 0; i < node->inputs->size; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteInt32) {
        return false;
      }
    }
    for (int i = 0; i < node->outputs->size; ++i) {
      auto& tensor = context->tensors[node->outputs->data[i]];
      if (tensor.type != kTfLiteInt32) {
        return false;
      }
    }
    // Only support static int32 tensors
    for (int i = 0; i < node->inputs->size; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteInt32 || IsDynamicTensor(tensor)) {
        return false;
      }
    }
    for (int i = 0; i < node->outputs->size; ++i) {
      auto& tensor = context->tensors[node->outputs->data[i]];
      if (tensor.type != kTfLiteInt32 || IsDynamicTensor(tensor)) {
        return false;
      }
    }
    return true;
  }
  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "AddFpgaTestDelegate";
    // This name is used for debugging/logging/profiling.
    // It is important to use a unique name for each delegate.
    // If multiple delegates have the same name, it can lead to confusion in
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<AddFpgaTestDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const AddFpgaTestDelegateOptions options_;
  
  // Utility to check for dynamic tensors
  bool IsDynamicTensor(const TfLiteTensor& tensor) const {
    return tensor.allocation_type == kTfLiteDynamic;
}

};

}  // namespace add_fpga_test
}  // namespace tflite

AddFpgaTestDelegateOptions TfLiteAddFpgaTestDelegateOptionsDefault() {
  AddFpgaTestDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this dummy test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteAddFpgaTestDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteAddFpgaTestDelegateCreate(const AddFpgaTestDelegateOptions* options) {
  std::unique_ptr<tflite::add_fpga_test::AddFpgaTestDelegate> add_fpga_test_delegate(
      new tflite::add_fpga_test::AddFpgaTestDelegate(
          options ? *options : TfLiteAddFpgaTestDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(add_fpga_test_delegate));
}

// Destroys a delegate created with `TfLiteAddFpgaTestDelegateCreate` call.
void TfLiteAddFpgaTestDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}



