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

#ifndef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_KERNEL_H_
#define TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_KERNEL_H_

#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_plugin.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace delegate {
namespace nnapi {

constexpr int32_t kMinSdkVersionForNNAPI = 27;
constexpr int32_t kMinSdkVersionForNNAPI11 = 28;
constexpr int32_t kMinSdkVersionForNNAPI12 = 29;
constexpr int32_t kMinSdkVersionForNNAPI13 = 30;
constexpr int32_t kNNAPIRuntimeFeatureLevel5 = 31;
constexpr int32_t kNNAPIRuntimeFeatureLevel6 = 1000006;
constexpr int32_t kNNAPIRuntimeFeatureLevel7 = 1000007;
constexpr int32_t kNNAPIRuntimeFeatureLevel8 = 1000008;

class NNAPIOpBuilder;

// The kernel that represents the node sub set of TF Lite being run on NN API.
struct NNAPIOpMappingArgs {
  TfLiteContext* context;
  NNAPIOpBuilder* builder;
  TfLiteNode* node;
  int node_index;
  std::vector<int>* model_state_outputs;
  std::vector<int>* model_state_tfl_inputs;
  std::vector<std::tuple<int, int>>* feedback_loops;
  int* nnapi_errno;
};

// RAII NN API Model Destructor for use with std::unique_ptr
class NNFreeModel {
 public:
  explicit NNFreeModel(const NnApi* nnapi) : nnapi_(nnapi) {}
  void operator()(ANeuralNetworksModel* model) {
    nnapi_->ANeuralNetworksModel_free(model);
  }

 private:
  // NnApi instance to use. Not owned by this object.
  const NnApi* nnapi_;
};
// RAII NN API Compilation Destructor for use with std::unique_ptr
class NNFreeCompilation {
 public:
  explicit NNFreeCompilation(const NnApi* nnapi) : nnapi_(nnapi) {}
  void operator()(ANeuralNetworksCompilation* model) {
    nnapi_->ANeuralNetworksCompilation_free(model);
  }

 private:
  // NnApi instance to use. Not owned by this object.
  const NnApi* nnapi_;
};
// RAII NN API Execution Destructor for use with std::unique_ptr
class NNFreeExecution {
 public:
  explicit NNFreeExecution(const NnApi* nnapi) : nnapi_(nnapi) {}
  void operator()(ANeuralNetworksExecution* execution) {
    nnapi_->ANeuralNetworksExecution_free(execution);
  }

 private:
  // NnApi instance to use. Not owned by this object.
  const NnApi* nnapi_;
};
// RAII NN API Burst Destructor for use with std::unique_ptr
class NNFreeBurst {
 public:
  explicit NNFreeBurst(const NnApi* nnapi) : nnapi_(nnapi) {}
  void operator()(ANeuralNetworksBurst* model) {
    nnapi_->ANeuralNetworksBurst_free(model);
  }

 private:
  // NnApi instance to use. Not owned by this object.
  const NnApi* nnapi_;
};

using UniqueExecution =
    std::unique_ptr<ANeuralNetworksExecution, NNFreeExecution>;

// RAII NN API MappingUtil Destructor for use with std::unique_ptr
class NNFreeMappingUtil {
 public:
  void operator()(NnapiMappingUtilCInterface* mapping_util);
};

// Manage NNAPI shared memory handle
class NNMemory {
 public:
  NNMemory(const NnApi* nnapi, const char* name, size_t size);

  ~NNMemory();

  ANeuralNetworksMemory* get_handle() { return nn_memory_handle_; }
  uint8_t* get_data_ptr() { return data_ptr_; }
  size_t get_byte_size() { return byte_size_; }

 private:
  // NnApi instance to use. Not owned by this object.
  const NnApi* nnapi_;
  int fd_ = 0;
  size_t byte_size_ = 0;
  uint8_t* data_ptr_ = nullptr;
  ANeuralNetworksMemory* nn_memory_handle_ = nullptr;
#ifndef __ANDROID__
  std::string shm_region_name_;
#endif
};

// LINT.IfChange
enum class NNAPIValidationFailureType : int {
  // The operator is not supported by either NNAPI or the NNAPI Delegate.
  kUnsupportedOperator = 0,
  // The given operation or operands are not supported on the specified
  // Android SDK version. The min supported version is specified in the
  // validation failure message.
  kUnsupportedAndroidVersion = 1,
  // The version of the operator (value of TfLiteRegistration::version)
  // for the given op is not supported. The max supported version
  // is specified in the validation failure message.
  // For more details on each operator version see
  // the GetBuiltinOperatorVersion function in
  // third_party/tensorflow/lite/tools/versioning/op_version.cc.
  kUnsupportedOperatorVersion = 2,
  // The given input operand type is not supported for the current combination
  // of operator type and sdk version.
  kUnsupportedInputType = 3,
  // When using NN API version 1.0 or 1.1, the condition
  //   input_scale * filter_scale < output_scale
  // must be true for quantized versions of the following ops:
  // * CONV_2D
  // * DEPTHWISE_CONV_2D
  // * FULLY_CONNECTED (where filter actually stands for weights)
  // The condition is relaxed and no longer required since version 1.2.
  kNotRestrictedScaleCompliant = 4,
  // The given output operand type is not supported for the current combination
  // of operator type and sdk version.
  kUnsupportedOutputType = 5,
  // The size of the operand tensor is too large.
  kUnsupportedOperandSize = 6,
  // The value of one of the operands or of a combination of operands is
  // not supported. Details are provided in the failure message.
  kUnsupportedOperandValue = 7,
  // The combination of float inputs and quantized weights or filters
  // is not supported
  kUnsupportedHybridOperator = 8,
  // The quantization type (for example per-channel quantization) is not
  // supported.
  kUnsupportedQuantizationType = 9,
  // The accelerated version of operation requires a specific operand to be
  // specified.
  kMissingRequiredOperand = 10,
  // The rank of the operand is not supported. Details in the failure message.
  kUnsupportedOperandRank = 11,
  // The input tensor cannot be dynamically-sized.
  kInputTensorShouldHaveConstantShape = 12,
  // The operator has a different number of inputs of the one or ones that
  // are supported by NNAPI.
  kUnsupportedOperatorVariant = 13,
  // The accelerated version of the operator cannot specify an activation
  // function.
  kNoActivationExpected = 14,
  // Quantization scale and/or zero point are not in the supported value(s)
  // for the accelerated operation.
  kUnsupportedQuantizationParameters = 15,
};
// LINT.ThenChange(nnapi_linter/linter.proto)

struct NNAPIValidationFailure {
  NNAPIValidationFailureType type;
  std::string message;

  NNAPIValidationFailure(NNAPIValidationFailureType type, const char* message)
      : type(type), message(message) {}
};

// LRU cache of reusable NNAPI executions.
class NNAPIExecutionCache {
 public:
  // The cache signature. Uniquely identifies an execution request.
  struct Signature {
    std::vector<uint64_t> tensor_handle_timestamps;
    std::vector<int> dynamic_dimensions;

    bool operator==(const Signature& other) const;
    struct Hasher {
      std::size_t operator()(const Signature& signature) const;
    };
  };

  explicit NNAPIExecutionCache(uint32_t max_cache_size)
      : max_cache_size_(max_cache_size) {}

  // Gets the cached execution by signature.
  // On cache hit, the target execution is set to be the most recently used one.
  // On cache miss, nullptr is returned.
  ANeuralNetworksExecution* Get(const Signature& signature);

  // Puts the execution in cache and set it to be the most recently used one.
  // If the cache is full, the least recently used entry will be released.
  void Put(const Signature& signature, UniqueExecution execution);

  // Clears all cache entries.
  void Clear();

  // Resets the max cache size.
  void SetMaxCacheSize(uint32_t max_cache_size);

 private:
  // Releases the least recently used cache.
  void ReleaseLRU();

  // The maximum number of reusable executions to cache.
  uint32_t max_cache_size_;

  // Cache signatures in the order of most recent use. The most recently used
  // signature is at the front of the list.
  std::list<Signature> order_;

  // A hash map to lookup a managed execution by its signature.
  std::unordered_map<Signature,
                     std::pair<std::list<Signature>::iterator, UniqueExecution>,
                     Signature::Hasher>
      lookup_;
};

// The kernel that represents the node sub set of TF Lite being run on NN API.
class NNAPIDelegateKernel {
 public:
  explicit NNAPIDelegateKernel(
      const NnApi* nnapi, NnapiDelegateVendorPlugin* vendor_plugin = nullptr)
      : initialised_(false),
        nnapi_(nnapi),
        nn_model_(nullptr, NNFreeModel(nnapi_)),
        nn_compilation_(nullptr, NNFreeCompilation(nnapi_)),
        nn_burst_(nullptr, NNFreeBurst(nnapi_)),
        nn_execution_cache_(/*max_cache_size=*/4),
        mapping_util_(NnapiMappingUtilCInterfaceCreate(), NNFreeMappingUtil()),
        vendor_plugin_(vendor_plugin) {}
  NNAPIDelegateKernel() : NNAPIDelegateKernel(NnApiImplementation()) {}
  ~NNAPIDelegateKernel() {
    for (auto content : allocation_memory_mapping_) {
      nnapi_->ANeuralNetworksMemory_free(content.second);
    }
  }

  static NnapiMappingUtilCInterface* NnapiMappingUtilCInterfaceCreate();

  // Translate a node into its operands
  // It assumes that the call to Validate for has been successful for
  // the operation.
  // In case of success it returns kTfLiteOk and stores in n_op_type the
  // NNAPI Operation code.
  // Returns kTfLiteError in case of failures during mapping.
  static TfLiteStatus Map(TfLiteContext* context, int builtin_code, int version,
                          int android_sdk_version,
                          const NNAPIOpMappingArgs& mapping_args,
                          ANeuralNetworksOperationType* nn_op_type,
                          NnapiDelegateVendorPlugin* vendor_plugin = nullptr);

  // Returns true if the node can be accelerated with NNAPI.
  static bool Validate(
      const TfLiteContext* context, const TfLiteRegistration* registration,
      int android_sdk_version, const TfLiteNode* node,
      bool is_accelerator_specified,
      NnapiDelegateVendorPlugin* vendor_plugin = nullptr,
      // Collects lists of failures collected during
      // the validation of the possibility of accelerating
      // the given node
      std::vector<NNAPIValidationFailure>* map_failures = nullptr);

  // Initialize the kernel (a NN model) and builds the NN Model.
  // Any NNAPI Related error causing this method to fail will have the
  // associated error number stored in nnapi_errno
  TfLiteStatus Init(TfLiteContext* context, const TfLiteDelegateParams* params,
                    int* nnapi_errno);

  // Creates the NNAPI Compilation for the NN model. It assumes that Init has
  // been called and completed successfully.
  // Any NNAPI Related error causing this method to fail will have the
  // associated error number stored in nnapi_errno
  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node,
                       int* nnapi_errno);

  // Invoke the NN Model. Expects Init and Prepare to have been completed
  // successfully.
  // Any NNAPI Related error causing this method to fail will have the
  // associated error number stored in nnapi_errno
  TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node,
                      int* nnapi_errno);

  // Returns the list of operations supported by the current NNAPI model as
  // built in Prepare. Every operation is identified by the index as provided
  // in the delegate parameters given to the delegate during the Init call.
  // It expects the Init method has been called and completed successfully and
  // returns kTfLiteError if not. Returns an error if any of the NNAPI
  // operations fails or if the
  // ANeuralNetworksModel_getSupportedOperationsForDevices function is not
  // available in the NnApi object.
  TfLiteStatus GetOperationsSupportedByTargetNnApiDevices(
      TfLiteContext* context, std::vector<int>* supported_nodes,
      int* nnapi_errno);

 private:
  // True if initialization has been completed successfully
  bool initialised_;
  // Access to NNApi.
  const NnApi* nnapi_;
  // ANN device handle.
  std::vector<ANeuralNetworksDevice*> nnapi_devices_;
  // Name of the nnapi device, empty if nnapi_devices_ is empty;
  std::string device_name_;
  // ANN API state.
  std::unique_ptr<ANeuralNetworksModel, NNFreeModel> nn_model_;
  std::unique_ptr<ANeuralNetworksCompilation, NNFreeCompilation>
      nn_compilation_;
  std::unique_ptr<ANeuralNetworksBurst, NNFreeBurst> nn_burst_;
  NNAPIExecutionCache nn_execution_cache_;
  // The mappings of tenor id to BufferHandle. Needed to track BufferHandle
  // change and alter nn_reusable_execution_ if necessary.
  std::vector<int> tensor_handle_map_;
  // Node indices that this delegate is responsible for. Indices here
  // indexes into the nodes array in the TfLiteContext.
  std::vector<int> nodes_;
  // Track indices we use
  std::unique_ptr<NnapiMappingUtilCInterface, NNFreeMappingUtil> mapping_util_;

  std::map<const MMAPAllocation*, ANeuralNetworksMemory*>
      allocation_memory_mapping_;
  // Track memory map
  const std::vector<StatefulNnApiDelegate::MemoryRegistration>*
      tensor_memory_map_;
  std::vector<int> model_state_outputs_;
  std::vector<int> model_state_tfl_inputs_;
  // This is the equivalent of the pair model_state_outputs_,
  // model_state_tfl_inputs_ for all tensors where we have to keep the output
  // data available for TFLite model users
  std::vector<std::tuple<int, int>> feedback_loops_;
  // The mappings of tenor id to max size in bytes. If the hint is not provided
  // for a tensor, it is set to 0.
  std::vector<size_t> tensor_max_size_hints_;

  std::unique_ptr<NNMemory> nn_input_memory_;
  std::unique_ptr<NNMemory> nn_output_memory_;

  std::vector<uint8_t> nn_compilation_cache_token_;

  // Map of DENSIFY output tensor id to node id.
  std::vector<int> densify_output_to_node_mapping_;
  // Map of DEQUANTIZE output tensor id to node id.
  // Only contains DEQUANTIZE nodes with non-const input.
  std::vector<int> non_const_dequantize_output_to_node_mapping_;

  NnapiDelegateVendorPlugin* vendor_plugin_ = nullptr;

  // Fully initialized in NNAPIDelegateKernel::AddOpsAndTensors
  int target_feature_level_ = 27;  // kMinSdkVersionForNNAPI10

  void AddDequantizeOperatorsWhereNeeded(
      const TfLiteContext* context, int builtin_code, const TfLiteNode* node,
      int tflite_node_index, NNAPIOpBuilder* builder, int* nnapi_errno);

  TfLiteStatus DensifyAndDequantizeConstTensor(TfLiteContext* context,
                                               int densify_node_id,
                                               bool should_dequantize,
                                               NNAPIOpBuilder& builder);

  TfLiteStatus AddOpsAndTensors(TfLiteContext* context, int* nnapi_errno,
                                bool allow_dynamic_dimensions);

  TfLiteStatus BuildGraph(TfLiteContext* context,
                          const StatefulNnApiDelegate::Options& options,
                          const TfLiteIntArray* input_tensors,
                          const TfLiteIntArray* output_tensors,
                          int* nnapi_errno);
};

}  // namespace nnapi
}  // namespace delegate
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_KERNEL_H_
