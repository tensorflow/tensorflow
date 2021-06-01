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
#ifndef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

typedef struct ANeuralNetworksMemory ANeuralNetworksMemory;

namespace tflite {

namespace delegate {
namespace nnapi {
class NNAPIDelegateKernel;
}  // namespace nnapi
}  // namespace delegate

using tflite::delegate::nnapi::NNAPIDelegateKernel;

// TFliteDelegate to interface with NNAPI.
class StatefulNnApiDelegate : public TfLiteDelegate {
 public:
  // Encapsulates all options that are specific to NNAPI delegate.
  struct Options {
    // Preferred Power/perf trade-off. For more details please see
    // ANeuralNetworksCompilation_setPreference documentation in :
    // https://developer.android.com/ndk/reference/group/neural-networks.html
    enum ExecutionPreference {
      kUndefined = -1,
      kLowPower = 0,
      kFastSingleAnswer = 1,
      kSustainedSpeed = 2,
    };

    // Preferred Power/perf trade-off.
    ExecutionPreference execution_preference = kUndefined;

    // Selected NNAPI accelerator with nul-terminated name.
    // Default to nullptr, which implies the NNAPI default behavior: NNAPI
    // runtime is allowed to use all available accelerators. If the selected
    // accelerator cannot be found, NNAPI will not be used.
    // It is the caller's responsibility to ensure the string is valid for the
    // duration of the Options object lifetime.
    const char* accelerator_name = nullptr;

    // The nul-terminated cache dir for NNAPI model.
    // Default to nullptr, which implies the NNAPI will not try caching the
    // compilation.
    const char* cache_dir = nullptr;

    // The unique nul-terminated token string for NNAPI model.
    // Default to nullptr, which implies the NNAPI will not try caching the
    // compilation. It is the caller's responsibility to ensure there is no
    // clash of the tokens.
    // NOTE: when using compilation caching, it is not recommended to use the
    // same delegate instance for multiple models.
    const char* model_token = nullptr;

    // Whether to disallow NNAPI CPU usage. Only effective on Android 10 and
    // above. The NNAPI CPU typically performs less well than built-in TfLite
    // kernels, but allowing CPU allows partial acceleration of models. If this
    // is set to true, NNAPI is only used if the whole model is accelerated.
    bool disallow_nnapi_cpu = true;

    // Specifies the max number of partitions to delegate. A value <= 0 means
    // no limit.
    // If the delegation of the full set of supported nodes would generate a
    // number of partition greater than this parameter, only
    // <max_number_delegated_partitions> of them will be actually accelerated.
    // The selection is currently done sorting partitions in decreasing order
    // of number of nodes and selecting them until the limit is reached.
    int max_number_delegated_partitions = 3;

    // allow fp32 compuation to be run in fp16.
    bool allow_fp16 = false;

    // Specifies the relative priority for executions of the model.
    // Available values are {ANEURALNETWORKS_PRIORITY_LOW,
    // ANEURALNETWORKS_PRIORITY_MEDIUM, ANEURALNETWORKS_PRIORITY_HIGH,
    // ANEURALNETWORKS_PRIORITY_DEFAULT}.
    int execution_priority = ANEURALNETWORKS_PRIORITY_DEFAULT;

    // Specifies the maximum expected duration in nanosecond for compiling the
    // model. If the device is not able to complete the compilation within the
    // specified duration, the compilation may be aborted. If set to 0, the
    // timeout duration is considered infinite.
    uint64_t max_compilation_timeout_duration_ns = 0;

    // Specifies the maximum expected duration in nanosecond for executing the
    // model. If the device is not able to complete the execution within the
    // specified duration, the execution may be aborted. If set to 0, the
    // timeout duration is considered infinite.
    uint64_t max_execution_timeout_duration_ns = 0;

    // Specifies the maximum expected duration in nanosecond for WHILE loops in
    // the execution. If a WHILE loop condition model does not output false
    // within the specified duration, the execution will be aborted. If set to
    // 0, the default timeout for loops will be used.
    uint64_t max_execution_loop_timeout_duration_ns = 0;

    // Whether to allow dynamic dimension sizes without re-compilation.
    // A tensor of with dynamic dimension must have a valid dim_signature
    // defined.
    // Only supported in NNAPI 1.1 and newer versions.
    // WARNING: Setting this flag to true may result in model being rejected by
    // accelerator. This should only be enabled if the target device supports
    // dynamic dimensions of the model.
    bool allow_dynamic_dimensions = false;

    // Force using NNAPI Burst mode if supported.
    // Burst mode allows accelerators to efficiently manage resources, which
    // would significantly reduce overhead especially if the same delegate
    // instance is to be used for multiple inferences.
    // If NNAPI devices are specified and are of NNAPI feature level 5 or
    // higher, NNAPI delegate will automatically enable burst mode for better
    // performance.
    // Default: Disabled for devices with NNAPI feature level 4 or lower.
    bool use_burst_computation = false;
  };

  // Uses default options.
  StatefulNnApiDelegate();

  // The ownership of the NnApi instance is left to the caller of the
  // StatefulNnApiDelegate constructor; the caller must ensure that the lifetime
  // of the NnApi instance exceeds the lifetime of the StatefulNnApiDelegate.
  explicit StatefulNnApiDelegate(const NnApi* nnapi);

  // The constructor that accepts options from user.
  // This makes a copy of any data that it needs from Options, so
  // the caller can safely deallocate any storage pointed to by
  // the 'const char *' members of Options immediately after calling this.
  explicit StatefulNnApiDelegate(Options options);

  // Constructor that accepts both an NnApi instance and options.
  // The ownership of the NnApi instance is left to the caller of the
  // StatefulNnApiDelegate constructor; the caller must ensure that the lifetime
  // of the NnApi instance exceeds the lifetime of the StatefulNnApiDelegate.
  // This constructor makes a copy of any data that it needs from Options, so
  // the caller can safely deallocate any storage pointed to by
  // the 'const char *' members of Options immediately after calling this.
  StatefulNnApiDelegate(const NnApi* nnapi, Options options);

  ~StatefulNnApiDelegate() = default;

  // Returns the delegate options.
  // The lifetime of the storage pointed to by the 'const char *' members of the
  // returned Options object is the same as the lifetime of the supplied
  // TfLiteDelegate instance.
  static const Options GetOptions(TfLiteDelegate* delegate);

  // Callback function which copies data from ANeuralNetworksMemory to host
  // tensor CPU buffer. It is the users responsibility to implement these
  // callbacks for the specific types of shared memory they intend to use.
  // WARNING: This is an experimental interface that is subject to change.
  typedef TfLiteStatus (*CopyToHostTensorFnPtr)(TfLiteTensor* tensor,
                                                ANeuralNetworksMemory* memory,
                                                size_t memory_offset,
                                                size_t byte_size,
                                                void* callback_context);

  // Encapsulates all fields related to memory registration for internal
  // bookkeeping only.
  struct MemoryRegistration {
    ANeuralNetworksMemory* memory;
    CopyToHostTensorFnPtr callback;
    void* callback_context;
  };

  // Register the ANeuralNetworksMemory handle with the delegate. A
  // TfLiteBufferHandle will be returned to be used with
  // Interpreter::SetBufferHandle. The callback_context will be passed to the
  // callback function when invoked.
  // Note: the returned TfLiteBufferHandle can only be used with a single
  // Interpreter instance. However, the caller can register the same memory
  // multiple times to get different handles to use with difference Interpreter
  // instances
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteBufferHandle RegisterNnapiMemory(ANeuralNetworksMemory* memory,
                                         CopyToHostTensorFnPtr callback,
                                         void* callback_context);

  // Returns the vector of known ANeuralNetworksMemory handles.
  // Note: this function is not intended to be called by developers.
  // WARNING: This is an experimental interface that is subject to change.
  static const std::vector<MemoryRegistration>& GetTensorMemoryMap(
      TfLiteDelegate* delegate);

  // Returns the int value of the ResultCode returned by the latest
  // failed call to NNAPI, if any. Zero only in case of NO failed calls since
  // the construction of this instance of StatefulNnApiDelegate.
  // The error code is reset when the delegate is re-initialized
  // (i.e. when calling interpreter.ModifyGraphWithDelegate(delegate)).
  int GetNnApiErrno() const;

 private:
  // Encapsulates all delegate data.
  struct Data {
    // Pointer to NNAPI implementation to be used by this delegate as
    // set when building the StatefulNnApiDelegate instance.
    // Will generally be the NnApiInstance() singleton but can be overridden
    // for testing or for users needing to wrap or stub parts of NNAPI.
    // The ownership of the nnapi instance is left to the caller of
    // the StatefulNnApiDelegate constructor.
    const NnApi* nnapi;
    // Preferred Power/perf trade-off.
    Options::ExecutionPreference execution_preference;
    // Selected NNAPI accelerator name.
    std::string accelerator_name;
    // The cache dir for NNAPI model.
    std::string cache_dir;
    // The unique token string for NNAPI model.
    std::string model_token;
    // Whether to disallow NNAPI CPU.
    bool disallow_nnapi_cpu;
    // Tensor to ANeuralNetworksMemory mapping.
    std::vector<MemoryRegistration> tensor_memory_map;
    // Contains a non zero value if any NNAPI method call
    // operation returned a non zero result code.
    int nnapi_errno = ANEURALNETWORKS_NO_ERROR;
    // Cache of kernels already built in StatefulNnApiDelegate::DoPrepare
    // when trying to understand if all nodes are supported by the target
    // accelerators.
    // The key is the index of the first node in the partition.
    // Couldn't use unique_ptr because of problems building on gcc
    std::unordered_map<int, NNAPIDelegateKernel*> delegate_state_cache;
    // Maximum number of NNAPI partition to delegate. Zero or negative means
    // no limit. Copied from StatefulNnApiDelegate::Options
    int max_number_delegated_partitions;
    // allow fp32 computation to be run in fp16.
    bool allow_fp16;
    // Specifies the relative priority for executions of the model.
    int execution_priority = ANEURALNETWORKS_PRIORITY_DEFAULT;
    // Specifies the maximum expected duration in nanosecond for compiling the
    // model.
    uint64_t max_compilation_timeout_duration_ns = 0;
    // Specifies the maximum expected duration in nanosecond for executing the
    // model.
    uint64_t max_execution_timeout_duration_ns = 0;
    // Specifies the maximum expected duration in nanosecond for WHILE loops in
    // the execution
    uint64_t max_execution_loop_timeout_duration_ns = 0;
    // Whether to allow dynamic dimension sizes without re-compilation.
    bool allow_dynamic_dimensions = false;
    // Whether to use NNAPI Burst mode.
    bool use_burst_computation = false;

    explicit Data(const NnApi* nnapi);
    ~Data();

    // Caches an initialised NNAPIDelegateKernel.
    void CacheDelegateKernel(const TfLiteDelegateParams* delegate_params,
                             NNAPIDelegateKernel* delegate_state);
    // Returns a cached NNAPIDelegateKernel if available and removes it
    // from the cache transferring the ownership to the caller.
    NNAPIDelegateKernel* MaybeGetCachedDelegateKernel(
        const TfLiteDelegateParams* delegate_params);
  };

  // Implements TfLiteDelegate::Prepare. Please refer to TFLiteDelegate
  // documentation for more info.
  static TfLiteStatus DoPrepare(TfLiteContext* context,
                                TfLiteDelegate* delegate);

  // Copy the data from delegate buffer handle into raw memory of the given
  // 'tensor'. The delegate is allowed to allocate the raw
  // bytes as long as it follows the rules for kTfLiteDynamic tensors.
  static TfLiteStatus DoCopyFromBufferHandle(TfLiteContext* context,
                                             TfLiteDelegate* delegate,
                                             TfLiteBufferHandle buffer_handle,
                                             TfLiteTensor* tensor);

  // Copy the data from raw memory of the given 'tensor' to delegate buffer
  // handle. Currently this function is not supported, and calling the function
  // will result in an error.
  static TfLiteStatus DoCopyToBufferHandle(TfLiteContext* context,
                                           TfLiteDelegate* delegate,
                                           TfLiteBufferHandle buffer_handle,
                                           TfLiteTensor* tensor);

  // Free the Delegate Buffer Handle. Note: This only frees the handle, but
  // this doesn't release the underlying resource (e.g. textures). The
  // resources are either owned by application layer or the delegate.
  static void DoFreeBufferHandle(TfLiteContext* context,
                                 TfLiteDelegate* delegate,
                                 TfLiteBufferHandle* handle);

  // Returns the nodes that can be delegated via NNAPI to the accelerator
  // specified in the delegate options and information about the way the
  // graph will be partitioned if the supported nodes will be delegated.
  // Partition information is composed by the number of partitions and
  // the delegate parameters associated to each partition.
  // The method also caches in delegate->data the NNApiDelegateKernel instances
  // that have been created during the device evaluation.
  // All arguments are expected to be non-null.
  static TfLiteStatus GetNodesSupportedByAccelerator(
      TfLiteContext* context, TfLiteDelegate* delegate, const NnApi* nnapi,
      const std::vector<int>& supported_nodes,
      std::vector<int>* device_supported_nodes, int* num_partitions,
      TfLiteDelegateParams** params_array, int* nnapi_errno);

  // Alters the given array of nodes_to_delegate to limit the number of NNAPI
  // owned partition to be less or equal than num_partitions. If num_partitions
  // is less or equal to zero the input is left unaltered.
  // The nodes_to_delegate array is expected to contain at element 0 the number
  // of nodes to delegate and in remaining elements the set of nodes
  // that would be delegated to NNAPI if this function wouldn't be
  // called. It will be altered storing in the first element the count of
  // nodes to actually delegate and in the remainder of the array the indexes.
  // The params_array params might be altered during the functions execution.
  static TfLiteStatus LimitDelegatedPartitions(
      int max_partitions,
      std::vector<TfLiteDelegateParams> partition_params_array,
      std::vector<int>* nodes_to_delegate);

  // Delegate data presented through TfLiteDelegate::data_.
  Data delegate_data_;
};

// DEPRECATED: Please use StatefulNnApiDelegate class instead.
//
// Returns a singleton delegate that can be used to use the NN API.
// e.g.
//   NnApiDelegate* delegate = NnApiDelegate();
//   interpreter->ModifyGraphWithDelegate(&delegate);
// NnApiDelegate() returns a singleton, so you should not free this
// pointer or worry about its lifetime.
TfLiteDelegate* NnApiDelegate();

}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
