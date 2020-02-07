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

#include "absl/types/optional.h"
#include "tensorflow/lite/c/common.h"

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
    bool disallow_nnapi_cpu = false;
  };

  // Uses default options.
  StatefulNnApiDelegate();

  // The constructor that accepts options from user.
  explicit StatefulNnApiDelegate(Options options);

  ~StatefulNnApiDelegate() = default;

  // Returns the delegate options.
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
    // Constains a non zero value if any NNAPI method call
    // operation returned a non zero result code.
    int nnapi_errno;
    // Cache of kernels already built in StatefulNnApiDelegate::DoPrepare
    // when trying to understand if all nodes are supported by the target
    // accelerators.
    // The key is the index of the first node in the partition.
    // Couldn't use unique_ptr because of problems building on gcc
    std::unordered_map<int, NNAPIDelegateKernel*> delegate_state_cache;

    ~Data();

    // Caches an initialised NNAPIDelegateKernel.
    void CacheDelegateKernel(const TfLiteDelegateParams* delegate_params,
                             NNAPIDelegateKernel* delegate_state);
    // Returns a cached NNAPIDelegateKernel if available.
    absl::optional<NNAPIDelegateKernel*> GetCachedDelegateKernel(
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
