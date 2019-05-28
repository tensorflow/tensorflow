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

#include "tensorflow/lite/c/c_api_internal.h"

namespace tflite {

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
  };

  // Uses default options.
  StatefulNnApiDelegate();

  // The constructor that accepts options from user.
  explicit StatefulNnApiDelegate(Options options);

  ~StatefulNnApiDelegate() = default;

  // Returns the delegate options.
  static const Options GetOptions(TfLiteDelegate* delegate);

 private:
  // Encapsulates all delegate data.
  struct Data {
    // Preferred Power/perf trade-off.
    Options::ExecutionPreference execution_preference;
    // Selected NNAPI accelerator name.
    std::string accelerator_name;
  };

  // Implements TfLiteDelegate::Prepare. Please refer to TFLiteDelegate
  // documentation for more info.
  static TfLiteStatus DoPrepare(TfLiteContext* context,
                                TfLiteDelegate* delegate);

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
