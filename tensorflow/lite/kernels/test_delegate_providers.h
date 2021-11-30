/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_TEST_DELEGATE_PROVIDERS_H_
#define TENSORFLOW_LITE_KERNELS_TEST_DELEGATE_PROVIDERS_H_

#include <vector>

#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {
// A utility class to provide TfLite delegate creations for kernel tests. The
// options of a particular delegate could be specified from commandline flags by
// using the delegate provider registrar as implemented in lite/tools/delegates
// directory.
class KernelTestDelegateProviders {
 public:
  // Returns a global KernelTestDelegateProviders instance.
  static KernelTestDelegateProviders* Get();

  KernelTestDelegateProviders();

  // Initialize delegate-related parameters from commandline arguments and
  // returns true if successful.
  bool InitFromCmdlineArgs(int* argc, const char** argv);

  // This provides a way to overwrite parameter values programmatically before
  // creating TfLite delegates. Note, changes to the returned ToolParams will
  // have a global impact on creating TfLite delegates.
  // If a local-only change is preferred, recommend using the following workflow
  // create TfLite delegates via delegate providers:
  // tools::ToolParams local_params;
  // local_params.Merge(KernelTestDelegateProviders::Get()->ConstParams());
  // Overwrite params in local_params by calling local_params.Set<...>(...);
  // Get TfLite delegates via
  // KernelTestDelegateProviders::Get()->CreateAllDelegates(local_params);
  tools::ToolParams* MutableParams() { return &params_; }
  const tools::ToolParams& ConstParams() const { return params_; }

  // Create a list of TfLite delegates based on the provided parameters
  // `params`.
  std::vector<tools::ProvidedDelegateList::ProvidedDelegate> CreateAllDelegates(
      const tools::ToolParams& params) const {
    tools::ProvidedDelegateList util;
    return util.CreateAllRankedDelegates(params);
  }

  // Similar to the above, but creating a list of TfLite delegates based on what
  // have been initialized (i.e. 'params_').
  std::vector<tools::ProvidedDelegateList::ProvidedDelegate>
  CreateAllDelegates() const {
    return delegate_list_util_.CreateAllRankedDelegates();
  }

  // An option name to use Simple Memory Allocator.
  static constexpr char kUseSimpleAllocator[] = "use_simple_allocator";

 private:
  // Contain delegate-related parameters that are initialized from command-line
  // flags.
  tools::ToolParams params_;

  // A helper to create TfLite delegates.
  tools::ProvidedDelegateList delegate_list_util_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_TEST_DELEGATE_PROVIDERS_H_
