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
#include "tensorflow/lite/kernels/test_delegate_providers.h"

#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
/*static*/ KernelTestDelegateProviders* KernelTestDelegateProviders::Get() {
  static KernelTestDelegateProviders* const providers =
      new KernelTestDelegateProviders();
  return providers;
}

KernelTestDelegateProviders::KernelTestDelegateProviders()
    : delegate_list_util_(&params_) {
  delegate_list_util_.AddAllDelegateParams();
}

bool KernelTestDelegateProviders::InitFromCmdlineArgs(int* argc,
                                                      const char** argv) {
  std::vector<tflite::Flag> flags;
  delegate_list_util_.AppendCmdlineFlags(&flags);

  bool parse_result = tflite::Flags::Parse(argc, argv, flags);
  if (!parse_result || params_.Get<bool>("help")) {
    std::string usage = Flags::Usage(argv[0], flags);
    TFLITE_LOG(ERROR) << usage;
    // Returning false intentionally when "--help=true" is specified so that
    // the caller could check the return value to decide stopping the execution.
    parse_result = false;
  }
  return parse_result;
}
}  // namespace tflite
