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

KernelTestDelegateProviders::KernelTestDelegateProviders() {
  for (const auto& one : tools::GetRegisteredDelegateProviders()) {
    params_.Merge(one->DefaultParams());
  }
}

bool KernelTestDelegateProviders::InitFromCmdlineArgs(int* argc,
                                                      const char** argv) {
  std::vector<tflite::Flag> flags;
  for (const auto& one : tools::GetRegisteredDelegateProviders()) {
    auto one_flags = one->CreateFlags(&params_);
    flags.insert(flags.end(), one_flags.begin(), one_flags.end());
  }

  // Note: when "--help" is passed, the 'Parse' function will return false.
  // TODO(b/181868587): The above logic to print out the all supported flags is
  // not intuitive, so considering adding the "--help" flag explicitly.
  const bool parse_result = tflite::Flags::Parse(argc, argv, flags);
  if (!parse_result) {
    std::string usage = Flags::Usage(argv[0], flags);
    TFLITE_LOG(ERROR) << usage;
  }
  return parse_result;
}

std::vector<tools::TfLiteDelegatePtr>
KernelTestDelegateProviders::CreateAllDelegates(
    const tools::ToolParams& params) const {
  std::vector<tools::TfLiteDelegatePtr> delegates;
  for (const auto& one : tools::GetRegisteredDelegateProviders()) {
    auto ptr = one->CreateTfLiteDelegate(params);
    // It's possible that a delegate of certain type won't be created as
    // user-specified benchmark params tells not to.
    if (ptr == nullptr) continue;
    delegates.emplace_back(std::move(ptr));
    TFLITE_LOG(INFO) << one->GetName() << " delegate is created.";
  }
  return delegates;
}
}  // namespace tflite
