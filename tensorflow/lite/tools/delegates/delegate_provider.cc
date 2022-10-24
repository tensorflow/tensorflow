/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

#include <algorithm>
#include <string>
#include <utility>

namespace tflite {
namespace tools {

TfLiteDelegatePtr CreateNullDelegate() {
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

void ProvidedDelegateList::AddAllDelegateParams() const {
  for (const auto& provider : providers_) {
    params_->Merge(provider->DefaultParams());
  }
}

void ProvidedDelegateList::AppendCmdlineFlags(std::vector<Flag>& flags) const {
  for (const auto& provider : providers_) {
    auto delegate_flags = provider->CreateFlags(params_);
    flags.insert(flags.end(), delegate_flags.begin(), delegate_flags.end());
  }
}

void ProvidedDelegateList::RemoveCmdlineFlag(std::vector<Flag>& flags,
                                             const std::string& name) const {
  decltype(flags.begin()) it;
  for (it = flags.begin(); it < flags.end();) {
    if (it->GetFlagName() == name) {
      it = flags.erase(it);
    } else {
      ++it;
    }
  }
}

std::vector<ProvidedDelegateList::ProvidedDelegate>
ProvidedDelegateList::CreateAllRankedDelegates(const ToolParams& params) const {
  std::vector<ProvidedDelegateList::ProvidedDelegate> delegates;
  for (const auto& provider : providers_) {
    auto ptr_rank = provider->CreateRankedTfLiteDelegate(params);
    // It's possible that a delegate of certain type won't be created as
    // user-specified tool params tells not to.
    if (ptr_rank.first == nullptr) continue;
    TFLITE_LOG(INFO) << provider->GetName() << " delegate created.";

    ProvidedDelegateList::ProvidedDelegate info;
    info.provider = provider.get();
    info.delegate = std::move(ptr_rank.first);
    info.rank = ptr_rank.second;
    delegates.emplace_back(std::move(info));
  }

  std::sort(delegates.begin(), delegates.end(),
            [](const ProvidedDelegateList::ProvidedDelegate& a,
               const ProvidedDelegateList::ProvidedDelegate& b) {
              return a.rank < b.rank;
            });

  return delegates;
}

}  // namespace tools
}  // namespace tflite
