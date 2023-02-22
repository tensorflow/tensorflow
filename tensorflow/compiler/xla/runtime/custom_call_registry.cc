/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace xla {
namespace runtime {

void DynamicCustomCallRegistry::Register(
    std::unique_ptr<CustomCall> custom_call) {
  std::string_view name = custom_call->name();
  auto emplaced = custom_calls_.try_emplace(name, std::move(custom_call));
  assert(emplaced.second && "duplicate custom call registration");
  (void)emplaced;
}

CustomCall* DynamicCustomCallRegistry::Find(std::string_view callee) const {
  auto it = custom_calls_.find(callee);
  if (it == custom_calls_.end()) return nullptr;
  return it->second.get();
}

void DirectCustomCallRegistry::Register(std::string_view name,
                                        DirectCustomCall custom_call) {
  auto emplaced = custom_calls_.try_emplace(name, std::move(custom_call));
  assert(emplaced.second && "duplicate custom call registration");
  (void)emplaced;
}

void DirectCustomCallRegistry::ForEach(
    std::function<void(std::string_view, DirectCustomCall)> f) const {
  for (auto& kv : custom_calls_) f(kv.first(), kv.second);
}

}  // namespace runtime
}  // namespace xla
