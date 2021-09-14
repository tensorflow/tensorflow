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

#include "ral/ral_registry.h"

#include <assert.h>

#include <mutex>  // NOLINT
#include <unordered_map>

#include "ral/ral_logging.h"

namespace mlir {
namespace disc_ral {

struct FunctionRegistry::Impl {
  std::mutex mu;
  std::unordered_map<std::string, ral_func_t> funcs;
};

FunctionRegistry::FunctionRegistry() : impl_(new FunctionRegistry::Impl) {}

FunctionRegistry::~FunctionRegistry() {}

FunctionRegistry& FunctionRegistry::Global() {
  static FunctionRegistry registry;  // NOLINT
  return registry;
}

bool FunctionRegistry::Register(const std::string& name, ral_func_t func) {
  DISC_VLOG(2) << "register function " << name << " to the registry";
  std::lock_guard<std::mutex> lock(impl_->mu);
  auto it = impl_->funcs.emplace(name, func);
  assert(it.second && "duplicated key is not allowed");
  return it.second;
}

ral_func_t FunctionRegistry::Find(const std::string& name) {
  ral_func_t func;
  std::lock_guard<std::mutex> lock(impl_->mu);
  auto it = impl_->funcs.find(name);
  if (it != impl_->funcs.end()) {
    func = it->second;
  }
  return func;
}

}  // namespace disc_ral
}  // namespace mlir
