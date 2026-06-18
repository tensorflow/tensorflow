/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/mutable_op_resolver.h"

#include <unordered_map>
#include <utility>

#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/api/op_resolver_internal.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

const TfLiteRegistration* MutableOpResolver::FindOp(tflite::BuiltinOperator op,
                                                    int version) const {
  auto it = builtins_.find(std::make_pair(op, version));
  if (it != builtins_.end()) {
    return &it->second;
  }
  for (const OpResolver* other : other_op_resolvers_) {
    const TfLiteRegistration* result = other->FindOp(op, version);
    if (result != nullptr) {
      return result;
    }
  }
  return nullptr;
}

const TfLiteRegistration* MutableOpResolver::FindOp(const char* op,
                                                    int version) const {
  auto it = custom_ops_.find(std::make_pair(op, version));
  if (it != custom_ops_.end()) {
    return &it->second;
  }
  for (const OpResolver* other : other_op_resolvers_) {
    const TfLiteRegistration* result = other->FindOp(op, version);
    if (result != nullptr) {
      return result;
    }
  }
  return nullptr;
}

void MutableOpResolver::AddBuiltin(tflite::BuiltinOperator op,
                                   const TfLiteRegistration* registration,
                                   int version) {
  if (registration == nullptr) {
    // Under certain conditions, builtin TfLiteRegistration factory methods may
    // return null in the client library. This is generally benign, and we
    // silently suppress resulting AddBuiltin calls here.
    return;
  }
  TfLiteRegistration new_registration = *registration;
  new_registration.custom_name = nullptr;
  new_registration.builtin_code = op;
  new_registration.version = version;
  auto op_key = std::make_pair(op, version);
  builtins_[op_key] = new_registration;
  // The builtin op that is being added may be one that is not supported by
  // tflite::ops::builtin::BuiltinOpResolver. Or the TfLiteRegistration for this
  // builtin may be different than the one that BuiltinOpResolver would use,
  // which could lead to different semantics. Both of those cases are considered
  // "user defined ops".
  may_directly_contain_user_defined_ops_ = true;
}

void MutableOpResolver::AddBuiltin(tflite::BuiltinOperator op,
                                   const TfLiteRegistration* registration,
                                   int min_version, int max_version) {
  for (int version = min_version; version <= max_version; ++version) {
    AddBuiltin(op, registration, version);
  }
}

void MutableOpResolver::AddCustom(const char* name,
                                  const TfLiteRegistration* registration,
                                  int version) {
  TfLiteRegistration new_registration = *registration;
  new_registration.builtin_code = BuiltinOperator_CUSTOM;
  new_registration.custom_name = name;
  new_registration.version = version;
  auto op_key = std::make_pair(name, version);
  custom_ops_[op_key] = new_registration;
  may_directly_contain_user_defined_ops_ = true;
}

void MutableOpResolver::AddCustom(const char* name,
                                  const TfLiteRegistration* registration,
                                  int min_version, int max_version) {
  for (int version = min_version; version <= max_version; ++version) {
    AddCustom(name, registration, version);
  }
}

void MutableOpResolver::AddAll(const MutableOpResolver& other) {
  // map::insert does not replace existing elements, and map::insert_or_assign
  // wasn't added until C++17.
  for (const auto& other_builtin : other.builtins_) {
    builtins_[other_builtin.first] = other_builtin.second;
  }
  for (const auto& other_custom_op : other.custom_ops_) {
    custom_ops_[other_custom_op.first] = other_custom_op.second;
  }
  other_op_resolvers_.insert(other_op_resolvers_.begin(),
                             other.other_op_resolvers_.begin(),
                             other.other_op_resolvers_.end());
}

void MutableOpResolver::ChainOpResolver(const OpResolver* other) {
  other_op_resolvers_.push_back(other);
}

bool MutableOpResolver::MayContainUserDefinedOps() const {
  if (may_directly_contain_user_defined_ops_) {
    return true;
  }
  for (const OpResolver* other : other_op_resolvers_) {
    if (OpResolverInternal::MayContainUserDefinedOps(*other)) {
      return true;
    }
  }
  return false;
}

}  // namespace tflite
