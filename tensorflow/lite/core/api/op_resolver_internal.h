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
#ifndef TENSORFLOW_LITE_CORE_API_OP_RESOLVER_INTERNAL_H_
#define TENSORFLOW_LITE_CORE_API_OP_RESOLVER_INTERNAL_H_

/// \file
///
/// This header op_resolver_internal.h exists so that we can have fine-grained
/// access control on the MayContainUserDefinedOps method
/// and registration_externals_cache_ member.

#include <memory>

#include "tensorflow/lite/core/api/op_resolver.h"

namespace tflite {

class OpResolverInternal {
 public:
  OpResolverInternal() = delete;

  static bool MayContainUserDefinedOps(const OpResolver& op_resolver) {
    return op_resolver.MayContainUserDefinedOps();
  }

  // Get a shared_ptr to the OperatorsCache from an OpResolver.
  // This is used to allow the InterpreterBuilder and OpResolver to share
  // the same OperatorsCache, so that the Operator objects in it can persist
  // for the lifetimes of both the InterpreterBuilder and OpResolver.
  static std::shared_ptr<::tflite::internal::OperatorsCache> GetSharedCache(
      const ::tflite::OpResolver& op_resolver) {
    return op_resolver.registration_externals_cache_;
  }
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_API_OP_RESOLVER_INTERNAL_H_
