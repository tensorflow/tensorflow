/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_FRAMEWORK_GRAD_OP_REGISTRY_H_
#define TENSORFLOW_CC_FRAMEWORK_GRAD_OP_REGISTRY_H_

#include <unordered_map>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"

namespace tensorflow {
namespace ops {

/// GradFunc is the signature for all gradient functions in GradOpRegistry.
/// Implementations should add operations to compute the gradient outputs of
/// 'op' (returned in 'grad_outputs') using 'scope' and 'grad_inputs'.
typedef Status (*GradFunc)(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs);

/// GradOpRegistry maintains a static registry of gradient functions.
/// Gradient functions are indexed in the registry by the forward op name (i.e.
/// "MatMul" -> MatMulGrad func).
class GradOpRegistry {
 public:
  /// Registers 'func' as the gradient function for 'op'.
  /// Returns true if registration was successful, check fails otherwise.
  bool Register(const string& op, GradFunc func);

  /// Sets 'func' to the gradient function for 'op' and returns Status OK if
  /// the gradient function for 'op' exists in the registry.
  /// Note that 'func' can be null for ops that have registered no-gradient with
  /// the registry.
  /// Returns error status otherwise.
  Status Lookup(const string& op, GradFunc* func) const;

  /// Returns a pointer to the global gradient function registry.
  static GradOpRegistry* Global();

 private:
  std::unordered_map<string, GradFunc> registry_;
};

}  // namespace ops

// Macros used to define gradient functions for ops.
#define REGISTER_GRADIENT_OP(name, fn) \
  REGISTER_GRADIENT_OP_UNIQ_HELPER(__COUNTER__, name, fn)

#define REGISTER_NO_GRADIENT_OP(name) \
  REGISTER_GRADIENT_OP_UNIQ_HELPER(__COUNTER__, name, nullptr)

#define REGISTER_GRADIENT_OP_UNIQ_HELPER(ctr, name, fn) \
  REGISTER_GRADIENT_OP_UNIQ(ctr, name, fn)

#define REGISTER_GRADIENT_OP_UNIQ(ctr, name, fn) \
  static bool unused_ret_val_##ctr =             \
      ::tensorflow::ops::GradOpRegistry::Global()->Register(name, fn)

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_FRAMEWORK_GRAD_OP_REGISTRY_H_
