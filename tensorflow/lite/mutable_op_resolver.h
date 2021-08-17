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
#ifndef TENSORFLOW_LITE_MUTABLE_OP_RESOLVER_H_
#define TENSORFLOW_LITE_MUTABLE_OP_RESOLVER_H_

#include <stddef.h>

#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

namespace tflite {

// Some versions of gcc don't support partial specialization in class scope,
// so these are defined in a namescope.
namespace op_resolver_hasher {
template <typename V>
struct ValueHasher {
  size_t operator()(const V& v) const { return std::hash<V>()(v); }
};

template <>
struct ValueHasher<tflite::BuiltinOperator> {
  size_t operator()(const tflite::BuiltinOperator& v) const {
    return std::hash<int>()(static_cast<int>(v));
  }
};

template <typename T>
struct OperatorKeyHasher {
  size_t operator()(const T& x) const {
    size_t a = ValueHasher<typename T::first_type>()(x.first);
    size_t b = ValueHasher<typename T::second_type>()(x.second);
    return CombineHashes({a, b});
  }
};
}  // namespace op_resolver_hasher

/// An OpResolver that is mutable, also used as the op in gen_op_registration.
/// A typical usage:
///   MutableOpResolver resolver;
///   resolver.AddBuiltin(BuiltinOperator_ADD, Register_ADD());
///   resolver.AddCustom("CustomOp", Register_CUSTOM_OP());
///   InterpreterBuilder(model, resolver)(&interpreter);
class MutableOpResolver : public OpResolver {
 public:
  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                   int version) const override;
  const TfLiteRegistration* FindOp(const char* op, int version) const override;

  /// Registers the specified `version` of the specified builtin operator `op`.
  /// Replaces any previous registration for the same operator version.
  void AddBuiltin(tflite::BuiltinOperator op,
                  const TfLiteRegistration* registration, int version = 1);

  /// Registers the specified version range (versions `min_version` to
  /// `max_version`, inclusive) of the specified builtin operator `op`.
  /// Replaces any previous registration for the same operator version.
  void AddBuiltin(tflite::BuiltinOperator op,
                  const TfLiteRegistration* registration, int min_version,
                  int max_version);

  /// Registers the specified `version` of the specified builtin operator `op`.
  /// Replaces any previous registration for the same operator version.
  void AddCustom(const char* name, const TfLiteRegistration* registration,
                 int version = 1);

  /// Registers the specified version range (versions `min_version` to
  /// `max_version`, inclusive) of the specified custom operator `name`.
  /// Replaces any previous registration for the same operator version.
  void AddCustom(const char* name, const TfLiteRegistration* registration,
                 int min_version, int max_version);

  /// Registers all operator versions supported by another MutableOpResolver.
  /// Replaces any previous registrations for the same operator versions,
  /// except that registrations made with `AddBuiltin` or `AddCustom` always
  /// take precedence over registrations made with `ChainOpResolver`.
  void AddAll(const MutableOpResolver& other);

 protected:
  /// Registers all operator versions supported by another OpResolver,
  /// except any already registered in this MutableOpResolver.
  /// `other` must point to an OpResolver whose lifetime is at least as long
  /// as the lifetime of the MutableOpResolver pointed to by `this`.
  /// The OpResolver pointed to by `other` should not be modified during the
  /// lifetime of this MutableOpResolver.
  void ChainOpResolver(const OpResolver* other);

  /// True if this OpResolver itself (as opposed to chained op resolvers
  /// registed with ChainOpResolver) may contain user defined ops.
  ///
  /// By "user defined" ops, we mean any op definitions other than those
  /// contained in tflite::ops::builtin::BuiltinOpResolver.
  bool may_directly_contain_user_defined_ops_ = false;

 private:
  bool MayContainUserDefinedOps() const override;

  typedef std::pair<tflite::BuiltinOperator, int> BuiltinOperatorKey;
  typedef std::pair<std::string, int> CustomOperatorKey;

  std::unordered_map<BuiltinOperatorKey, TfLiteRegistration,
                     op_resolver_hasher::OperatorKeyHasher<BuiltinOperatorKey> >
      builtins_;
  std::unordered_map<CustomOperatorKey, TfLiteRegistration,
                     op_resolver_hasher::OperatorKeyHasher<CustomOperatorKey> >
      custom_ops_;
  std::vector<const OpResolver*> other_op_resolvers_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MUTABLE_OP_RESOLVER_H_
