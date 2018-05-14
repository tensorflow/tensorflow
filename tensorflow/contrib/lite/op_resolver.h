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
#ifndef TENSORFLOW_CONTRIB_LITE_OP_RESOLVER_H_
#define TENSORFLOW_CONTRIB_LITE_OP_RESOLVER_H_

#include <unordered_map>
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"

namespace tflite {

// Abstract interface that returns TfLiteRegistrations given op codes or custom
// op names. This is the mechanism that ops being referenced in the flatbuffer
// model are mapped to executable function pointers (TfLiteRegistrations).
class OpResolver {
 public:
  // Finds the op registration for a builtin operator by enum code.
  virtual const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                           int version) const = 0;
  // Finds the op registration of a custom operator by op name.
  virtual const TfLiteRegistration* FindOp(const char* op,
                                           int version) const = 0;
  virtual ~OpResolver() {}
};

// Some versions of gcc doesn't support partial specialization in class scope,
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
    // Hash combinator used by TensorFlow core.
    return a ^ (b + 0x9e3779b97f4a7800ULL + (a << 10) + (a >> 4));
  }
};
}  // namespace op_resolver_hasher

// An OpResolver that is mutable, also used as the op in gen_op_registration.
// A typical usage:
//   MutableOpResolver resolver;
//   resolver.AddBuiltin(BuiltinOperator_ADD, Register_ADD());
//   resolver.AddCustom("CustomOp", Register_CUSTOM_OP());
//   InterpreterBuilder(model, resolver)(&interpreter);
class MutableOpResolver : public OpResolver {
 public:
  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                   int version) const override;
  const TfLiteRegistration* FindOp(const char* op, int version) const override;
  void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration,
                  int min_version = 1, int max_version = 1);
  void AddCustom(const char* name, TfLiteRegistration* registration,
                 int min_version = 1, int max_version = 1);

 private:
  typedef std::pair<tflite::BuiltinOperator, int> BuiltinOperatorKey;
  typedef std::pair<std::string, int> CustomOperatorKey;

  std::unordered_map<BuiltinOperatorKey, TfLiteRegistration,
                     op_resolver_hasher::OperatorKeyHasher<BuiltinOperatorKey> >
      builtins_;
  std::unordered_map<CustomOperatorKey, TfLiteRegistration,
                     op_resolver_hasher::OperatorKeyHasher<CustomOperatorKey> >
      custom_ops_;
};

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_OP_RESOLVER_H_
