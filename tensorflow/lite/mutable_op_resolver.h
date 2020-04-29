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

#include <string>
#include <unordered_map>

#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/util.h"

namespace tflite {

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
    return CombineHashes({a, b});
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
  void AddBuiltin(tflite::BuiltinOperator op,
                  const TfLiteRegistration* registration, int version = 1);
  void AddBuiltin(tflite::BuiltinOperator op,
                  const TfLiteRegistration* registration, int min_version,
                  int max_version);
  void AddCustom(const char* name, const TfLiteRegistration* registration,
                 int version = 1);
  void AddCustom(const char* name, const TfLiteRegistration* registration,
                 int min_version, int max_version);
  void AddAll(const MutableOpResolver& other);

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

#endif  // TENSORFLOW_LITE_MUTABLE_OP_RESOLVER_H_
