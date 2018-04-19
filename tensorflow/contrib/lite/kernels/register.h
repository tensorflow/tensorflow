/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_REGISTER_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_REGISTER_H_

#include <unordered_map>
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace ops {
namespace builtin {

class BuiltinOpResolver : public OpResolver {
 public:
  BuiltinOpResolver();
  TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const override;
  TfLiteRegistration* FindOp(const char* op) const override;
  void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration);
  void AddCustom(const char* name, TfLiteRegistration* registration);

 private:
  struct BuiltinOperatorHasher {
    size_t operator()(const tflite::BuiltinOperator& x) const {
      return std::hash<size_t>()(static_cast<size_t>(x));
    }
  };
  std::unordered_map<tflite::BuiltinOperator, TfLiteRegistration*,
                     BuiltinOperatorHasher>
      builtins_;
  std::unordered_map<std::string, TfLiteRegistration*> custom_ops_;
};

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_BUILTIN_KERNELS_H
