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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_MUTABLE_OP_RESOLVER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_MUTABLE_OP_RESOLVER_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/experimental/micro/compatibility.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef TFLITE_REGISTRATIONS_MAX
#define TFLITE_REGISTRATIONS_MAX (128)
#endif

namespace tflite {

// Op versions discussed in this file are enumerated here:
// tensorflow/lite/tools/versioning/op_version.cc

class MicroMutableOpResolver : public OpResolver {
 public:
  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                   int version) const override;
  const TfLiteRegistration* FindOp(const char* op, int version) const override;

  // Add a builtin op which supports only version 1.
  void AddBuiltin(tflite::BuiltinOperator op,
                  const TfLiteRegistration* registration);

  // Add a builtin op which supports the specified version(s).
  void AddBuiltin(tflite::BuiltinOperator op,
                  const TfLiteRegistration* registration,
                  const int* supported_versions, int supported_versions_len);

  // Add a custom op which supports only version 1.
  void AddCustom(const char* name, const TfLiteRegistration* registration);

  // Add a custom op which supports the specified version(s).
  void AddCustom(const char* name, const TfLiteRegistration* registration,
                 const int* supported_versions, int supported_versions_len);

 private:
  TfLiteRegistration registrations_[TFLITE_REGISTRATIONS_MAX];
  int registrations_len_ = 0;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_MUTABLE_OP_RESOLVER_H_
