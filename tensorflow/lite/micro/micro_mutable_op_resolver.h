/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_MICRO_MUTABLE_OP_RESOLVER_H_
#define TENSORFLOW_LITE_MICRO_MICRO_MUTABLE_OP_RESOLVER_H_

#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// TODO(b/151245712) TODO(b/149408647): remove any version once we no longer
// support op versions in the API or we switch most users and AllOpsResolver to
// the new selective registration API, whichever seems more appropriate.
inline int MicroOpResolverAnyVersion() { return 0; }

template <unsigned int tOpCount>
class MicroMutableOpResolver : public MicroOpResolver {
 public:
  explicit MicroMutableOpResolver(ErrorReporter* error_reporter = nullptr)
      : error_reporter_(error_reporter) {}

  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const override {
    if (op == BuiltinOperator_CUSTOM) return nullptr;

    for (unsigned int i = 0; i < registrations_len_; ++i) {
      const TfLiteRegistration& registration = registrations_[i];
      if (registration.builtin_code == op) {
        return &registration;
      }
    }
    return nullptr;
  }

  const TfLiteRegistration* FindOp(const char* op) const override {
    for (unsigned int i = 0; i < registrations_len_; ++i) {
      const TfLiteRegistration& registration = registrations_[i];
      if ((registration.builtin_code == BuiltinOperator_CUSTOM) &&
          (strcmp(registration.custom_name, op) == 0)) {
        return &registration;
      }
    }
    return nullptr;
  }

  MicroOpResolver::BuiltinParseFunction GetOpDataParser(
      tflite::BuiltinOperator) const override {
    // TODO(b/149408647): Replace with the more selective builtin parser.
    return ParseOpData;
  }

  TfLiteStatus AddBuiltin(tflite::BuiltinOperator op,
                          TfLiteRegistration* registration,
                          int version = 1) override {
    if (registrations_len_ >= tOpCount) {
      if (error_reporter_) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Couldn't register builtin op #%d, resolver size "
                             "is too small (%d)",
                             op, tOpCount);
      }
      return kTfLiteError;
    }

    if (FindOp(op) != nullptr) {
      if (error_reporter_ != nullptr) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Registering multiple versions of the same op is "
                             "not supported (Op: #%d, version: %d).",
                             op, version);
      }
      return kTfLiteError;
    }

    TfLiteRegistration* new_registration = &registrations_[registrations_len_];
    registrations_len_ += 1;

    *new_registration = *registration;
    new_registration->builtin_code = op;
    new_registration->version = version;

    return kTfLiteOk;
  }

  TfLiteStatus AddCustom(const char* name, TfLiteRegistration* registration,
                         int version = 1) {
    printf("registrations_len_: %d\n", registrations_len_);
    if (registrations_len_ >= tOpCount) {
      if (error_reporter_) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Couldn't register custom op '%s', resolver size is too small (%d)",
            name, tOpCount);
      }
      return kTfLiteError;
    }

    if (FindOp(name) != nullptr) {
      if (error_reporter_ != nullptr) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Registering multiple versions of the same op is "
                             "not supported (Op: %s, version: %d).",
                             name, version);
      }
      return kTfLiteError;
    }

    TfLiteRegistration* new_registration = &registrations_[registrations_len_];
    registrations_len_ += 1;

    *new_registration = *registration;
    new_registration->builtin_code = BuiltinOperator_CUSTOM;
    new_registration->custom_name = name;
    new_registration->version = version;

    return kTfLiteOk;
  }

  unsigned int GetRegistrationLength() { return registrations_len_; }

 private:
  TfLiteRegistration registrations_[tOpCount];
  unsigned int registrations_len_ = 0;
  ErrorReporter* error_reporter_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

};  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_MUTABLE_OP_RESOLVER_H_
