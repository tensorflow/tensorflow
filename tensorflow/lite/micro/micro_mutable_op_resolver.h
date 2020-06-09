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

#include <cstdio>
#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

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
      BuiltinOperator op) const override {
    TFLITE_DCHECK(num_buitin_ops_ <= tOpCount);
    for (unsigned int i = 0; i < num_buitin_ops_; ++i) {
      if (builtin_codes_[i] == op) return builtin_parsers_[i];
    }
    return nullptr;
  }

  // Registers a Custom Operator with the MicroOpResolver.
  //
  // Only the first call for a given name will be successful. i.e. if this
  // function is called again for a previously added Custom Operator, the
  // MicroOpResolver will be unchanged and this function will return
  // kTfLiteError.
  TfLiteStatus AddCustom(const char* name, TfLiteRegistration* registration) {
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
                             "Calling AddCustom for the same op more than once "
                             "is not supported (Op: %s).",
                             name);
      }
      return kTfLiteError;
    }

    TfLiteRegistration* new_registration = &registrations_[registrations_len_];
    registrations_len_ += 1;

    *new_registration = *registration;
    new_registration->builtin_code = BuiltinOperator_CUSTOM;
    new_registration->custom_name = name;
    return kTfLiteOk;
  }

  // Registers a Builtin Operator with the MicroOpResolver.
  //
  // Only the first call for a given BuiltinOperator enum will be successful.
  // i.e. if this function is called again for a previously added
  // BuiltinOperator, the MicroOpResolver will be unchanged and this function
  // will return kTfLiteError.
  //
  // TODO(b/149408647): remove this API once the BuiltinOperator specific Add
  // functions are fully implemented.
  TfLiteStatus AddBuiltin(tflite::BuiltinOperator op,
                          TfLiteRegistration* registration) {
    TFLITE_DCHECK(registration != nullptr);
    // For code that is not switched over to the new selective registration of
    // the parse function, we pass in ParseOpData. This allows for backwards
    // compatibility.
    return AddBuiltin(op, *registration, ParseOpData);
  }

  // The Add* functions below add the various Builtin operators to the
  // MicroMutableOpResolver object.
  //
  // This API is currently experimental (and only supported for a small subset
  // of operators). It will soon be preferred over the AddBuiltin function for
  // the following reason:
  //  * If all calls to AddBuiltin for an application use this API, the code
  //    size will be smaller by 5-8K (compared to the using the AddBuiltin
  //    override).

  TfLiteStatus AddConv2D() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function once cl/313453102 lands.
    return AddBuiltin(BuiltinOperator_CONV_2D,
                      *tflite::ops::micro::Register_CONV_2D(), ParseOpData);
  }

  TfLiteStatus AddDequantize() {
    return AddBuiltin(BuiltinOperator_DEQUANTIZE,
                      *tflite::ops::micro::Register_DEQUANTIZE(),
                      ParseDequantize);
  }

  TfLiteStatus AddFullyConnected() {
    return AddBuiltin(BuiltinOperator_FULLY_CONNECTED,
                      *tflite::ops::micro::Register_FULLY_CONNECTED(),
                      ParseFullyConnected);
  }

  TfLiteStatus AddLogistic() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function once cl/313453102 lands.
    return AddBuiltin(BuiltinOperator_LOGISTIC,
                      *tflite::ops::micro::Register_LOGISTIC(), ParseOpData);
  }

  TfLiteStatus AddQuantize() {
    return AddBuiltin(BuiltinOperator_QUANTIZE,
                      *tflite::ops::micro::Register_QUANTIZE(), ParseQuantize);
  }

  TfLiteStatus AddReshape() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function once cl/313453102 lands.
    return AddBuiltin(BuiltinOperator_RESHAPE,
                      *tflite::ops::micro::Register_RESHAPE(), ParseOpData);
  }

  TfLiteStatus AddSoftmax() {
    return AddBuiltin(BuiltinOperator_SOFTMAX,
                      *tflite::ops::micro::Register_SOFTMAX(), ParseSoftmax);
  }

  TfLiteStatus AddSvdf() {
    return AddBuiltin(BuiltinOperator_SVDF,
                      *tflite::ops::micro::Register_SVDF(), ParseSvdf);
  }

  unsigned int GetRegistrationLength() { return registrations_len_; }

 private:
  TfLiteStatus AddBuiltin(tflite::BuiltinOperator op,
                          const TfLiteRegistration& registration,
                          MicroOpResolver::BuiltinParseFunction parser) {
    if (op == BuiltinOperator_CUSTOM) {
      if (error_reporter_ != nullptr) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Invalid parameter BuiltinOperator_CUSTOM to the "
                             "AddBuiltin function.");
      }
      return kTfLiteError;
    }

    if (FindOp(op) != nullptr) {
      if (error_reporter_ != nullptr) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Calling AddBuiltin with the same op more than "
                             "once is not supported (Op: #%d).",
                             op);
      }
      return kTfLiteError;
    }

    if (registrations_len_ >= tOpCount) {
      if (error_reporter_) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Couldn't register builtin op #%d, resolver size "
                             "is too small (%d).",
                             op, tOpCount);
      }
      return kTfLiteError;
    }

    registrations_[registrations_len_] = registration;
    // Strictly speaking, the builtin_code is not necessary for TFLM but filling
    // it in regardless.
    registrations_[registrations_len_].builtin_code = op;
    registrations_len_++;

    builtin_codes_[num_buitin_ops_] = op;
    builtin_parsers_[num_buitin_ops_] = parser;
    num_buitin_ops_++;

    return kTfLiteOk;
  }

  TfLiteRegistration registrations_[tOpCount];
  unsigned int registrations_len_ = 0;

  // Arrays (and counter) to store the builtin codes and their corresponding
  // parse functions as these are registered with the Op Resolver.
  BuiltinOperator builtin_codes_[tOpCount];
  MicroOpResolver::BuiltinParseFunction builtin_parsers_[tOpCount];
  unsigned int num_buitin_ops_ = 0;

  ErrorReporter* error_reporter_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

};  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_MUTABLE_OP_RESOLVER_H_
