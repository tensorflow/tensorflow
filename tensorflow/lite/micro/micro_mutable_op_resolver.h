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

  // The Add* functions below add the various Builtin operators to the
  // MicroMutableOpResolver object.

  TfLiteStatus AddAbs() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_ABS, *tflite::ops::micro::Register_ABS(),
                      ParseOpData);
  }

  TfLiteStatus AddAdd() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_ADD, *tflite::ops::micro::Register_ADD(),
                      ParseOpData);
  }

  TfLiteStatus AddArgMax() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_ARG_MAX,
                      *tflite::ops::micro::Register_ARG_MAX(), ParseOpData);
  }

  TfLiteStatus AddArgMin() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_ARG_MIN,
                      *tflite::ops::micro::Register_ARG_MIN(), ParseOpData);
  }

  TfLiteStatus AddAveragePool2D() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D,
                      *tflite::ops::micro::Register_AVERAGE_POOL_2D(),
                      ParseOpData);
  }

  TfLiteStatus AddCeil() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_CEIL,
                      *tflite::ops::micro::Register_CEIL(), ParseOpData);
  }

  TfLiteStatus AddConcatenation() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_CONCATENATION,
                      *tflite::ops::micro::Register_CONCATENATION(),
                      ParseOpData);
  }

  TfLiteStatus AddConv2D() {
    return AddBuiltin(BuiltinOperator_CONV_2D,
                      *tflite::ops::micro::Register_CONV_2D(), ParseConv2D);
  }

  TfLiteStatus AddCos() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_COS, *tflite::ops::micro::Register_COS(),
                      ParseOpData);
  }

  TfLiteStatus AddDepthwiseConv2D() {
    return AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D,
                      *tflite::ops::micro::Register_DEPTHWISE_CONV_2D(),
                      ParseDepthwiseConv2D);
  }

  TfLiteStatus AddDequantize() {
    return AddBuiltin(BuiltinOperator_DEQUANTIZE,
                      *tflite::ops::micro::Register_DEQUANTIZE(),
                      ParseDequantize);
  }

  TfLiteStatus AddEqual() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_EQUAL,
                      *tflite::ops::micro::Register_EQUAL(), ParseOpData);
  }

  TfLiteStatus AddFloor() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_FLOOR,
                      *tflite::ops::micro::Register_FLOOR(), ParseOpData);
  }

  TfLiteStatus AddFullyConnected() {
    return AddBuiltin(BuiltinOperator_FULLY_CONNECTED,
                      *tflite::ops::micro::Register_FULLY_CONNECTED(),
                      ParseFullyConnected);
  }

  TfLiteStatus AddGreater() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_GREATER,
                      *tflite::ops::micro::Register_GREATER(), ParseOpData);
  }

  TfLiteStatus AddGreaterEqual() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_GREATER_EQUAL,
                      *tflite::ops::micro::Register_GREATER_EQUAL(),
                      ParseOpData);
  }

  TfLiteStatus AddL2Normalization() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_L2_NORMALIZATION,
                      *tflite::ops::micro::Register_L2_NORMALIZATION(),
                      ParseOpData);
  }

  TfLiteStatus AddLess() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_LESS,
                      *tflite::ops::micro::Register_LESS(), ParseOpData);
  }

  TfLiteStatus AddLessEqual() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_LESS_EQUAL,
                      *tflite::ops::micro::Register_LESS_EQUAL(), ParseOpData);
  }

  TfLiteStatus AddLog() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_LOG, *tflite::ops::micro::Register_LOG(),
                      ParseOpData);
  }

  TfLiteStatus AddLogicalAnd() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_LOGICAL_AND,
                      *tflite::ops::micro::Register_LOGICAL_AND(), ParseOpData);
  }

  TfLiteStatus AddLogicalNot() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_LOGICAL_NOT,
                      *tflite::ops::micro::Register_LOGICAL_NOT(), ParseOpData);
  }

  TfLiteStatus AddLogicalOr() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_LOGICAL_OR,
                      *tflite::ops::micro::Register_LOGICAL_OR(), ParseOpData);
  }

  TfLiteStatus AddLogistic() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_LOGISTIC,
                      *tflite::ops::micro::Register_LOGISTIC(), ParseOpData);
  }

  TfLiteStatus AddMaximum() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_MAXIMUM,
                      *tflite::ops::micro::Register_MAXIMUM(), ParseOpData);
  }

  TfLiteStatus AddMaxPool2D() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_MAX_POOL_2D,
                      *tflite::ops::micro::Register_MAX_POOL_2D(), ParseOpData);
  }

  TfLiteStatus AddMean() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_MEAN,
                      *tflite::ops::micro::Register_MEAN(), ParseOpData);
  }

  TfLiteStatus AddMinimum() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_MINIMUM,
                      *tflite::ops::micro::Register_MINIMUM(), ParseOpData);
  }

  TfLiteStatus AddMul() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_MUL, *tflite::ops::micro::Register_MUL(),
                      ParseOpData);
  }

  TfLiteStatus AddNeg() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_NEG, *tflite::ops::micro::Register_NEG(),
                      ParseOpData);
  }

  TfLiteStatus AddNotEqual() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_NOT_EQUAL,
                      *tflite::ops::micro::Register_NOT_EQUAL(), ParseOpData);
  }

  TfLiteStatus AddPack() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_PACK,
                      *tflite::ops::micro::Register_PACK(), ParseOpData);
  }

  TfLiteStatus AddPad() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_PAD, *tflite::ops::micro::Register_PAD(),
                      ParseOpData);
  }

  TfLiteStatus AddPadV2() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_PADV2,
                      *tflite::ops::micro::Register_PADV2(), ParseOpData);
  }

  TfLiteStatus AddPrelu() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_PRELU,
                      *tflite::ops::micro::Register_PRELU(), ParseOpData);
  }

  TfLiteStatus AddQuantize() {
    return AddBuiltin(BuiltinOperator_QUANTIZE,
                      *tflite::ops::micro::Register_QUANTIZE(), ParseQuantize);
  }

  TfLiteStatus AddRelu() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_RELU,
                      *tflite::ops::micro::Register_RELU(), ParseOpData);
  }

  TfLiteStatus AddRelu6() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_RELU6,
                      *tflite::ops::micro::Register_RELU6(), ParseOpData);
  }

  TfLiteStatus AddReshape() {
    return AddBuiltin(BuiltinOperator_RESHAPE,
                      *tflite::ops::micro::Register_RESHAPE(), ParseReshape);
  }

  TfLiteStatus AddResizeNearestNeighbor() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                      *tflite::ops::micro::Register_RESIZE_NEAREST_NEIGHBOR(),
                      ParseOpData);
  }

  TfLiteStatus AddRound() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_ROUND,
                      *tflite::ops::micro::Register_ROUND(), ParseOpData);
  }

  TfLiteStatus AddRsqrt() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_RSQRT,
                      *tflite::ops::micro::Register_RSQRT(), ParseOpData);
  }

  TfLiteStatus AddSin() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_SIN, *tflite::ops::micro::Register_SIN(),
                      ParseOpData);
  }

  TfLiteStatus AddSoftmax() {
    return AddBuiltin(BuiltinOperator_SOFTMAX,
                      *tflite::ops::micro::Register_SOFTMAX(), ParseSoftmax);
  }

  TfLiteStatus AddSplit() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_SPLIT,
                      *tflite::ops::micro::Register_SPLIT(), ParseOpData);
  }

  TfLiteStatus AddSqrt() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_SQRT,
                      *tflite::ops::micro::Register_SQRT(), ParseOpData);
  }

  TfLiteStatus AddSquare() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_SQUARE,
                      *tflite::ops::micro::Register_SQUARE(), ParseOpData);
  }

  TfLiteStatus AddStridedSlice() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_STRIDED_SLICE,
                      *tflite::ops::micro::Register_STRIDED_SLICE(),
                      ParseOpData);
  }

  TfLiteStatus AddSub() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_SUB, *tflite::ops::micro::Register_SUB(),
                      ParseOpData);
  }

  TfLiteStatus AddSvdf() {
    return AddBuiltin(BuiltinOperator_SVDF,
                      *tflite::ops::micro::Register_SVDF(), ParseSvdf);
  }

  TfLiteStatus AddTanh() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_TANH,
                      *tflite::ops::micro::Register_TANH(), ParseOpData);
  }

  TfLiteStatus AddUnpack() {
    // TODO(b/149408647): Replace ParseOpData with the operator specific parse
    // function.
    return AddBuiltin(BuiltinOperator_UNPACK,
                      *tflite::ops::micro::Register_UNPACK(), ParseOpData);
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
