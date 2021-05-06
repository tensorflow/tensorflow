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
#ifndef TENSORFLOW_LITE_CORE_API_OP_RESOLVER_H_
#define TENSORFLOW_LITE_CORE_API_OP_RESOLVER_H_

#include <memory>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

/// Abstract interface that returns TfLiteRegistrations given op codes or custom
/// op names. This is the mechanism that ops being referenced in the flatbuffer
/// model are mapped to executable function pointers (TfLiteRegistrations).
class OpResolver {
 public:
  /// Finds the op registration for a builtin operator by enum code.
  virtual const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                           int version) const = 0;
  /// Finds the op registration of a custom operator by op name.
  virtual const TfLiteRegistration* FindOp(const char* op,
                                           int version) const = 0;

  // Returns optional delegates for resolving and handling ops in the flatbuffer
  // model. This may be used in addition to the standard TfLiteRegistration
  // lookup for graph resolution.
  using TfLiteDelegatePtrVector =
      std::vector<std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>>;
  virtual TfLiteDelegatePtrVector GetDelegates(int num_threads) const {
    return TfLiteDelegatePtrVector();
  }

  virtual ~OpResolver() {}
};

// Handles the logic for converting between an OperatorCode structure extracted
// from a flatbuffer and information about a registered operator
// implementation.
TfLiteStatus GetRegistrationFromOpCode(const OperatorCode* opcode,
                                       const OpResolver& op_resolver,
                                       ErrorReporter* error_reporter,
                                       const TfLiteRegistration** registration);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_API_OP_RESOLVER_H_
