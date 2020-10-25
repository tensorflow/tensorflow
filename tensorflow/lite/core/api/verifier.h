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
/// \file
/// Abstract interface for verifying a model.
#ifndef TENSORFLOW_LITE_CORE_API_VERIFIER_H_
#define TENSORFLOW_LITE_CORE_API_VERIFIER_H_

#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {

/// Abstract interface that verifies whether a given model is legit.
/// It facilitates the use-case to verify and build a model without loading it
/// twice.
/// (See also "tensorflow/lite/tools/verifier.h".)
class TfLiteVerifier {
 public:
  /// Returns true if the model is legit.
  virtual bool Verify(const char* data, int length,
                      ErrorReporter* reporter) = 0;
  virtual ~TfLiteVerifier() {}
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_API_VERIFIER_H_
