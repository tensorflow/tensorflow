/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_SIGNATURE_RUNNER_H_
#define TENSORFLOW_LITE_SIGNATURE_RUNNER_H_
/// \file
///
/// An abstraction for invoking the TF Lite interpreter.
/// Provides support for named parameters, and for including multiple
/// named computations in a single model, each with its own inputs/outputs.

/// For documentation, see
/// third_party/tensorflow/lite/core/signature_runner.h.

#include "tensorflow/lite/core/signature_runner.h"  // IWYU pragma: export

namespace tflite {
using SignatureRunner = ::tflite::impl::SignatureRunner;
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SIGNATURE_RUNNER_H_
