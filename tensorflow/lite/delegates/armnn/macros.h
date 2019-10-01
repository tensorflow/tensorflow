/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_ARMNN_MACROS_H_
#define TENSORFLOW_LITE_DELEGATES_ARMNN_MACROS_H_

#define RETURN_EMPTY_ON_INVALID_OPTIONAL(optional) \
  do {                                             \
    if (!bool(optional)) {                         \
      armnn::EmptyOptional();                      \
    }                                              \
  } while (0)

// Returns kTfLiteError if condition is true
#define RETURN_TFLITE_ERROR_IF(cond) \
  do {                               \
    if ((cond)) {                    \
      return kTfLiteError;           \
    }                                \
  } while (0)

// Returns kTfLiteError if condition is true
#define RETURN_TFLITE_ERROR_ON_FALSE(cond) \
  do {                                     \
    if (!(cond)) {                         \
      return kTfLiteError;                 \
    }                                      \
  } while (0)

// Returns false if condition is true
#define RETURN_FALSE_IF(cond) \
  do {                        \
    if ((cond)) {             \
      return false;           \
    }                         \
  } while (0)

// Return if condition is false
#define RETURN_ON_FALSE(cond) \
  do {                        \
    if (!(cond)) {            \
      return cond;            \
    }                         \
  } while (0)

#endif  // TENSORFLOW_LITE_DELEGATES_ARMNN_MACROS_H_
