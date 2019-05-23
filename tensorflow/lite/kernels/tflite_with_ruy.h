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
#ifndef TENSORFLOW_LITE_KERNELS_TFLITE_WITH_RUY_H_
#define TENSORFLOW_LITE_KERNELS_TFLITE_WITH_RUY_H_

#if (defined TFLITE_WITH_RUY_EXPLICIT_TRUE) && \
    (defined TFLITE_WITH_RUY_EXPLICIT_FALSE)
#error TFLITE_WITH_RUY_EXPLICIT_TRUE and TFLITE_WITH_RUY_EXPLICIT_FALSE should not be simultaneously defined.
#endif

#if defined TFLITE_WITH_RUY_EXPLICIT_TRUE
#define TFLITE_WITH_RUY
#elif defined TFLITE_WITH_RUY_EXPLICIT_FALSE
// Leave TFLITE_WITH_RUY undefined
#else
// For now leave TFLITE_WITH_RUY undefined, could change defaults here later.
#endif

#endif  // TENSORFLOW_LITE_KERNELS_TFLITE_WITH_RUY_H_
