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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_PERSON_DETECTION_IMAGE_PROVIDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_PERSON_DETECTION_IMAGE_PROVIDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"

// This is an abstraction around an image source like a camera, and is
// expected to return 8-bit sample data.  The assumption is that this will be
// called in a low duty-cycle fashion in a low-power application.  In these
// cases, the imaging sensor need not be run in a streaming mode, but rather can
// be idled in a relatively low-power mode between calls to GetImage().  The
// assumption is that the overhead and time of bringing the low-power sensor out
// of this standby mode is commensurate with the expected duty cycle of the
// application.  The underlying sensor may actually be put into a streaming
// configuration, but the image buffer provided to GetImage should not be
// overwritten by the driver code until the next call to GetImage();
//
// The reference implementation can have no platform-specific dependencies, so
// it just returns a static image. For real applications, you should
// ensure there's a specialized implementation that accesses hardware APIs.
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, uint8_t* image_data);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_PERSON_DETECTION_IMAGE_PROVIDER_H_
