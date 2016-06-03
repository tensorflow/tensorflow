/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_FINGERPRINT_H_
#define TENSORFLOW_CORE_PLATFORM_FINGERPRINT_H_

#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// This is a portable fingerprint interface for strings that will never change.
// However, it is not suitable for cryptography.
uint64 Fingerprint64(const string& s);

}  // namespace tensorflow

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/fingerprint.h"
#else
#include "tensorflow/core/platform/default/fingerprint.h"
#endif

#endif  // TENSORFLOW_CORE_PLATFORM_FINGERPRINT_H_
