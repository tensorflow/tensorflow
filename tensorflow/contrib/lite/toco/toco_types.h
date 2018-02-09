/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_TOCO_TYPES_H_
#define TENSORFLOW_CONTRIB_LITE_TOCO_TYPES_H_

#include <string>
#include "tensorflow/core/platform/platform.h"

#if defined(PLATFORM_GOOGLE) || defined(GOOGLE_INTEGRAL_TYPES)
#include "tensorflow/core/platform/google/integral_types.h"
#else
#include "tensorflow/core/platform/default/integral_types.h"
#endif

namespace toco {
#ifdef PLATFORM_GOOGLE
using ::string;
#else
using std::string;
#endif

using tensorflow::int16;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::int8;
using tensorflow::uint16;
using tensorflow::uint32;
using tensorflow::uint64;
using tensorflow::uint8;

}  // namespace toco

#endif  // TENSORFLOW_CONTRIB_LITE_TOCO_TYPES_H_
