/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_TYPES_H_
#define TENSORFLOW_TSL_PLATFORM_TYPES_H_

#include <string>

#include "tensorflow/tsl/platform/bfloat16.h"
#include "tensorflow/tsl/platform/platform.h"
#include "tensorflow/tsl/platform/tstring.h"

// Include appropriate platform-dependent implementations
#if defined(PLATFORM_GOOGLE) || defined(GOOGLE_INTEGRAL_TYPES)
#include "tensorflow/tsl/platform/google/integral_types.h"  // IWYU pragma: export
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) ||    \
    defined(PLATFORM_GOOGLE_ANDROID) || defined(PLATFORM_POSIX_IOS) || \
    defined(PLATFORM_GOOGLE_IOS) || defined(PLATFORM_WINDOWS)
#include "tensorflow/tsl/platform/default/integral_types.h"  // IWYU pragma: export
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

namespace tsl {

// Alias tensorflow::string to std::string.
using std::string;

static const tensorflow::uint8 kuint8max = static_cast<tensorflow::uint8>(0xFF);
static const tensorflow::uint16 kuint16max =
    static_cast<tensorflow::uint16>(0xFFFF);
static const tensorflow::uint32 kuint32max =
    static_cast<tensorflow::uint32>(0xFFFFFFFF);
static const tensorflow::uint64 kuint64max =
    static_cast<tensorflow::uint64>(0xFFFFFFFFFFFFFFFFull);
static const int8_t kint8min = static_cast<tensorflow::int8>(~0x7F);
static const int8_t kint8max = static_cast<tensorflow::int8>(0x7F);
static const int16_t kint16min = static_cast<tensorflow::int16>(~0x7FFF);
static const int16_t kint16max = static_cast<tensorflow::int16>(0x7FFF);
static const int32_t kint32min = static_cast<tensorflow::int32>(~0x7FFFFFFF);
static const int32_t kint32max = static_cast<tensorflow::int32>(0x7FFFFFFF);
static const int64_t kint64min = static_cast<int64_t>(~0x7FFFFFFFFFFFFFFFll);
static const int64_t kint64max = static_cast<int64_t>(0x7FFFFFFFFFFFFFFFll);

// A typedef for a uint64 used as a short fingerprint.
using Fprint = tensorflow::uint64;

}  // namespace tsl

// Alias namespace ::stream_executor as ::tensorflow::se.
namespace stream_executor {}
namespace tensorflow {
namespace se = ::stream_executor;
}  // namespace tensorflow

#if defined(PLATFORM_WINDOWS)
#include <cstddef>
typedef std::ptrdiff_t ssize_t;
#endif

#endif  // TENSORFLOW_TSL_PLATFORM_TYPES_H_
