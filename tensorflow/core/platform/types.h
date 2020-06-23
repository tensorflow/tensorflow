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

#ifndef TENSORFLOW_CORE_PLATFORM_TYPES_H_
#define TENSORFLOW_CORE_PLATFORM_TYPES_H_

#include <string>

#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/tstring.h"

// Include appropriate platform-dependent implementations
#if defined(PLATFORM_GOOGLE) || defined(GOOGLE_INTEGRAL_TYPES)
#include "tensorflow/core/platform/google/integral_types.h"
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) ||    \
    defined(PLATFORM_GOOGLE_ANDROID) || defined(PLATFORM_POSIX_IOS) || \
    defined(PLATFORM_GOOGLE_IOS) || defined(PLATFORM_WINDOWS)
#include "tensorflow/core/platform/default/integral_types.h"
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

namespace tensorflow {

// Alias tensorflow::string to std::string.
using std::string;

static const uint8 kuint8max = static_cast<uint8>(0xFF);
static const uint16 kuint16max = static_cast<uint16>(0xFFFF);
static const uint32 kuint32max = static_cast<uint32>(0xFFFFFFFF);
static const uint64 kuint64max = static_cast<uint64>(0xFFFFFFFFFFFFFFFFull);
static const int8 kint8min = static_cast<int8>(~0x7F);
static const int8 kint8max = static_cast<int8>(0x7F);
static const int16 kint16min = static_cast<int16>(~0x7FFF);
static const int16 kint16max = static_cast<int16>(0x7FFF);
static const int32 kint32min = static_cast<int32>(~0x7FFFFFFF);
static const int32 kint32max = static_cast<int32>(0x7FFFFFFF);
static const int64 kint64min = static_cast<int64>(~0x7FFFFFFFFFFFFFFFll);
static const int64 kint64max = static_cast<int64>(0x7FFFFFFFFFFFFFFFll);

// A typedef for a uint64 used as a short fingerprint.
typedef uint64 Fprint;

}  // namespace tensorflow

// Alias namespace ::stream_executor as ::tensorflow::se.
namespace stream_executor {}
namespace tensorflow {
namespace se = ::stream_executor;
}  // namespace tensorflow

#if defined(PLATFORM_WINDOWS)
#include <cstddef>
typedef std::ptrdiff_t ssize_t;
#endif

#endif  // TENSORFLOW_CORE_PLATFORM_TYPES_H_
