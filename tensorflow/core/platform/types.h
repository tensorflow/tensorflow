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

#include <cstdint>
#include <string>

#include "xla/tsl/platform/types.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/tstring.h"
#include "tsl/platform/bfloat16.h"
#include "tsl/platform/tstring.h"
#include "tsl/platform/types.h"

namespace tensorflow {

// Alias tensorflow::string to std::string.
using tsl::string;

using tsl::uint16;
using tsl::uint2;
using tsl::uint32;
using tsl::uint4;
using tsl::uint64;
using tsl::uint8;

using tsl::int16;
using tsl::int2;
using tsl::int32;
using tsl::int4;
using tsl::int64;
using tsl::int8;

using tsl::float8_e4m3b11fnuz;
using tsl::float8_e4m3fn;
using tsl::float8_e4m3fnuz;
using tsl::float8_e5m2;
using tsl::float8_e5m2fnuz;

static const uint8 kuint8max = tsl::kuint8max;
static const uint16 kuint16max = tsl::kuint16max;
static const uint32 kuint32max = tsl::kuint32max;
static const uint64 kuint64max = tsl::kuint64max;
static const int8_t kint8min = tsl::kint8min;
static const int8_t kint8max = tsl::kint8max;
static const int16_t kint16min = tsl::kint16min;
static const int16_t kint16max = tsl::kint16max;
static const int32_t kint32min = tsl::kint32min;
static const int32_t kint32max = tsl::kint32max;
static const int64_t kint64min = tsl::kint64min;
static const int64_t kint64max = tsl::kint64max;

// A typedef for a uint64 used as a short fingerprint.
using tsl::bfloat16;
using tsl::Fprint;
using tsl::tstring;  // NOLINT: suppress 'using decl 'tstring' is unused'
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_TYPES_H_
