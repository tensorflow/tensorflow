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

#ifndef XLA_TSL_PLATFORM_TYPES_H_
#define XLA_TSL_PLATFORM_TYPES_H_

#include <cstdint>
#include <limits>
#include <string>

#include "absl/base/const_init.h"
#include "absl/base/macros.h"
#include "tsl/platform/bfloat16.h"  // IWYU pragma: export
#include "tsl/platform/ml_dtypes.h"  // IWYU pragma: export
#include "tsl/platform/tstring.h"

namespace tsl {

// Alias tsl::string to std::string.
using string ABSL_DEPRECATE_AND_INLINE() = std::string;
using uint8 ABSL_DEPRECATE_AND_INLINE() = uint8_t;
using uint16 ABSL_DEPRECATE_AND_INLINE() = uint16_t;
using uint32 ABSL_DEPRECATE_AND_INLINE() = uint32_t;
using uint64 ABSL_DEPRECATE_AND_INLINE() = uint64_t;
using int8 ABSL_DEPRECATE_AND_INLINE() = int8_t;
using int16 ABSL_DEPRECATE_AND_INLINE() = int16_t;
using int32 ABSL_DEPRECATE_AND_INLINE() = int32_t;
using int64 ABSL_DEPRECATE_AND_INLINE() = int64_t;

// Note: This duplication is necessary because the inliner doesn't handle
// macros very well and templates will cause it to replace int32_t with int.
namespace detail {
class Uint8Max {
 public:
  constexpr explicit Uint8Max(absl::ConstInitType) {}
  // Not copyable or movable.
  Uint8Max(const Uint8Max&) = delete;
  Uint8Max& operator=(const Uint8Max&) = delete;

  ABSL_DEPRECATE_AND_INLINE()
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator uint8_t() const {
    return std::numeric_limits<uint8_t>::max();
  }
};

class Uint16Max {
 public:
  constexpr explicit Uint16Max(absl::ConstInitType) {}
  // Not copyable or movable.
  Uint16Max(const Uint16Max&) = delete;
  Uint16Max& operator=(const Uint16Max&) = delete;

  ABSL_DEPRECATE_AND_INLINE()
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator uint16_t() const {
    return std::numeric_limits<uint16_t>::max();
  }
};

class Uint32Max {
 public:
  constexpr explicit Uint32Max(absl::ConstInitType) {}
  // Not copyable or movable.
  Uint32Max(const Uint32Max&) = delete;
  Uint32Max& operator=(const Uint32Max&) = delete;

  ABSL_DEPRECATE_AND_INLINE()
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator uint32_t() const {
    return std::numeric_limits<uint32_t>::max();
  }
};

class Uint64Max {
 public:
  constexpr explicit Uint64Max(absl::ConstInitType) {}
  // Not copyable or movable.
  Uint64Max(const Uint64Max&) = delete;
  Uint64Max& operator=(const Uint64Max&) = delete;

  ABSL_DEPRECATE_AND_INLINE()
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator uint64_t() const {
    return std::numeric_limits<uint64_t>::max();
  }
};

class Int8Min {
 public:
  constexpr explicit Int8Min(absl::ConstInitType) {}
  // Not copyable or movable.
  Int8Min(const Int8Min&) = delete;
  Int8Min& operator=(const Int8Min&) = delete;

  ABSL_DEPRECATE_AND_INLINE()
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator int8_t() const {
    return std::numeric_limits<int8_t>::min();
  }
};

class Int16Min {
 public:
  constexpr explicit Int16Min(absl::ConstInitType) {}
  // Not copyable or movable.
  Int16Min(const Int16Min&) = delete;
  Int16Min& operator=(const Int16Min&) = delete;

  ABSL_DEPRECATE_AND_INLINE()
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator int16_t() const {
    return std::numeric_limits<int16_t>::min();
  }
};

class Int32Min {
 public:
  constexpr explicit Int32Min(absl::ConstInitType) {}
  // Not copyable or movable.
  Int32Min(const Int32Min&) = delete;
  Int32Min& operator=(const Int32Min&) = delete;

  ABSL_DEPRECATE_AND_INLINE()
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator int32_t() const {
    return std::numeric_limits<int32_t>::min();
  }
};

class Int64Min {
 public:
  constexpr explicit Int64Min(absl::ConstInitType) {}
  // Not copyable or movable.
  Int64Min(const Int64Min&) = delete;
  Int64Min& operator=(const Int64Min&) = delete;

  ABSL_DEPRECATE_AND_INLINE()
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator int64_t() const {
    return std::numeric_limits<int64_t>::min();
  }
};

class Int8Max {
 public:
  constexpr explicit Int8Max(absl::ConstInitType) {}
  // Not copyable or movable.
  Int8Max(const Int8Max&) = delete;
  Int8Max& operator=(const Int8Max&) = delete;

  ABSL_DEPRECATE_AND_INLINE()
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator int8_t() const {
    return std::numeric_limits<int8_t>::max();
  }
};

class Int16Max {
 public:
  constexpr explicit Int16Max(absl::ConstInitType) {}
  // Not copyable or movable.
  Int16Max(const Int16Max&) = delete;
  Int16Max& operator=(const Int16Max&) = delete;

  ABSL_DEPRECATE_AND_INLINE()
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator int16_t() const {
    return std::numeric_limits<int16_t>::max();
  }
};

class Int32Max {
 public:
  constexpr explicit Int32Max(absl::ConstInitType) {}
  // Not copyable or movable.
  Int32Max(const Int32Max&) = delete;
  Int32Max& operator=(const Int32Max&) = delete;

  ABSL_DEPRECATE_AND_INLINE()
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator int32_t() const {
    return std::numeric_limits<int32_t>::max();
  }
};

class Int64Max {
 public:
  constexpr explicit Int64Max(absl::ConstInitType) {}
  // Not copyable or movable.
  Int64Max(const Int64Max&) = delete;
  Int64Max& operator=(const Int64Max&) = delete;

  ABSL_DEPRECATE_AND_INLINE()
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator int64_t() const {
    return std::numeric_limits<int64_t>::max();
  }
};
}  // namespace detail

inline constexpr detail::Uint8Max kuint8max{absl::kConstInit};
inline constexpr detail::Uint16Max kuint16max{absl::kConstInit};
inline constexpr detail::Uint32Max kuint32max{absl::kConstInit};
inline constexpr detail::Uint64Max kuint64max{absl::kConstInit};

inline constexpr detail::Int8Min kint8min{absl::kConstInit};
inline constexpr detail::Int16Min kint16min{absl::kConstInit};
inline constexpr detail::Int32Min kint32min{absl::kConstInit};
inline constexpr detail::Int64Min kint64min{absl::kConstInit};

inline constexpr detail::Int8Max kint8max{absl::kConstInit};
inline constexpr detail::Int16Max kint16max{absl::kConstInit};
inline constexpr detail::Int32Max kint32max{absl::kConstInit};
inline constexpr detail::Int64Max kint64max{absl::kConstInit};

// A typedef for a uint64 used as a short fingerprint.
using Fprint = uint64_t;

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

#endif  // XLA_TSL_PLATFORM_TYPES_H_
