/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PATH_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PATH_H_

#include <cstdint>

#include "tensorflow/lite/experimental/ruy/size_util.h"

namespace ruy {

// A Path is a choice of implementation path, e.g. between reference code
// and optimized code, or between different optimized code paths using different
// instruction sets.
//
// It's important that any symbol that depends on such implementation
// details, is somehow templatized in such a Path, so that different Path values
// yield different symbols, so we never have the situation where a symbols has
// multiple inequivalent definitions based on which code paths are compiled.
// That would be a violation of the ODR (One Definition Rule) which is Undefined
// Behavior, and one of the most serious issues plaguing both Eigen and
// gemmlowp.
//
// This enum is actually a bit-field: aside from kNone, all other values are
// powers of two, thus are one bit each. We define bit-wise operators below
// for this enum. Some places in Ruy accept a Path bit-field where multiple
// Paths may be selected, while some other places require a single Path (i.e.
// just one of the enum values here). Typically, user-facing parts of Ruy
// accept arbitrary bit-fields, allowing the user to compile support for
// multiple paths and to inform Ruy of all the paths that are to be enabled
// at runtime; then, typically in dispatch.h, we internally pick one
// specific path and from there on, internal Ruy code deals with only one
// path.
enum class Path : std::uint8_t {
  // Higher values have higher precedence.
  kNone = 0,
  kReference = 0x1,    // reference code.
  kStandardCpp = 0x2,  // Standard C++ only. No SIMD or other arch features.
  kNeon = 0x4,
  kNeonDotprod = 0x8,
};

inline constexpr Path operator|(Path p, Path q) {
  return static_cast<Path>(static_cast<std::uint32_t>(p) |
                           static_cast<std::uint32_t>(q));
}

inline constexpr Path operator&(Path p, Path q) {
  return static_cast<Path>(static_cast<std::uint32_t>(p) &
                           static_cast<std::uint32_t>(q));
}

inline constexpr Path operator^(Path p, Path q) {
  return static_cast<Path>(static_cast<std::uint32_t>(p) ^
                           static_cast<std::uint32_t>(q));
}

inline Path GetMostSignificantPath(Path path_mask) {
  return static_cast<Path>(round_down_pot(static_cast<int>(path_mask)));
}

#ifdef __aarch64__
constexpr Path kAllPaths =
    Path::kReference | Path::kStandardCpp | Path::kNeon | Path::kNeonDotprod;
#else
constexpr Path kAllPaths = Path::kReference | Path::kStandardCpp;
#endif

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PATH_H_
