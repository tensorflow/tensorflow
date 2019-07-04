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
//
// When a user selects a set of compiled paths, Ruy internally dispatches to the
// "best" one, which typically means the newest optimized instructions for a
// given base architecture (such as ARM). Higher values of this enum correspond
// to "better" code paths within a given base architecture for which Ruy has
// optimized code paths.
enum class Path : std::uint8_t {
  // This is a special null value, representing the absence of any path.
  kNone = 0,
  // Reference multiplication code.
  // The main purpose of this path is to have a very simple standalone Mul
  // implementation to check against.
  // This path bypasses almost all of Ruy's internal implementation details.
  //
  // This is intended for testing/development.
  kReference = 0x1,
  // Standard C++ implementation of Ruy's architecture-specific parts.
  // Unlike Path::kReference, this path exercises most of Ruy's internal logic.
  //
  // This is intended for testing/development.
  kStandardCpp = 0x2,
  // Optimized path using a widely available subset of ARM NEON instructions.
  kNeon = 0x4,
  // Optimized path making use of ARM NEON dot product instructions that are
  // available on newer ARM cores.
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

inline constexpr Path operator~(Path p) {
  return static_cast<Path>(~static_cast<std::uint32_t>(p));
}

inline Path GetMostSignificantPath(Path path_mask) {
  return static_cast<Path>(round_down_pot(static_cast<int>(path_mask)));
}

// ruy::kAllPaths represents all Path's that make sense to on a given
// base architecture.
#ifdef __aarch64__
#ifdef __linux__
constexpr Path kAllPaths =
    Path::kReference | Path::kStandardCpp | Path::kNeon | Path::kNeonDotprod;
#else
// We don't know how to do runtime dotprod detection outside of linux for now.
constexpr Path kAllPaths = Path::kReference | Path::kStandardCpp | Path::kNeon;
#endif
#else
constexpr Path kAllPaths = Path::kReference | Path::kStandardCpp;
#endif

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PATH_H_
