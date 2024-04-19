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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_MATH_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_MATH_UTILS_H_

#include <cstdint>

#include "absl/base/macros.h"
#include "tsl/profiler/utils/math_utils.h"

// TODO: b/323943471 - This macro should eventually be provided by Abseil.
#ifndef ABSL_DEPRECATE_AND_INLINE
#define ABSL_DEPRECATE_AND_INLINE()
#endif

namespace tensorflow {
namespace profiler {

ABSL_DEPRECATE_AND_INLINE()
inline double CyclesToSeconds(double cycles, double frequency_hz) {
  return tsl::profiler::CyclesToSeconds(cycles, frequency_hz);
}

ABSL_DEPRECATE_AND_INLINE()
inline double GibibytesPerSecond(double gigabytes, double ns) {
  return tsl::profiler::GibibytesPerSecond(gigabytes, ns);
}

ABSL_DEPRECATE_AND_INLINE()
inline double GibiToGiga(double gibi) {
  return tsl::profiler::GibiToGiga(gibi);
}

ABSL_DEPRECATE_AND_INLINE()
inline double GigaToGibi(double giga) {
  return tsl::profiler::GigaToGibi(giga);
}

ABSL_DEPRECATE_AND_INLINE()
inline double GigaToTera(double giga) {
  return tsl::profiler::GigaToTera(giga);
}

ABSL_DEPRECATE_AND_INLINE()
inline double GigaToUni(double giga) { return tsl::profiler::GigaToUni(giga); }

ABSL_DEPRECATE_AND_INLINE()
inline double MicroToMilli(double u) { return tsl::profiler::MicroToMilli(u); }

ABSL_DEPRECATE_AND_INLINE()
inline double MicroToNano(double u) { return tsl::profiler::MicroToNano(u); }

ABSL_DEPRECATE_AND_INLINE()
inline uint64_t MilliToNano(double m) { return tsl::profiler::MilliToNano(m); }

ABSL_DEPRECATE_AND_INLINE()
inline uint64_t MilliToPico(double m) { return tsl::profiler::MilliToPico(m); }

ABSL_DEPRECATE_AND_INLINE()
inline double MilliToUni(double m) { return tsl::profiler::MilliToUni(m); }

ABSL_DEPRECATE_AND_INLINE()
inline double NanoToMicro(uint64_t n) { return tsl::profiler::NanoToMicro(n); }

ABSL_DEPRECATE_AND_INLINE()
inline double NanoToMilli(uint64_t n) { return tsl::profiler::NanoToMilli(n); }

ABSL_DEPRECATE_AND_INLINE()
inline uint64_t NanoToPico(uint64_t n) { return tsl::profiler::NanoToPico(n); }

ABSL_DEPRECATE_AND_INLINE()
inline double PicoToMicro(uint64_t p) { return tsl::profiler::PicoToMicro(p); }

ABSL_DEPRECATE_AND_INLINE()
inline double PicoToMilli(uint64_t p) { return tsl::profiler::PicoToMilli(p); }

ABSL_DEPRECATE_AND_INLINE()
inline double PicoToNano(uint64_t p) { return tsl::profiler::PicoToNano(p); }

ABSL_DEPRECATE_AND_INLINE()
inline double PicoToUni(uint64_t p) { return tsl::profiler::PicoToUni(p); }

ABSL_DEPRECATE_AND_INLINE()
inline double SafeDivide(double dividend, double divisor) {
  return tsl::profiler::SafeDivide(dividend, divisor);
}
ABSL_DEPRECATE_AND_INLINE()
inline double TeraToGiga(double tera) {
  return tsl::profiler::TeraToGiga(tera);
}

ABSL_DEPRECATE_AND_INLINE()
inline double UniToGiga(double uni) { return tsl::profiler::UniToGiga(uni); }

ABSL_DEPRECATE_AND_INLINE()
inline double UniToMicro(double uni) { return tsl::profiler::UniToMicro(uni); }

ABSL_DEPRECATE_AND_INLINE()
inline uint64_t UniToNano(double uni) { return tsl::profiler::UniToNano(uni); }

ABSL_DEPRECATE_AND_INLINE()
inline uint64_t UniToPico(double uni) { return tsl::profiler::UniToPico(uni); }

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_MATH_UTILS_H_
