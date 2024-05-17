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

#ifndef TENSORFLOW_TSL_PROFILER_UTILS_MATH_UTILS_H_
#define TENSORFLOW_TSL_PROFILER_UTILS_MATH_UTILS_H_

#include <cstdint>

namespace tsl {
namespace profiler {

// Converts among different SI units.
// https://en.wikipedia.org/wiki/International_System_of_Units
// NOTE: We use uint64 for picos and nanos, which are used in
// storage, and double for other units that are used in the UI.
inline double PicoToNano(uint64_t p) { return p / 1E3; }
inline double PicoToMicro(uint64_t p) { return p / 1E6; }
inline double PicoToMilli(uint64_t p) { return p / 1E9; }
inline double PicoToUni(uint64_t p) { return p / 1E12; }
inline uint64_t NanoToPico(uint64_t n) { return n * 1000; }
inline double NanoToMicro(uint64_t n) { return n / 1E3; }
inline double NanoToMilli(uint64_t n) { return n / 1E6; }
inline double MicroToNano(double u) { return u * 1E3; }
inline double MicroToMilli(double u) { return u / 1E3; }
inline uint64_t MilliToPico(double m) { return m * 1E9; }
inline uint64_t MilliToNano(double m) { return m * 1E6; }
inline double MilliToUni(double m) { return m / 1E3; }
inline uint64_t UniToPico(double uni) { return uni * 1E12; }
inline uint64_t UniToNano(double uni) { return uni * 1E9; }
inline double UniToMicro(double uni) { return uni * 1E6; }
inline double UniToGiga(double uni) { return uni / 1E9; }
inline double GigaToUni(double giga) { return giga * 1E9; }
inline double GigaToTera(double giga) { return giga / 1E3; }
inline double TeraToGiga(double tera) { return tera * 1E3; }

// Convert from clock cycles to seconds.
inline double CyclesToSeconds(double cycles, double frequency_hz) {
  // cycles / (cycles/s) = s.
  return cycles / frequency_hz;
}

// Checks the divisor and returns 0 to avoid divide by zero.
inline double SafeDivide(double dividend, double divisor) {
  constexpr double kEpsilon = 1.0E-10;
  if ((-kEpsilon < divisor) && (divisor < kEpsilon)) return 0.0;
  return dividend / divisor;
}

inline double GibiToGiga(double gibi) { return gibi * ((1 << 30) / 1.0e9); }
inline double GigaToGibi(double giga) { return giga / ((1 << 30) / 1.0e9); }

// Calculates GiB/s.
inline double GibibytesPerSecond(double gigabytes, double ns) {
  return GigaToGibi(SafeDivide(gigabytes, ns));
}

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_UTILS_MATH_UTILS_H_
