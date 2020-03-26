/* Copyright 2020 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_CPU_CACHE_SIZE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_CPU_CACHE_SIZE_H_

#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/platform.h"

namespace ruy {

// LocalDataCacheSize returns a sane default size for each CPU core's local
// data cache, i.e. the largest data cache that is local to that CPU core, not
// shared with other cores. That allows coarse tuning of code that aims for
// most of its memory accesses to hit such a typically fast data cache.
//
// SharedDataCacheSize returns a sane default size of the total data cache
// accessible to each CPU, including any shared cache.
//
// For example, if we design tune this code for a ARM Cortex-A55 with a local L1
// cache of 32k, a local L2 cache of 128k and a shared L3 cache of 1M,
// LocalDataCacheSize should return 128k and SharedDataCacheSize
// should return 1M.
//
// Ideally these values would be queried at runtime, and we should probably
// do that on x86, but that is hard to do on ARM.
#if RUY_PLATFORM(ARM_64)
inline int LocalDataCacheSize() { return 1 << 15; }
inline int SharedDataCacheSize() { return 1 << 19; }
#elif RUY_PLATFORM(ARM_32)
inline int LocalDataCacheSize() { return 1 << 14; }
inline int SharedDataCacheSize() { return 1 << 18; }
#elif RUY_PLATFORM(X86)
inline int LocalDataCacheSize() { return 1 << 17; }
inline int SharedDataCacheSize() { return 1 << 21; }
#else
inline int LocalDataCacheSize() { return 1 << 14; }
inline int SharedDataCacheSize() { return 1 << 18; }
#endif
// Variants taking a Path argument which acts
// as a hint telling whether we're targeting more or less recent/powerful CPUs.
inline int LocalDataCacheSize(Path path) {
#if RUY_PLATFORM(ARM_64)
  if (path == Path::kNeonDotprod) {
    // At the moment, the smallest CPU with dotprod is probably Cortex-A55 with
    // 128k L2 local cache.
    return 1 << 17;
  }
#else
  (void)path;
#endif
  return LocalDataCacheSize();
}
inline int SharedDataCacheSize(Path path) {
#if RUY_PLATFORM(ARM_64)
  if (path == Path::kNeonDotprod) {
    // At the moment, the smallest CPU with dotprod is probably Cortex-A55 with
    // 1M L3 shared cache.
    return 1 << 20;
  }
#else
  (void)path;
#endif
  return SharedDataCacheSize();
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_CPU_CACHE_SIZE_H_
