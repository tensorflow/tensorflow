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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PLATFORM_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PLATFORM_H_

#define RUY_PLATFORM(X) ((RUY_DONOTUSEDIRECTLY_##X) != 0)

// Detect ARM 32-bit
#ifdef __arm__
#define RUY_DONOTUSEDIRECTLY_ARM_32 1
#else
#define RUY_DONOTUSEDIRECTLY_ARM_32 0
#endif

// Detect ARM 64-bit
#ifdef __aarch64__
#define RUY_DONOTUSEDIRECTLY_ARM_64 1
#else
#define RUY_DONOTUSEDIRECTLY_ARM_64 0
#endif

// Detect NEON
#if (defined __ARM_NEON) || (defined __ARM_NEON__)
#define RUY_DONOTUSEDIRECTLY_NEON 1
#else
#define RUY_DONOTUSEDIRECTLY_NEON 0
#endif

// Define ARM 32-bit NEON
#define RUY_DONOTUSEDIRECTLY_NEON_32 \
  (RUY_DONOTUSEDIRECTLY_NEON && RUY_DONOTUSEDIRECTLY_ARM_32)

// Define ARM 64-bit NEON
// Note: NEON is implied by ARM64, so this define is redundant.
// It still allows some conveyance of intent.
#define RUY_DONOTUSEDIRECTLY_NEON_64 \
  (RUY_DONOTUSEDIRECTLY_NEON && RUY_DONOTUSEDIRECTLY_ARM_64)

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PLATFORM_H_
