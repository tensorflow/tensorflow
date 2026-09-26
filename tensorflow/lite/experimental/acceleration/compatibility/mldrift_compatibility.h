/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_MLDRIFT_COMPATIBILITY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_MLDRIFT_COMPATIBILITY_H_

namespace tflite {
namespace acceleration {

// Returns true if the current device is supported for ML Drift GPU
// acceleration according to the embedded mldrift_compatibility.bin database.
//
// Contract details:
// - Thread safety: This function is thread-safe. Initialization occurs once in
//   a thread-safe manner (Meyer's singleton), and the result is cached for the
//   lifetime of the process.
// - Latency / Blocking: On Android, the first call synchronously spawns an
//   ephemeral helper thread to initialize an EGL context and probe GPU
//   capabilities, blocking the calling thread until completion. Callers on
//   performance-critical threads (such as UI threads) should be aware of this
//   one-time initialization latency.
// - EGL State Isolation: The helper thread isolates the EGL context; the
//   active EGL display, context, and surface of the calling thread are not
//   modified.
// - Failure semantics: Returns false if device properties (AndroidInfo) cannot
//   be read, if EGL environment initialization fails, or if the database is
//   corrupted. Warning diagnostics are logged via TFLITE_LOG_PROD.
// - Allowlist semantics: Any device/GPU combination not explicitly allowlisted
//   in the embedded database is considered unsupported and returns false (CPU
//   fallback).
// - Non-Android platforms: Returns true on non-Android platforms (e.g., Linux,
//   macOS) because the ML Drift GPU allowlist is designed specifically for
//   heterogeneous Android GPU drivers; on other platforms acceleration is
//   governed by standard delegate availability.
// - Static initializers warning: Must NOT be called from static initializers or
//   library constructors (__attribute__((constructor))). On Android, Bionic
//   holds the dynamic linker loader lock during library initialization; the
//   helper thread spawned to probe EGL may need to dlopen/initialize GPU driver
//   libraries, which requires the loader lock and can result in deadlock.
bool IsMlDriftGpuSupportedOnThisDevice();

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_MLDRIFT_COMPATIBILITY_H_
