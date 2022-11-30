/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_PLATFORM_PLATFORM_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_PLATFORM_PLATFORM_H_

#if !defined(PLATFORM_POSIX) && !defined(PLATFORM_GOOGLE) &&                 \
    !defined(PLATFORM_POSIX_ANDROID) && !defined(PLATFORM_GOOGLE_ANDROID) && \
    !defined(PLATFORM_WINDOWS) && !defined(PLATFORM_CHROMIUMOS)

// Choose which platform we are on.
#if defined(ANDROID) || defined(__ANDROID__)
#define PLATFORM_POSIX_ANDROID

#elif defined(__APPLE__)
#define PLATFORM_POSIX

#elif defined(_WIN32)
#define PLATFORM_WINDOWS

#elif defined(__TF_CHROMIUMOS__)
#define PLATFORM_CHROMIUMOS

#else
// If no platform specified, use:
#define PLATFORM_POSIX

#endif
#endif

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_PLATFORM_PLATFORM_H_
