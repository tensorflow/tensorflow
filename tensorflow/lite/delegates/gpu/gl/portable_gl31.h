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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_PORTABLE_GL31_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_PORTABLE_GL31_H_

#define HAS_EGL 1

#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#ifdef __ANDROID__
// Weak-link all GL APIs included from this point on.
// TODO(camillol): Annotate these with availability attributes for the
// appropriate versions of Android, by including gl{3,31,31}.h and resetting
// GL_APICALL for each.
#undef GL_APICALL
#define GL_APICALL __attribute__((weak_import)) KHRONOS_APICALL
#endif  // __ANDROID__

#include <GLES3/gl31.h>

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_PORTABLE_GL31_H_
