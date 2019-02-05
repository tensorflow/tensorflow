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

// Tensorflow on Android requires selective registration to be enabled in order
// for certain types (e.g. DT_UINT8) to work.
// Checks below ensure that for Android build, the right flags are passed to
// the compiler.

#if defined(__ANDROID__) && (!defined(__ANDROID_TYPES_FULL__) || \
                             !defined(SUPPORT_SELECTIVE_REGISTRATION))
#error \
    "Binary needs custom kernel support. For enabling custom kernels on " \
    "Android, please pass -D__ANDROID_TYPES_FULL__ && " \
    "-DSUPPORT_SELECTIVE_REGISTRATION for including the kernel in the binary."
#endif
