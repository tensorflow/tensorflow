/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_CPU_WINDOWS_COMPATIBILITY_H_
#define XLA_SERVICE_CPU_WINDOWS_COMPATIBILITY_H_

#ifdef _MSC_VER

extern "C" {

// MSVC does not have sincos[f].
void sincos(double x, double *sinv, double *cosv);
void sincosf(float x, float *sinv, float *cosv);

}

#endif  // _MSC_VER

#endif  // XLA_SERVICE_CPU_WINDOWS_COMPATIBILITY_H_
