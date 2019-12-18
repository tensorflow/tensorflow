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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_C_API_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_C_API_H_

#define TPUDRIVER_CAPI_EXPORT __attribute__((visibility("default")))

extern "C" {

TPUDRIVER_CAPI_EXPORT extern void TpuDriver_Initialize();

TPUDRIVER_CAPI_EXPORT extern void TpuDriver_Open(const char* worker);

TPUDRIVER_CAPI_EXPORT extern const char* TpuDriver_Version(void);
}

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_C_API_H_
