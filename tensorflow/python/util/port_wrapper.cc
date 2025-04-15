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

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/core/util/port.h"

PYBIND11_MODULE(_pywrap_util_port, m) {
  m.def("IsGoogleCudaEnabled", tensorflow::IsGoogleCudaEnabled);
  m.def("IsBuiltWithROCm", tensorflow::IsBuiltWithROCm);
  m.def("IsBuiltWithXLA", tensorflow::IsBuiltWithXLA);
  m.def("IsBuiltWithNvcc", tensorflow::IsBuiltWithNvcc);
  m.def("IsAArch32Available", tensorflow::IsAArch32Available);
  m.def("IsAArch64Available", tensorflow::IsAArch64Available);
  m.def("IsPowerPCAvailable", tensorflow::IsPowerPCAvailable);
  m.def("IsSystemZAvailable", tensorflow::IsSystemZAvailable);
  m.def("IsX86Available", tensorflow::IsX86Available);
  m.def("GpuSupportsHalfMatMulAndConv",
        tensorflow::GpuSupportsHalfMatMulAndConv);
  m.def("IsMklEnabled", tensorflow::IsMklEnabled);
}
