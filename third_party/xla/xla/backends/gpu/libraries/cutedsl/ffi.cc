/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/ffi/api/ffi.h"

#include "xla/ffi/api/c_api.h"

XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallPrepare);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallExecute);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "CuteDSLRT_NvJaxCutlassCall",
                         "CUDA",
                         (XLA_FFI_Handler_Bundle{
                             /*instantiate=*/nullptr,
                             /*prepare=*/CuteDSLRT_NvJaxCutlassCallPrepare,
                             /*initialize=*/nullptr,
                             /*execute=*/CuteDSLRT_NvJaxCutlassCallExecute}));
