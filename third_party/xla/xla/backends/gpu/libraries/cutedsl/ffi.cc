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

// v1
XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallPrepare_v1);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallExecute_v1);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallExecuteNoCudaGraph_v1);

// v2
XLA_FFI_DECLARE_TYPE_ID_SYMBOL(CuteDSLRT_NvJaxCutlassCallStateTypeId_v2);
XLA_FFI_DECLARE_TYPE_INFO_SYMBOL(CuteDSLRT_NvJaxCutlassCallStateTypeInfo_v2);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallInstantiate_v2);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallPrepare_v2);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallExecute_v2);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallExecuteNoCudaGraph_v2);

// Type Registrations
XLA_FFI_REGISTER_TYPE(XLA_FFI_GetApi(), "CuteDSLRT_NvJaxCutlassCallTypes",
                      CuteDSLRT_NvJaxCutlassCallStateTypeId_v2(),
                      CuteDSLRT_NvJaxCutlassCallStateTypeInfo_v2());

XLA_FFI_REGISTER_TYPE(XLA_FFI_GetApi(), "CuteDSLRT_NvJaxCutlassCallTypes_v2",
                      CuteDSLRT_NvJaxCutlassCallStateTypeId_v2(),
                      CuteDSLRT_NvJaxCutlassCallStateTypeInfo_v2());

// v0 Registration
XLA_FFI_REGISTER_HANDLER(
    XLA_FFI_GetApi(), "CuteDSLRT_NvJaxCutlassCall", "CUDA",
    (XLA_FFI_Handler_Bundle{
        /*instantiate=*/CuteDSLRT_NvJaxCutlassCallInstantiate_v2,
        /*prepare=*/CuteDSLRT_NvJaxCutlassCallPrepare_v2,
        /*initialize=*/nullptr,
        /*execute=*/CuteDSLRT_NvJaxCutlassCallExecute_v2,
        /*record=*/nullptr}));

XLA_FFI_REGISTER_HANDLER(
    XLA_FFI_GetApi(), "CuteDSLRT_NvJaxCutlassCallNoCudaGraph", "CUDA",
    (XLA_FFI_Handler_Bundle{
        /*instantiate=*/CuteDSLRT_NvJaxCutlassCallInstantiate_v2,
        /*prepare=*/CuteDSLRT_NvJaxCutlassCallPrepare_v2,
        /*initialize=*/nullptr,
        /*execute=*/CuteDSLRT_NvJaxCutlassCallExecuteNoCudaGraph_v2,
        /*record=*/nullptr}));

// v1 Registration
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "CuteDSLRT_NvJaxCutlassCall_v1",
                         "CUDA",
                         (XLA_FFI_Handler_Bundle{
                             /*instantiate=*/nullptr,
                             /*prepare=*/CuteDSLRT_NvJaxCutlassCallPrepare_v1,
                             /*initialize=*/nullptr,
                             /*execute=*/CuteDSLRT_NvJaxCutlassCallExecute_v1,
                             /*record=*/nullptr}));

XLA_FFI_REGISTER_HANDLER(
    XLA_FFI_GetApi(), "CuteDSLRT_NvJaxCutlassCallNoCudaGraph_v1", "CUDA",
    (XLA_FFI_Handler_Bundle{
        /*instantiate=*/nullptr,
        /*prepare=*/CuteDSLRT_NvJaxCutlassCallPrepare_v1,
        /*initialize=*/nullptr,
        /*execute=*/CuteDSLRT_NvJaxCutlassCallExecuteNoCudaGraph_v1,
        /*record=*/nullptr}));

// v2 Registration
XLA_FFI_REGISTER_HANDLER(
    XLA_FFI_GetApi(), "CuteDSLRT_NvJaxCutlassCall_v2", "CUDA",
    (XLA_FFI_Handler_Bundle{
        /*instantiate=*/CuteDSLRT_NvJaxCutlassCallInstantiate_v2,
        /*prepare=*/CuteDSLRT_NvJaxCutlassCallPrepare_v2,
        /*initialize=*/nullptr,
        /*execute=*/CuteDSLRT_NvJaxCutlassCallExecute_v2,
        /*record=*/nullptr}));

XLA_FFI_REGISTER_HANDLER(
    XLA_FFI_GetApi(), "CuteDSLRT_NvJaxCutlassCallNoCudaGraph_v2", "CUDA",
    (XLA_FFI_Handler_Bundle{
        /*instantiate=*/CuteDSLRT_NvJaxCutlassCallInstantiate_v2,
        /*prepare=*/CuteDSLRT_NvJaxCutlassCallPrepare_v2,
        /*initialize=*/nullptr,
        /*execute=*/CuteDSLRT_NvJaxCutlassCallExecuteNoCudaGraph_v2,
        /*record=*/nullptr}));
