/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This is the enum definition file for TF Status attribute.

#ifndef TF_FRAMEWORK_STATUS
#define TF_FRAMEWORK_STATUS

include "mlir/IR/OpBase.td"

def TFFramework_ERROR_CODE_OK
    : I32EnumAttrCase<"OK", 0>;
def TFFramework_ERROR_CODE_CANCELLED
    : I32EnumAttrCase<"CANCELLED", 1>;
def TFFramework_ERROR_CODE_UNKNOWN
    : I32EnumAttrCase<"UNKNOWN", 2>;
def TFFramework_ERROR_CODE_INVALID_ARGUMENT
    : I32EnumAttrCase<"INVALID_ARGUMENT", 3>;
def TFFramework_ERROR_CODE_DEADLINE_EXCEEDED
    : I32EnumAttrCase<"DEADLINE_EXCEEDED", 4>;
def TFFramework_ERROR_CODE_NOT_FOUND
    : I32EnumAttrCase<"NOT_FOUND", 5>;
def TFFramework_ERROR_CODE_ALREADY_EXISTS
    : I32EnumAttrCase<"ALREADY_EXISTS", 6>;
def TFFramework_ERROR_CODE_PERMISSION_DENIED
    : I32EnumAttrCase<"PERMISSION_DENIED", 7>;
def TFFramework_ERROR_CODE_UNAUTHENTICATED
    : I32EnumAttrCase<"UNAUTHENTICATED", 16>;
def TFFramework_ERROR_CODE_RESOURCE_EXHAUSTED
    : I32EnumAttrCase<"RESOURCE_EXHAUSTED", 8>;
def TFFramework_ERROR_CODE_FAILED_PRECONDITION
    : I32EnumAttrCase<"FAILED_PRECONDITION", 9>;
def TFFramework_ERROR_CODE_ABORTED
    : I32EnumAttrCase<"ABORTED", 10>;
def TFFramework_ERROR_CODE_OUT_OF_RANGE
    : I32EnumAttrCase<"OUT_OF_RANGE", 11>;
def TFFramework_ERROR_CODE_UNIMPLEMENTED
    : I32EnumAttrCase<"UNIMPLEMENTED", 12>;
def TFFramework_ERROR_CODE_INTERNAL
    : I32EnumAttrCase<"INTERNAL", 13>;
def TFFramework_ERROR_CODE_UNAVAILABLE
    : I32EnumAttrCase<"UNAVAILABLE", 14>;
def TFFramework_ERROR_CODE_DATA_LOSS
    : I32EnumAttrCase<"DATA_LOSS", 15>;

def TFFramework_ErrorCodeAttr : I32EnumAttr<"ErrorCode", "error code", [
  TFFramework_ERROR_CODE_OK,
  TFFramework_ERROR_CODE_CANCELLED,
  TFFramework_ERROR_CODE_UNKNOWN,
  TFFramework_ERROR_CODE_INVALID_ARGUMENT,
  TFFramework_ERROR_CODE_DEADLINE_EXCEEDED,
  TFFramework_ERROR_CODE_NOT_FOUND,
  TFFramework_ERROR_CODE_ALREADY_EXISTS,
  TFFramework_ERROR_CODE_PERMISSION_DENIED,
  TFFramework_ERROR_CODE_UNAUTHENTICATED,
  TFFramework_ERROR_CODE_RESOURCE_EXHAUSTED,
  TFFramework_ERROR_CODE_FAILED_PRECONDITION,
  TFFramework_ERROR_CODE_ABORTED,
  TFFramework_ERROR_CODE_OUT_OF_RANGE,
  TFFramework_ERROR_CODE_UNIMPLEMENTED,
  TFFramework_ERROR_CODE_INTERNAL,
  TFFramework_ERROR_CODE_UNAVAILABLE,
  TFFramework_ERROR_CODE_DATA_LOSS
]> {
  let cppNamespace = "::mlir::kernel_gen::tf_framework";
}

#endif // TF_FRAMEWORK_STATUS
