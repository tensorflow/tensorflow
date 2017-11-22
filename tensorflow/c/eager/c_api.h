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

#ifndef TENSORFLOW_C_EAGER_C_API_H_
#define TENSORFLOW_C_EAGER_C_API_H_

// C API extensions to experiment with eager execution of kernels.

#include "tensorflow/c/c_api.h"

// Macro to control visibility of exported symbols in the shared library (.so,
// .dylib, .dll).
// This duplicates the TF_EXPORT macro definition in
// tensorflow/core/platform/macros.h in order to keep this .h file independent
// of any other includes.$a
#ifdef SWIG
#define TF_CAPI_EXPORT
#else
#if defined(COMPILER_MSVC)
#ifdef TF_COMPILE_LIBRARY
#define TF_CAPI_EXPORT __declspec(dllexport)
#else
#define TF_CAPI_EXPORT __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define TF_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // COMPILER_MSVC
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TFE_ContextOptions TFE_ContextOptions;

// Return a new options object.
TF_CAPI_EXPORT extern TFE_ContextOptions* TFE_NewContextOptions();

// Set the config in TF_ContextOptions.options.
// config should be a serialized tensorflow.ConfigProto proto.
// If config was not parsed successfully as a ConfigProto, record the
// error information in *status.
TF_CAPI_EXPORT extern void TFE_ContextOptionsSetConfig(
    TFE_ContextOptions* options, const void* proto, size_t proto_len,
    TF_Status* status);

// Controls how to act when we try to run an operation on a given device but
// some input tensors are not on that device.
typedef enum TFE_ContextDevicePlacementPolicy {
  // The default: running operations with input tensors on the wrong device will
  // fail.
  TFE_DEVICE_PLACEMENT_EXPLICIT = 0,
  // Copy the tensor to the right device but log a warning.
  TFE_DEVICE_PLACEMENT_WARN = 1,
  // Silently copy the tensor, which has a performance cost since the
  // operation will be blocked till the copy completes.
  TFE_DEVICE_PLACEMENT_SILENT = 2,
} TFE_ContextDevicePlacementPolicy;

TF_CAPI_EXPORT extern void TFE_ContextOptionsSetDevicePlacementPolicy(
    TFE_ContextOptions*, TFE_ContextDevicePlacementPolicy);

// Destroy an options object.
TF_CAPI_EXPORT extern void TFE_DeleteContextOptions(TFE_ContextOptions*);

// "Context" under which operations/functions are executed. It encapsulates
// things like the available devices, resource manager etc.
//
// TODO(ashankar): Merge with TF_Session?
typedef struct TFE_Context TFE_Context;

TF_CAPI_EXPORT extern TFE_Context* TFE_NewContext(
    const TFE_ContextOptions* opts, TF_Status* status);
TF_CAPI_EXPORT extern void TFE_DeleteContext(TFE_Context* ctx, TF_Status* status);
TF_CAPI_EXPORT extern TF_DeviceList* TFE_ContextListDevices(TFE_Context* ctx,
                                                            TF_Status* status);

// A handle to a tensor on a device.
//
// Like a TF_Tensor, a TFE_TensorHandle refers to a tensor with a value, shape,
// type etc. Unlike a TF_Tensor, a TFE_TensorHandle may refer to such tensors
// placed in memory of different devices or remote address spaces.
typedef struct TFE_TensorHandle TFE_TensorHandle;

TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_NewTensorHandle(TF_Tensor* t,
                                                            TF_Status* status);
TF_CAPI_EXPORT extern void TFE_DeleteTensorHandle(TFE_TensorHandle* h);
TF_CAPI_EXPORT extern TF_DataType TFE_TensorHandleDataType(TFE_TensorHandle* h);
TF_CAPI_EXPORT extern int TFE_TensorHandleNumDims(TFE_TensorHandle* h);
TF_CAPI_EXPORT extern int64_t TFE_TensorHandleDim(TFE_TensorHandle* h, int dim_index);
TF_CAPI_EXPORT extern const char* TFE_TensorHandleDeviceName(TFE_TensorHandle* h);
TF_CAPI_EXPORT extern TF_Tensor* TFE_TensorHandleResolve(TFE_TensorHandle* h,
                                                         TF_Status* status);

// Create a new TFE_TensorHandle with the same contents as 'h' but placed
// in the memory of the device name 'device_name'.
// If source and destination are the same device, then this creates a new handle
// that shares the underlying buffer. Otherwise, it currently requires at least
// one of the source or destination devices to be CPU (i.e., for the source or
// destination tensor to be placed in host memory).
TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_TensorHandleCopyToDevice(TFE_TensorHandle* h,
                                                                     TFE_Context* ctx,
                                                                     const char* device_name,
                                                                     TF_Status* status);

// Description of the TensorFlow op to execute.
//
// Assumes that the provided 'ctx' outlives the returned TFE_Op, i.e.,
// TFE_DeleteOp() is called before TFE_DeleteContext().
//
// Very similar to TF_OperationDescription with some differences:
// (1) TF_Output or TFE_TensorHandle* as arguments to TF_AddInput,
//     TF_AddInputList
// (2) TF_ColocateWith, TF_AddControlInput etc. do not make sense.
// (3) Implementation detail: Avoid use of NodeBuilder/NodeDefBuilder since
//     the additional sanity checks there seem unnecessary;
typedef struct TFE_Op TFE_Op;

TF_CAPI_EXPORT extern TFE_Op* TFE_NewOp(TFE_Context* ctx, const char* op_or_function_name,
                                        TF_Status* status);
TF_CAPI_EXPORT extern void TFE_DeleteOp(TFE_Op* op);

TF_CAPI_EXPORT extern void TFE_OpSetDevice(TFE_Op* op, const char* device_name,
                                           TF_Status* status);

TF_CAPI_EXPORT extern void TFE_OpAddInput(TFE_Op* op, TFE_TensorHandle* h, TF_Status* status);

TF_CAPI_EXPORT extern TF_AttrType TFE_OpGetAttrType(TFE_Op* op, const char* attr_name,
                                                    unsigned char* is_list, TF_Status* status);
// Get an attribute type given an op name; a fusion of TFE_NewOp and
// TFE_OpGetAttrType for use from Python without the overhead of the individual
// calls and memory management of TFE_Op.
TF_CAPI_EXPORT extern TF_AttrType TFE_OpNameGetAttrType(
    TFE_Context* ctx, const char* op_or_function_name, const char* attr_name,
    unsigned char* is_list, TF_Status* status);

TF_CAPI_EXPORT extern void TFE_OpSetAttrString(TFE_Op* op, const char* attr_name,
                                               const char* value);
TF_CAPI_EXPORT extern void TFE_OpSetAttrInt(TFE_Op* op, const char* attr_name, int64_t value);
TF_CAPI_EXPORT extern void TFE_OpSetAttrFloat(TFE_Op* op, const char* attr_name, float value);
TF_CAPI_EXPORT extern void TFE_OpSetAttrBool(TFE_Op* op, const char* attr_name,
                                             unsigned char value);
TF_CAPI_EXPORT extern void TFE_OpSetAttrType(TFE_Op* op, const char* attr_name,
                                             TF_DataType value);
// If the number of dimensions is unknown, `num_dims` must be set to
// -1 and `dims` can be null.  If a dimension is unknown, the
// corresponding entry in the `dims` array must be -1.
TF_CAPI_EXPORT extern void TFE_OpSetAttrShape(TFE_Op* op, const char* attr_name,
                                              const int64_t* dims, const int num_dims,
                                              TF_Status* out_status);

// Sets the attribute attr_name to be a function specified by 'function'.
//
// TODO(ashankar,iga): Add this functionality to the C API for graph
// construction. Perhaps we want an AttrValueMap equivalent in the C API?
TF_CAPI_EXPORT extern void TFE_OpSetAttrFunction(TFE_Op* op,
                                                 const char* attr_name,
                                                 const TFE_Op* value);

TF_CAPI_EXPORT extern void TFE_OpSetAttrStringList(TFE_Op* op, const char* attr_name,
                                                   const char** value, int num_values);
TF_CAPI_EXPORT extern void TFE_OpSetAttrIntList(TFE_Op* op, const char* attr_name,
                                                const int64_t* values, int num_values);
TF_CAPI_EXPORT extern void TFE_OpSetAttrFloatList(TFE_Op* op, const char* attr_name,
                                                  const float* values, int num_values);
TF_CAPI_EXPORT extern void TFE_OpSetAttrBoolList(TFE_Op* op, const char* attr_name,
                                                 const unsigned char* values, int num_values);
TF_CAPI_EXPORT extern void TFE_OpSetAttrTypeList(TFE_Op* op, const char* attr_name,
                                                 const TF_DataType* values, int num_values);
TF_CAPI_EXPORT extern void TFE_OpSetAttrShapeList(TFE_Op* op, const char* attr_name,
                                                  const int64_t** dims, const int* num_dims,
                                                  int num_values, TF_Status* out_status);

// Execute the operation defined by 'op' and return handles to computed
// tensors in 'retvals'.
//
// 'retvals' must point to a pre-allocated array of TFE_TensorHandle*
// and '*num_retvals' should be set to the size of this array.
//
// On return, 'num_retvals' will be set to the actual number of outputs
// returned by the operation.
TF_CAPI_EXPORT extern void TFE_Execute(TFE_Op* op, TFE_TensorHandle** retvals,
                                       int* num_retvals, TF_Status* status);

// Add a function (serialized FunctionDef protocol buffer) to ctx so
// that it can be invoked using TFE_Execute.
TF_CAPI_EXPORT extern void TFE_ContextAddFunctionDef(TFE_Context* ctx,
                                                     const char* serialized_function_def,
                                                     size_t size, TF_Status* status);

// Adds a function (created from TF_GraphToFunction or
// TF_FunctionImportFunctionDef) to the context, allowing it to be executed with
// TFE_Execute by creating an op with the same name as the function.
TF_CAPI_EXPORT extern void TFE_ContextAddFunction(TFE_Context* ctx,
                                                  TF_Function* function,
                                                  TF_Status* status);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#ifdef __cplusplus
// A workaround to ease conversion to and from numpy objects and
// TFE_TensorHandle's.
//
// TODO(ashankar): Figure out an alternative scheme that precludes the need for
// these API-boundary breaking methods.
namespace tensorflow {
class Tensor;
}  // namespace tensorflow

const tensorflow::Tensor* TFE_TensorHandleUnderlyingTensorInHostMemory(
    TFE_TensorHandle* h, TF_Status* status);
TFE_TensorHandle* TFE_NewTensorHandle(const tensorflow::Tensor& t);
#endif

#endif  // TENSORFLOW_C_EAGER_C_API_H_
