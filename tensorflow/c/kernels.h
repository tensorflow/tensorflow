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

#ifndef TENSORFLOW_C_KERNELS_H_
#define TENSORFLOW_C_KERNELS_H_

#include <stdint.h>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

// Macro to control visibility of exported symbols in the shared library (.so,
// .dylib, .dll).
// This duplicates the TF_EXPORT macro definition in
// tensorflow/core/platform/macros.h in order to keep this .h file independent
// of any other includes.
#ifdef SWIG
#define TF_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TF_COMPILE_LIBRARY
#define TF_CAPI_EXPORT __declspec(dllexport)
#else
#define TF_CAPI_EXPORT __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define TF_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TF_Tensor TF_Tensor;

// --------------------------------------------------------------------------
// C API for TensorFlow Kernels.
//
// This API allows developers to register custom kernel implementations for
// TensorFlow.
//
// See c_api.h header comments for a discussion about API conventions.
//
// Users wishing to extend TensorFlow with new kernels will call
// `TF_NewKernelBuilder`. The resulting kernel builder can be registered with
// `TF_RegisterKernelBuilder`, which will allow TF to construct user-provided
// kernels when necessary.

typedef struct TF_KernelBuilder TF_KernelBuilder;
typedef struct TF_OpKernelConstruction TF_OpKernelConstruction;
typedef struct TF_OpKernelContext TF_OpKernelContext;

// TF_InitKernel to do op/kernel registration.
// Plugin should implement TF_InitKernel to register kernels. This function
// should register all kernels in a plugin.
void TF_InitKernel();

// Allocates a new kernel builder and returns a pointer to it.
//
// If non-null, TensorFlow will call create_func when it needs to instantiate
// the kernel. The pointer returned by create_func will be passed to
// compute_func and delete_func, thereby functioning as a "this" pointer for
// referring to kernel instances.
//
// The TF_OpKernelConstruction pointer passed to create_func is owned by
// TensorFlow and will be deleted once create_func returns. It must not be used
// after this.
//
// When TensorFlow needs to perform a computation with this kernel, it will
// call compute_func. This function will receive the pointer returned by
// create_func (or null if no create_func was provided), along with the inputs
// to the computation.
//
// The TF_OpKernelContext pointer received by compute_func is owned by
// TensorFlow and will be deleted once compute_func returns. It must not be used
// after this.
//
// Finally, when TensorFlow no longer needs the kernel, it will call
// delete_func if one is provided. This function will receive the pointer
// returned in `create_func` or nullptr if no `create_func` was provided.
//
// The caller should pass the result of this function to
// TF_RegisterKernelBuilder, which will take ownership of the pointer. If, for
// some reason, the kernel builder will not be registered, the caller should
// delete it with TF_DeleteKernelBuilder.
TF_CAPI_EXPORT extern TF_KernelBuilder* TF_NewKernelBuilder(
    const char* op_name, const char* device_name,
    void* (*create_func)(TF_OpKernelConstruction*),
    void (*compute_func)(void*, TF_OpKernelContext*),
    void (*delete_func)(void*));

// Specifies that this kernel's attribute only supports the given type.
TF_CAPI_EXPORT extern void TF_KernelBuilder_TypeConstraint(
    TF_KernelBuilder* kernel_builder, const char* attr_name,
    const TF_DataType type, TF_Status* status);

// Specify that this kernel requires/provides an input/output arg
// in host memory (instead of the default, device memory).
TF_CAPI_EXPORT extern void TF_KernelBuilder_HostMemory(
    TF_KernelBuilder* kernel_builder, const char* arg_name);

// Specify a priority number for this kernel.
TF_CAPI_EXPORT extern void TF_KernelBuilder_Priority(
    TF_KernelBuilder* kernel_builder, int32_t priority_number);

// Register the given kernel builder with the TensorFlow runtime. If
// registration fails, the given status will be populated.
//
// This call takes ownership of the `builder` pointer.
TF_CAPI_EXPORT extern void TF_RegisterKernelBuilder(const char* kernel_name,
                                                    TF_KernelBuilder* builder,
                                                    TF_Status* status);

// Register the given kernel builder with the TensorFlow runtime. If
// registration fails, the given status will be populated.
//
// This method is the same as TF_RegisterKernelBuilder except it takes in a
// serialized KernelDef, and uses it for registration, instead of building a new
// one. Users can choose to not provide a serialized KernelDef and in that case
// it's identical to TF_RegisterKernelBuilder.
TF_CAPI_EXPORT extern void TF_RegisterKernelBuilderWithKernelDef(
    const char* serialized_kernel_def, const char* name,
    TF_KernelBuilder* builder, TF_Status* status);

// Deletes the given TF_KernelBuilder. This should be called only if the kernel
// builder is not registered with TensorFlow via TF_RegisterKernelBuilder.
TF_CAPI_EXPORT extern void TF_DeleteKernelBuilder(TF_KernelBuilder* builder);

// --------------------------------------------------------------------------
// OpKernelContext routines

// TF_GetStream returns the SP_Stream available in ctx.
// This function returns a stream only for devices registered using the
// StreamExecutor C API
// (tensorflow/c/experimental/stream_executor/stream_executor.h). It will return
// nullptr and set error status in all other cases.
// Experimental: this function doesn't have compatibility guarantees and subject
// to change at any time.
TF_CAPI_EXPORT extern SP_Stream TF_GetStream(TF_OpKernelContext* ctx,
                                             TF_Status* status);

// TF_NumInputs returns the number of inputs available in ctx.
TF_CAPI_EXPORT extern int TF_NumInputs(TF_OpKernelContext* ctx);

// TF_NumOutputs returns the number of outputs to be placed in *ctx by the
// kernel.
TF_CAPI_EXPORT extern int TF_NumOutputs(TF_OpKernelContext* ctx);

// Retrieves the ith input from ctx. If TF_GetCode(status) is TF_OK, *tensor is
// populated and its ownership is passed to the caller. In any other case,
// *tensor is not modified.
//
// If i < 0 or i >= TF_NumInputs(ctx), *status is set to TF_OUT_OF_RANGE.
TF_CAPI_EXPORT extern void TF_GetInput(TF_OpKernelContext* ctx, int i,
                                       TF_Tensor** tensor, TF_Status* status);

// Sets the ith output of ctx to tensor. If TF_GetCode(status) is anything but
// TF_OK, ctx is left unmodified.
//
// If i < 0 or i >= TF_NumOutputs(ctx), *status is set to TF_OUT_OF_RANGE.
TF_CAPI_EXPORT extern void TF_SetOutput(TF_OpKernelContext* ctx, int i,
                                        const TF_Tensor* tensor,
                                        TF_Status* status);

// Notifies the given OpKernelConstruction that kernel construction has failed.
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_Failure(
    TF_OpKernelConstruction* ctx, TF_Status* status);

// Notifies the given OpKernelContext that the kernel's compute function has
// failed.
TF_CAPI_EXPORT extern void TF_OpKernelContext_Failure(TF_OpKernelContext* ctx,
                                                      TF_Status* status);

// Returns the expected output data type of the ith output. If i < 0 or
// i >= TF_NumOutputs(ctx), the program aborts.
TF_CAPI_EXPORT extern TF_DataType TF_ExpectedOutputDataType(
    TF_OpKernelContext* ctx, int i);

// Returns the step ID of the given context.
TF_CAPI_EXPORT extern int64_t TF_StepId(TF_OpKernelContext* ctx);

// Get the list_size and total_size of the attribute `attr_name` of `oper`.
// list_size - the length of the list.
// total_size - total size of the list.
//   (1) If attr_type == TF_ATTR_STRING
//       then total_size is the cumulative byte size
//       of all the strings in the list.
//   (3) If attr_type == TF_ATTR_SHAPE
//       then total_size is the number of dimensions
//       of the shape valued attribute, or -1
//       if its rank is unknown.
//   (4) If attr_type == TF_ATTR_SHAPE
//       then total_size is the cumulative number
//       of dimensions of all shapes in the list.
//   (5) Otherwise, total_size is undefined.
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrSize(
    TF_OpKernelConstruction* ctx, const char* attr_name, int32_t* list_size,
    int32_t* total_size, TF_Status* status);

// Interprets the named kernel construction attribute as a TF_DataType and
// places it into *val. *status is set to TF_OK.
//
// If the attribute could not be found or could not be interpreted as
// TF_DataType, *status is populated with an error.
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrType(
    TF_OpKernelConstruction* ctx, const char* attr_name, TF_DataType* val,
    TF_Status* status);

// Interprets the named kernel construction attribute as int32_t and
// places it into *val. *status is set to TF_OK.
//
// If the attribute could not be found or could not be interpreted as
// int32, *status is populated with an error.
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrInt32(
    TF_OpKernelConstruction* ctx, const char* attr_name, int32_t* val,
    TF_Status* status);

// Interprets the named kernel construction attribute as int64_t and
// places it into *val. *status is set to TF_OK.
//
// If the attribute could not be found or could not be interpreted as
// int64, *status is populated with an error.
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrInt64(
    TF_OpKernelConstruction* ctx, const char* attr_name, int64_t* val,
    TF_Status* status);

// Interprets the named kernel construction attribute as float and
// places it into *val. *status is set to TF_OK.
//
// If the attribute could not be found or could not be interpreted as
// float, *status is populated with an error.
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrFloat(
    TF_OpKernelConstruction* ctx, const char* attr_name, float* val,
    TF_Status* status);

// Interprets the named kernel construction attribute as bool and
// places it into *val. *status is set to TF_OK.
//
// If the attribute could not be found or could not be interpreted as
// bool, *status is populated with an error.
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrBool(
    TF_OpKernelConstruction* ctx, const char* attr_name, TF_Bool* val,
    TF_Status* status);

// Interprets the named kernel construction attribute as string and
// places it into *val. `val` must
// point to an array of length at least `max_length` (ideally set to
// total_size from TF_OpKernelConstruction_GetAttrSize(ctx,
// attr_name, list_size, total_size)). *status is set to TF_OK.
//
// If the attribute could not be found or could not be interpreted as
// string, *status is populated with an error.
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrString(
    TF_OpKernelConstruction* ctx, const char* attr_name, char* val,
    size_t max_length, TF_Status* status);

// Interprets the named kernel construction attribute as tensor and places it
// into *val. Allocates a new TF_Tensor which the caller is expected to take
// ownership of (and can deallocate using TF_DeleteTensor). *status is set to
// TF_OK.
//
// If the attribute could not be found or could not be interpreted as
// tensor, *status is populated with an error.
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrTensor(
    TF_OpKernelConstruction* ctx, const char* attr_name, TF_Tensor** val,
    TF_Status* status);

// Interprets the named kernel construction attribute as a TF_DataType array and
// places it into *vals. *status is set to TF_OK.
// `vals` must point to an array of length at least `max_values` (ideally set
// to list_size from
// TF_OpKernelConstruction_GetAttrSize(ctx, attr_name, list_size,
// total_size)).
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrTypeList(
    TF_OpKernelConstruction* ctx, const char* attr_name, TF_DataType* vals,
    int max_vals, TF_Status* status);

// Interprets the named kernel construction attribute as int32_t array and
// places it into *vals. *status is set to TF_OK.
// `vals` must point to an array of length at least `max_values` (ideally set
// to list_size from
// TF_OpKernelConstruction_GetAttrSize(ctx, attr_name, list_size,
// total_size)).
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrInt32List(
    TF_OpKernelConstruction* ctx, const char* attr_name, int32_t* vals,
    int max_vals, TF_Status* status);

// Interprets the named kernel construction attribute as int64_t array and
// places it into *vals. *status is set to TF_OK.
// `vals` must point to an array of length at least `max_values` (ideally set
// to list_size from
// TF_OpKernelConstruction_GetAttrSize(ctx, attr_name, list_size,
// total_size)).
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrInt64List(
    TF_OpKernelConstruction* ctx, const char* attr_name, int64_t* vals,
    int max_vals, TF_Status* status);

// Interprets the named kernel construction attribute as float array and
// places it into *vals. *status is set to TF_OK.
// `vals` must point to an array of length at least `max_values` (ideally set
// to list_size from
// TF_OpKernelConstruction_GetAttrSize(ctx, attr_name, list_size,
// total_size)).
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrFloatList(
    TF_OpKernelConstruction* ctx, const char* attr_name, float* vals,
    int max_vals, TF_Status* status);

// Interprets the named kernel construction attribute as bool array and
// places it into *vals. *status is set to TF_OK.
// `vals` must point to an array of length at least `max_values` (ideally set
// to list_size from
// TF_OpKernelConstruction_GetAttrSize(ctx, attr_name, list_size,
// total_size)).
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrBoolList(
    TF_OpKernelConstruction* ctx, const char* attr_name, TF_Bool* vals,
    int max_vals, TF_Status* status);

// Interprets the named kernel construction attribute as string array and fills
// in `vals` and `lengths`, each of which must point to an array of length at
// least `max_values`. *status is set to TF_OK. The elements of values will
// point to addresses in `storage` which must be at least `storage_size` bytes
// in length. Ideally, max_values would be set to list_size and `storage` would
// be at least total_size, obtained from
// TF_OpKernelConstruction_GetAttrSize(ctx, attr_name, list_size,
// total_size).
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrStringList(
    TF_OpKernelConstruction* ctx, const char* attr_name, char** vals,
    size_t* lengths, int max_values, void* storage, size_t storage_size,
    TF_Status* status);

// Interprets the named kernel construction attribute as tensor array and places
// it into *vals. *status is set to TF_OK.
// `vals` must point to an array of length at least `max_values`
// (ideally set to list_size from TF_OpKernelConstruction_GetAttrSize(ctx,
// attr_name, list_size, total_size)).
//
// The caller takes ownership of all the non-null TF_Tensor* entries in `vals`
// (which can be deleted using TF_DeleteTensor(vals[i])).
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrTensorList(
    TF_OpKernelConstruction* ctx, const char* attr_name, TF_Tensor** vals,
    int max_values, TF_Status* status);

// Return true if the kernel construction has the attr_name
TF_CAPI_EXPORT extern bool TF_OpKernelConstruction_HasAttr(
    TF_OpKernelConstruction* ctx, const char* attr_name, TF_Status* status);

// Returns the unique operation name for this OpKernel.
TF_CAPI_EXPORT extern TF_StringView TF_OpKernelConstruction_GetName(
    TF_OpKernelConstruction* ctx);

// Allocates Tensor for output at given index. Caller takes ownership of
// returned TF_Tensor and should deallocate it using TF_DeleteTensor(tensor).
//
// This function should be used to allocate outputs inside kernel
// compute function.
TF_CAPI_EXPORT TF_Tensor* TF_AllocateOutput(TF_OpKernelContext* context,
                                            int index, TF_DataType dtype,
                                            const int64_t* dims, int num_dims,
                                            size_t len, TF_Status* status);

// Tries to forward one of the inputs given in input_indices to
// output[output_index]. If none of the given inputs can be forwarded, calls
// allocate_output() to allocate a new output buffer. The index of the
// forwarded input will be assign to output argument forwarded_input (if it's
// not nullptr). If no inputs are forwarded, forwarded_input will be assigned
// -1.
TF_CAPI_EXPORT TF_Tensor* TF_ForwardInputOrAllocateOutput(
    TF_OpKernelContext* context, const int* candidate_input_indices,
    int num_candidate_input_indices, int output_index,
    const int64_t* output_dims, int output_num_dims, int* forwarded_input,
    TF_Status* status);

// Allocates a temporary Tensor of the specified type and shape. The
// Tensor must not be used after kernel construction is
// complete.
//
// num_dims must equal the size of array dims
TF_CAPI_EXPORT extern TF_Tensor* TF_AllocateTemp(
    TF_OpKernelContext* context, TF_DataType dtype, const int64_t* dims,
    int num_dims, TF_AllocatorAttributes* alloc_attrs, TF_Status* status);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_KERNELS_H_
