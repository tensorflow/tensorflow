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

#include "tensorflow/c/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

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

// Register the given kernel builder with the TensorFlow runtime. If
// registration fails, the given status will be populated.
//
// This call takes ownership of the `builder` pointer.
TF_CAPI_EXPORT extern void TF_RegisterKernelBuilder(const char* kernel_name,
                                                    TF_KernelBuilder* builder,
                                                    TF_Status* status);

// Deletes the given TF_KernelBuilder. This should be called only if the kernel
// builder is not registered with TensorFlow via TF_RegisterKernelBuilder.
TF_CAPI_EXPORT extern void TF_DeleteKernelBuilder(TF_KernelBuilder* builder);

// --------------------------------------------------------------------------
// OpKernelContext routines

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

// Interprets the named kernel construction attribute as a TF_DataType and
// places it into *val. *status is set to TF_OK.
//
// If the attribute could not be found or could not be interpreted as
// TF_DataType, *status is populated with an error.
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrType(
    TF_OpKernelConstruction* ctx, const char* attr_name, TF_DataType* val,
    TF_Status* status);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_KERNELS_H_
