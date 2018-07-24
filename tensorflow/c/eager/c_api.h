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
// WARNING: Unlike tensorflow/c/c_api.h, the API here is not guaranteed to be
// stable and can change without notice.

#include "tensorflow/c/c_api.h"

// Macro to control visibility of exported symbols in the shared library (.so,
// .dylib, .dll).
// This duplicates the TF_EXPORT macro definition in
// tensorflow/core/platform/macros.h in order to keep this .h file independent
// of any other includes.$a
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
  // Running operations with input tensors on the wrong device will fail.
  TFE_DEVICE_PLACEMENT_EXPLICIT = 0,
  // Copy the tensor to the right device but log a warning.
  TFE_DEVICE_PLACEMENT_WARN = 1,
  // Silently copy the tensor, which has a performance cost since the operation
  // will be blocked till the copy completes. This is the default placement
  // policy.
  TFE_DEVICE_PLACEMENT_SILENT = 2,
  // Placement policy which silently copies int32 tensors but not other dtypes.
  TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32 = 3,
} TFE_ContextDevicePlacementPolicy;

// Sets the default execution mode (sync/async). Note that this can be
// overridden per thread using TFE_ContextSetAsyncForThread.
TF_CAPI_EXPORT extern void TFE_ContextOptionsSetAsync(TFE_ContextOptions*,
                                                      unsigned char is_async);

TF_CAPI_EXPORT extern void TFE_ContextOptionsSetDevicePlacementPolicy(
    TFE_ContextOptions*, TFE_ContextDevicePlacementPolicy);

// A tensorflow.ServerDef specifies remote workers (in addition to the current
// workers name). Operations created on this context can then be executed on
// any of these remote workers by setting an appropriate device.
//
// If the following is set, all servers identified by the
// ServerDef must be up when the context is created.
TF_CAPI_EXPORT extern void TFE_ContextOptionsSetServerDef(
    TFE_ContextOptions* options, const void* proto, size_t proto_len,
    TF_Status* status);

// Destroy an options object.
TF_CAPI_EXPORT extern void TFE_DeleteContextOptions(TFE_ContextOptions*);

// "Context" under which operations/functions are executed. It encapsulates
// things like the available devices, resource manager etc.
//
// TODO(ashankar): Merge with TF_Session?
typedef struct TFE_Context TFE_Context;

TF_CAPI_EXPORT extern TFE_Context* TFE_NewContext(
    const TFE_ContextOptions* opts, TF_Status* status);
TF_CAPI_EXPORT extern void TFE_DeleteContext(TFE_Context* ctx,
                                             TF_Status* status);
TF_CAPI_EXPORT extern TF_DeviceList* TFE_ContextListDevices(TFE_Context* ctx,
                                                            TF_Status* status);

// Clears the internal caches in the TFE context. Useful when reseeding random
// ops.
TF_CAPI_EXPORT extern void TFE_ContextClearCaches(TFE_Context* ctx);

// Sets a thread-local device placement policy. After this call, other calls to
// TFE_Execute in the same thread will use the device policy specified here
// instead of the device policy used to construct the context. This has no
// effect on the device policy used by other program threads.
TF_CAPI_EXPORT extern void TFE_ContextSetThreadLocalDevicePlacementPolicy(
    TFE_Context*, TFE_ContextDevicePlacementPolicy);

// Returns the device placement policy to be used by this context in the current
// thread.
TF_CAPI_EXPORT extern TFE_ContextDevicePlacementPolicy
TFE_ContextGetDevicePlacementPolicy(TFE_Context*);

// Overrides the execution mode (sync/async) for the current thread.
TF_CAPI_EXPORT extern void TFE_ContextSetAsyncForThread(TFE_Context*,
                                                        unsigned char is_async,
                                                        TF_Status* status);

// Causes the calling thread to block till all ops dispatched in async mode
// have been executed. Note that "execution" here refers to kernel execution /
// scheduling of copies, etc. Similar to sync execution, it doesn't guarantee
// that lower level device queues (like GPU streams) have been flushed.
//
// This call may not block for execution of ops enqueued concurrently with this
// call.
TF_CAPI_EXPORT extern void TFE_ContextAsyncWait(TFE_Context*,
                                                TF_Status* status);

// When an error happens, any pending operations are discarded and newly issued
// ops return an error. This call clears the error state and re-enables
// execution of newly issued ops.
//
// Note that outputs of discarded ops remain in a corrupt state and should not
// be used for future calls.
// TODO(agarwal): mark the affected handles and raise errors if they are used.
TF_CAPI_EXPORT extern void TFE_ContextAsyncClearError(TFE_Context*);

// A handle to a tensor on a device.
//
// Like a TF_Tensor, a TFE_TensorHandle refers to a tensor with a value, shape,
// type etc. Unlike a TF_Tensor, a TFE_TensorHandle may refer to such tensors
// placed in memory of different devices or remote address spaces.
typedef struct TFE_TensorHandle TFE_TensorHandle;

TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_NewTensorHandle(TF_Tensor* t,
                                                            TF_Status* status);
// Indicates that the caller will not be using `h` any more.
TF_CAPI_EXPORT extern void TFE_DeleteTensorHandle(TFE_TensorHandle* h);
TF_CAPI_EXPORT extern TF_DataType TFE_TensorHandleDataType(TFE_TensorHandle* h);
// This function will block till the operation that produces `h` has completed.
TF_CAPI_EXPORT extern int TFE_TensorHandleNumDims(TFE_TensorHandle* h,
                                                  TF_Status* status);
// This function will block till the operation that produces `h` has completed.
TF_CAPI_EXPORT extern int64_t TFE_TensorHandleDim(TFE_TensorHandle* h,
                                                  int dim_index,
                                                  TF_Status* status);
// This function will block till the operation that produces `h` has completed.
TF_CAPI_EXPORT extern const char* TFE_TensorHandleDeviceName(
    TFE_TensorHandle* h, TF_Status* status);

// This function will block till the operation that produces `h` has
// completed. The memory returned might alias the internal memory used by
// TensorFlow. Hence, callers should not mutate this memory (for example by
// modifying the memory region pointed to by TF_TensorData() on the returned
// TF_Tensor).
TF_CAPI_EXPORT extern TF_Tensor* TFE_TensorHandleResolve(TFE_TensorHandle* h,
                                                         TF_Status* status);

// Create a new TFE_TensorHandle with the same contents as 'h' but placed
// in the memory of the device name 'device_name'.
// If source and destination are the same device, then this creates a new handle
// that shares the underlying buffer. Otherwise, it currently requires at least
// one of the source or destination devices to be CPU (i.e., for the source or
// destination tensor to be placed in host memory).
// If asynchronous execution is enabled, the copy may be enqueued and the call will
// return "non-ready" handle. Else, this function returns after the copy has
// been done.
TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_TensorHandleCopyToDevice(
    TFE_TensorHandle* h, TFE_Context* ctx, const char* device_name,
    TF_Status* status);

// Debugging/Profiling information for TFE_TensorHandle
//
// TFE_TensorDebugInfo contains information useful for debugging and
// profiling tensors.
typedef struct TFE_TensorDebugInfo TFE_TensorDebugInfo;

// Retrieves TFE_TensorDebugInfo for `handle`.
// If TFE_TensorHandleTensorDebugInfo succeeds, `status` is set to OK and caller
// is responsible for deleting returned TFE_TensorDebugInfo.
// If TFE_TensorHandleTensorDebugInfo fails, `status` is set to appropriate
// error and nullptr is returned. This function can block till the operation
// that produces `handle` has completed.
TF_CAPI_EXPORT extern TFE_TensorDebugInfo* TFE_TensorHandleTensorDebugInfo(
    TFE_TensorHandle* handle, TF_Status* status);

// Deletes `debug_info`.
TF_CAPI_EXPORT extern void TFE_DeleteTensorDebugInfo(
    TFE_TensorDebugInfo* debug_info);

// Returns the number of dimensions used to represent the tensor on its device.
// The number of dimensions used to reprensent the tensor on device can be
// different from the number returned by TFE_TensorHandleNumDims.
// The return value was current at the time of TFE_TensorDebugInfo creation.
TF_CAPI_EXPORT extern int TFE_TensorDebugInfoOnDeviceNumDims(
    TFE_TensorDebugInfo* debug_info);

// Returns the number of elements in dimension `dim_index`.
// Tensor representation on device can be transposed from its representation
// on host. The data contained in dimension `dim_index` on device
// can correspond to the data contained in another dimension in on-host
// representation. The dimensions are indexed using the standard TensorFlow
// major-to-minor order (slowest varying dimension first),
// not the XLA's minor-to-major order.
// On-device dimensions can be padded. TFE_TensorDebugInfoOnDeviceDim returns
// the number of elements in a dimension after padding.
// The return value was current at the time of TFE_TensorDebugInfo creation.
TF_CAPI_EXPORT extern int64_t TFE_TensorDebugInfoOnDeviceDim(
    TFE_TensorDebugInfo* debug_info, int dim_index);

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

TF_CAPI_EXPORT extern TFE_Op* TFE_NewOp(TFE_Context* ctx,
                                        const char* op_or_function_name,
                                        TF_Status* status);

TF_CAPI_EXPORT extern void TFE_DeleteOp(TFE_Op* op);

TF_CAPI_EXPORT extern void TFE_OpSetDevice(TFE_Op* op, const char* device_name,
                                           TF_Status* status);
// The returned string remains valid throughout the lifetime of 'op'.
TF_CAPI_EXPORT extern const char* TFE_OpGetDevice(TFE_Op* op,
                                                  TF_Status* status);

// When 'enable' is set to 1, and if TensorFlow library is built with XLA
// support, a subsequent TFE_Execute() call on `op` will run the op via XLA.
//
// If the library is not built with XLA support, this call would be a no-op.
TF_CAPI_EXPORT extern void TFE_OpSetXLACompilation(TFE_Op* op,
                                                   unsigned char enable);

TF_CAPI_EXPORT extern void TFE_OpAddInput(TFE_Op* op, TFE_TensorHandle* h,
                                          TF_Status* status);

TF_CAPI_EXPORT extern TF_AttrType TFE_OpGetAttrType(TFE_Op* op,
                                                    const char* attr_name,
                                                    unsigned char* is_list,
                                                    TF_Status* status);
// Get an attribute type given an op name; a fusion of TFE_NewOp and
// TFE_OpGetAttrType for use from Python without the overhead of the individual
// calls and memory management of TFE_Op.
TF_CAPI_EXPORT extern TF_AttrType TFE_OpNameGetAttrType(
    TFE_Context* ctx, const char* op_or_function_name, const char* attr_name,
    unsigned char* is_list, TF_Status* status);

TF_CAPI_EXPORT extern void TFE_OpSetAttrString(TFE_Op* op,
                                               const char* attr_name,
                                               const void* value,
                                               size_t length);
TF_CAPI_EXPORT extern void TFE_OpSetAttrInt(TFE_Op* op, const char* attr_name,
                                            int64_t value);
TF_CAPI_EXPORT extern void TFE_OpSetAttrFloat(TFE_Op* op, const char* attr_name,
                                              float value);
TF_CAPI_EXPORT extern void TFE_OpSetAttrBool(TFE_Op* op, const char* attr_name,
                                             unsigned char value);
TF_CAPI_EXPORT extern void TFE_OpSetAttrType(TFE_Op* op, const char* attr_name,
                                             TF_DataType value);
// If the number of dimensions is unknown, `num_dims` must be set to
// -1 and `dims` can be null.  If a dimension is unknown, the
// corresponding entry in the `dims` array must be -1.
TF_CAPI_EXPORT extern void TFE_OpSetAttrShape(TFE_Op* op, const char* attr_name,
                                              const int64_t* dims,
                                              const int num_dims,
                                              TF_Status* out_status);

// Sets the attribute attr_name to be a function specified by 'function'.
//
// TODO(ashankar,iga): Add this functionality to the C API for graph
// construction. Perhaps we want an AttrValueMap equivalent in the C API?
TF_CAPI_EXPORT extern void TFE_OpSetAttrFunction(TFE_Op* op,
                                                 const char* attr_name,
                                                 const TFE_Op* value);

TF_CAPI_EXPORT extern void TFE_OpSetAttrStringList(TFE_Op* op,
                                                   const char* attr_name,
                                                   const void* const* values,
                                                   const size_t* lengths,
                                                   int num_values);
TF_CAPI_EXPORT extern void TFE_OpSetAttrIntList(TFE_Op* op,
                                                const char* attr_name,
                                                const int64_t* values,
                                                int num_values);
TF_CAPI_EXPORT extern void TFE_OpSetAttrFloatList(TFE_Op* op,
                                                  const char* attr_name,
                                                  const float* values,
                                                  int num_values);
TF_CAPI_EXPORT extern void TFE_OpSetAttrBoolList(TFE_Op* op,
                                                 const char* attr_name,
                                                 const unsigned char* values,
                                                 int num_values);
TF_CAPI_EXPORT extern void TFE_OpSetAttrTypeList(TFE_Op* op,
                                                 const char* attr_name,
                                                 const TF_DataType* values,
                                                 int num_values);
TF_CAPI_EXPORT extern void TFE_OpSetAttrShapeList(
    TFE_Op* op, const char* attr_name, const int64_t** dims,
    const int* num_dims, int num_values, TF_Status* out_status);
TF_CAPI_EXPORT extern void TFE_OpSetAttrFunctionList(TFE_Op* op,
                                                     const char* attr_name,
                                                     const TFE_Op** value,
                                                     int num_values);

// Execute the operation defined by 'op' and return handles to computed
// tensors in `retvals`.
//
// 'retvals' must point to a pre-allocated array of TFE_TensorHandle* and
// '*num_retvals' should be set to the size of this array. It is an error if
// the size of 'retvals' is less than the number of outputs. This call sets
// *num_retvals to the number of outputs.
//
// If asynchronous execution is enabled, the call may simply enqueue the execution
// and return "non-ready" handles in `retvals`. Note that any handles contained
// in 'op' should not be mutated till the kernel execution actually finishes.
//
// For sync execution, if any of the inputs to `op` are not ready, this call
// will block till they become ready and then return when the kernel execution
// is done.
// TODO(agarwal): change num_retvals to int from int*.
TF_CAPI_EXPORT extern void TFE_Execute(TFE_Op* op, TFE_TensorHandle** retvals,
                                       int* num_retvals, TF_Status* status);

// Add a function (serialized FunctionDef protocol buffer) to ctx so
// that it can be invoked using TFE_Execute.
TF_CAPI_EXPORT extern void TFE_ContextAddFunctionDef(
    TFE_Context* ctx, const char* serialized_function_def, size_t size,
    TF_Status* status);

// Adds a function (created from TF_GraphToFunction or
// TF_FunctionImportFunctionDef) to the context, allowing it to be executed with
// TFE_Execute by creating an op with the same name as the function.
TF_CAPI_EXPORT extern void TFE_ContextAddFunction(TFE_Context* ctx,
                                                  TF_Function* function,
                                                  TF_Status* status);

// Enables tracing of RunMetadata on the ops executed from this context.
TF_CAPI_EXPORT extern void TFE_ContextEnableRunMetadata(TFE_Context* ctx);

// Disables tracing of RunMetadata on the ops executed from this context.
TF_CAPI_EXPORT extern void TFE_ContextDisableRunMetadata(TFE_Context* ctx);

// Populates the passed-in buffer with a serialized RunMetadata protocol buffer
// containing any run metadata information accumulated so far and clears this
// information.
// If asynchronous mode is enabled, this call blocks till all currently pending ops are
// done.
TF_CAPI_EXPORT extern void TFE_ContextExportRunMetadata(TFE_Context* ctx,
                                                        TF_Buffer* buf,
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
