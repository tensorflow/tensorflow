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
#ifndef TENSORFLOW_C_EAGER_C_API_EXPERIMENTAL_H_
#define TENSORFLOW_C_EAGER_C_API_EXPERIMENTAL_H_

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resets `op_to_reset` with `op_or_function_name` and `raw_device_name`. This
// is for performance optimization by reusing an exiting unused op rather than
// creating a new op every time. If `raw_device_name` is `NULL` or empty, it
// does not set the device name. If it's not `NULL`, then it attempts to parse
// and set the device name. It's effectively `TFE_OpSetDevice`, but it is faster
// than separately calling it because if the existing op has the same
// `raw_device_name`, it skips parsing and just leave as it is.
TF_CAPI_EXPORT extern void TFE_OpReset(TFE_Op* op_to_reset,
                                       const char* op_or_function_name,
                                       const char* raw_device_name,
                                       TF_Status* status);

// Enables only graph collection in RunMetadata on the functions executed from
// this context.
TF_CAPI_EXPORT extern void TFE_ContextEnableGraphCollection(TFE_Context* ctx);

// Disables only graph collection in RunMetadata on the functions executed from
// this context.
TF_CAPI_EXPORT extern void TFE_ContextDisableGraphCollection(TFE_Context* ctx);

// TODO(fishx): Move these monitoring APIs into a separate file.
// -----------------------------------------------------------------------------
// Monitoring Counter APIs.
// These APIs de-templated monitoring Counter for swig.

typedef struct TFE_MonitoringCounterCell TFE_MonitoringCounterCell;

// Atomically increments the value of the cell. The value must be non-negative.
TF_CAPI_EXPORT extern void TFE_MonitoringCounterCellIncrementBy(
    TFE_MonitoringCounterCell* cell, int64_t value);

// Retrieves the current value of the cell.
TF_CAPI_EXPORT extern int64_t TFE_MonitoringCounterCellValue(
    TFE_MonitoringCounterCell* cell);

// APIs for Counter without label.
typedef struct TFE_MonitoringCounter0 TFE_MonitoringCounter0;
// Returns a new Counter metric object. The caller should manage lifetime of
// the object. Using duplicate metric name will crash the program with fatal
// error.
TF_CAPI_EXPORT extern TFE_MonitoringCounter0* TFE_MonitoringNewCounter0(
    const char* name, TF_Status* status, const char* description);
// Deletes the Counter object.
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteCounter0(
    TFE_MonitoringCounter0* counter);
// Retrieves the cell from the Counter object. The Counter object will manage
// lifetime of the cell.
TF_CAPI_EXPORT extern TFE_MonitoringCounterCell* TFE_MonitoringGetCellCounter0(
    TFE_MonitoringCounter0* counter);

// APIs for Counter with 1 label.
typedef struct TFE_MonitoringCounter1 TFE_MonitoringCounter1;
TF_CAPI_EXPORT extern TFE_MonitoringCounter1* TFE_MonitoringNewCounter1(
    const char* name, TF_Status* status, const char* description,
    const char* label1);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteCounter1(
    TFE_MonitoringCounter1* counter);
TF_CAPI_EXPORT extern TFE_MonitoringCounterCell* TFE_MonitoringGetCellCounter1(
    TFE_MonitoringCounter1* counter, const char* label1);

// APIs for Counter with 2 labels.
typedef struct TFE_MonitoringCounter2 TFE_MonitoringCounter2;
TF_CAPI_EXPORT extern TFE_MonitoringCounter2* TFE_MonitoringNewCounter2(
    const char* name, TF_Status* status, const char* description,
    const char* label1, const char* label2);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteCounter2(
    TFE_MonitoringCounter2* counter);
TF_CAPI_EXPORT extern TFE_MonitoringCounterCell* TFE_MonitoringGetCellCounter2(
    TFE_MonitoringCounter2* counter, const char* label1, const char* label2);

// -----------------------------------------------------------------------------
// Monitoring Gauge APIs.
// These APIs de-templated monitoring Gauge for swig.

typedef struct TFE_MonitoringIntGaugeCell TFE_MonitoringIntGaugeCell;

// Atomically set the value of the cell.
TF_CAPI_EXPORT extern void TFE_MonitoringIntGaugeCellSet(
    TFE_MonitoringIntGaugeCell* cell, int64_t value);

// Retrieves the current value of the cell.
TF_CAPI_EXPORT extern int64_t TFE_MonitoringIntGaugeCellValue(
    TFE_MonitoringIntGaugeCell* cell);

// APIs for Int Gauge without label.
typedef struct TFE_MonitoringIntGauge0 TFE_MonitoringIntGauge0;
TF_CAPI_EXPORT extern TFE_MonitoringIntGauge0* TFE_MonitoringNewIntGauge0(
    const char* name, TF_Status* out_status, const char* description);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteIntGauge0(
    TFE_MonitoringIntGauge0* gauge);
TF_CAPI_EXPORT extern TFE_MonitoringIntGaugeCell*
TFE_MonitoringGetCellIntGauge0(TFE_MonitoringIntGauge0* gauge);

// APIs for Int Gauge with 1 label.
typedef struct TFE_MonitoringIntGauge1 TFE_MonitoringIntGauge1;
TF_CAPI_EXPORT extern TFE_MonitoringIntGauge1* TFE_MonitoringNewIntGauge1(
    const char* name, TF_Status* out_status, const char* description,
    const char* label1);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteIntGauge1(
    TFE_MonitoringIntGauge1* gauge);
TF_CAPI_EXPORT extern TFE_MonitoringIntGaugeCell*
TFE_MonitoringGetCellIntGauge1(TFE_MonitoringIntGauge1* gauge,
                               const char* label1);

// APIs for Int Gauge with 2 label.
typedef struct TFE_MonitoringIntGauge2 TFE_MonitoringIntGauge2;
TF_CAPI_EXPORT extern TFE_MonitoringIntGauge2* TFE_MonitoringNewIntGauge2(
    const char* name, TF_Status* out_status, const char* description,
    const char* label1, const char* label2);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteIntGauge2(
    TFE_MonitoringIntGauge2* gauge);
TF_CAPI_EXPORT extern TFE_MonitoringIntGaugeCell*
TFE_MonitoringGetCellIntGauge2(TFE_MonitoringIntGauge2* gauge,
                               const char* label1, const char* label2);

typedef struct TFE_MonitoringStringGaugeCell TFE_MonitoringStringGaugeCell;
TF_CAPI_EXPORT extern void TFE_MonitoringStringGaugeCellSet(
    TFE_MonitoringStringGaugeCell* cell, const char* value);
// Retrieves the string value and saves it in the buffer.
TF_CAPI_EXPORT extern const void TFE_MonitoringStringGaugeCellValue(
    TFE_MonitoringStringGaugeCell* cell, TF_Buffer* buf);

// APIs for String Gauge without label.
typedef struct TFE_MonitoringStringGauge0 TFE_MonitoringStringGauge0;
TF_CAPI_EXPORT extern TFE_MonitoringStringGauge0* TFE_MonitoringNewStringGauge0(
    const char* name, TF_Status* out_status, const char* description);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteStringGauge0(
    TFE_MonitoringStringGauge0* gauge);
TF_CAPI_EXPORT extern TFE_MonitoringStringGaugeCell*
TFE_MonitoringGetCellStringGauge0(TFE_MonitoringStringGauge0* gauge);

// APIs for String Gauge with 1 label.
typedef struct TFE_MonitoringStringGauge1 TFE_MonitoringStringGauge1;
TF_CAPI_EXPORT extern TFE_MonitoringStringGauge1* TFE_MonitoringNewStringGauge1(
    const char* name, TF_Status* out_status, const char* description,
    const char* label1);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteStringGauge1(
    TFE_MonitoringStringGauge1* gauge);
TF_CAPI_EXPORT extern TFE_MonitoringStringGaugeCell*
TFE_MonitoringGetCellStringGauge1(TFE_MonitoringStringGauge1* gauge,
                                  const char* label1);

// APIs for String Gauge with 2 label.
typedef struct TFE_MonitoringStringGauge2 TFE_MonitoringStringGauge2;
TF_CAPI_EXPORT extern TFE_MonitoringStringGauge2* TFE_MonitoringNewStringGauge2(
    const char* name, TF_Status* out_status, const char* description,
    const char* label1, const char* label2);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteStringGauge2(
    TFE_MonitoringStringGauge2* gauge);
TF_CAPI_EXPORT extern TFE_MonitoringStringGaugeCell*
TFE_MonitoringGetCellStringGauge2(TFE_MonitoringStringGauge2* gauge,
                                  const char* label1, const char* label2);

// APIs for String Gauge with 3 labels.
typedef struct TFE_MonitoringStringGauge3 TFE_MonitoringStringGauge3;
TF_CAPI_EXPORT extern TFE_MonitoringStringGauge3* TFE_MonitoringNewStringGauge3(
    const char* name, TF_Status* out_status, const char* description,
    const char* label1, const char* label2, const char* label3);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteStringGauge3(
    TFE_MonitoringStringGauge3* gauge);
TF_CAPI_EXPORT extern TFE_MonitoringStringGaugeCell*
TFE_MonitoringGetCellStringGauge3(TFE_MonitoringStringGauge3* gauge,
                                  const char* label1, const char* label2,
                                  const char* label3);

// APIs for String Gauge with 4 labels.
typedef struct TFE_MonitoringStringGauge4 TFE_MonitoringStringGauge4;
TF_CAPI_EXPORT extern TFE_MonitoringStringGauge4* TFE_MonitoringNewStringGauge4(
    const char* name, TF_Status* out_status, const char* description,
    const char* label1, const char* label2, const char* label3,
    const char* label4);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteStringGauge4(
    TFE_MonitoringStringGauge4* gauge);
TF_CAPI_EXPORT extern TFE_MonitoringStringGaugeCell*
TFE_MonitoringGetCellStringGauge4(TFE_MonitoringStringGauge4* gauge,
                                  const char* label1, const char* label2,
                                  const char* label3, const char* label4);

typedef struct TFE_MonitoringBoolGaugeCell TFE_MonitoringBoolGaugeCell;
TF_CAPI_EXPORT extern void TFE_MonitoringBoolGaugeCellSet(
    TFE_MonitoringBoolGaugeCell* cell, bool value);
TF_CAPI_EXPORT extern bool TFE_MonitoringBoolGaugeCellValue(
    TFE_MonitoringBoolGaugeCell* cell);

// APIs for Bool Gauge without label.
typedef struct TFE_MonitoringBoolGauge0 TFE_MonitoringBoolGauge0;
TF_CAPI_EXPORT extern TFE_MonitoringBoolGauge0* TFE_MonitoringNewBoolGauge0(
    const char* name, TF_Status* out_status, const char* description);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteBoolGauge0(
    TFE_MonitoringBoolGauge0* gauge);
TF_CAPI_EXPORT extern TFE_MonitoringBoolGaugeCell*
TFE_MonitoringGetCellBoolGauge0(TFE_MonitoringBoolGauge0* gauge);

// APIs for Bool Gauge with 1 label.
typedef struct TFE_MonitoringBoolGauge1 TFE_MonitoringBoolGauge1;
TF_CAPI_EXPORT extern TFE_MonitoringBoolGauge1* TFE_MonitoringNewBoolGauge1(
    const char* name, TF_Status* out_status, const char* description,
    const char* label1);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteBoolGauge1(
    TFE_MonitoringBoolGauge1* gauge);
TF_CAPI_EXPORT extern TFE_MonitoringBoolGaugeCell*
TFE_MonitoringGetCellBoolGauge1(TFE_MonitoringBoolGauge1* gauge,
                                const char* label1);

// APIs for Bool Gauge with 2 label.
typedef struct TFE_MonitoringBoolGauge2 TFE_MonitoringBoolGauge2;
TF_CAPI_EXPORT extern TFE_MonitoringBoolGauge2* TFE_MonitoringNewBoolGauge2(
    const char* name, TF_Status* out_status, const char* description,
    const char* label1, const char* label2);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteBoolGauge2(
    TFE_MonitoringBoolGauge2* gauge);
TF_CAPI_EXPORT extern TFE_MonitoringBoolGaugeCell*
TFE_MonitoringGetCellBoolGauge2(TFE_MonitoringBoolGauge2* gauge,
                                const char* label1, const char* label2);

// -----------------------------------------------------------------------------
// Monitoring Sampler APIs.
// These APIs de-templated monitoring Sampler for swig.

typedef struct TFE_MonitoringSamplerCell TFE_MonitoringSamplerCell;

// Atomically add the value of the cell.
TF_CAPI_EXPORT extern void TFE_MonitoringSamplerCellAdd(
    TFE_MonitoringSamplerCell* cell, double value);

// Retrieves the current value of the cell. The return value is a HistogramProto
// saved in the buffer.
TF_CAPI_EXPORT extern void TFE_MonitoringSamplerCellValue(
    TFE_MonitoringSamplerCell* cell, TF_Buffer* buf);

// APIs for sampler buckets
typedef struct TFE_MonitoringBuckets TFE_MonitoringBuckets;
TF_CAPI_EXPORT extern TFE_MonitoringBuckets*
TFE_MonitoringNewExponentialBuckets(double scale, double growth_factor,
                                    int bucket_count);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteBuckets(
    TFE_MonitoringBuckets* buckets);

// APIs for Sampler without label.
typedef struct TFE_MonitoringSampler0 TFE_MonitoringSampler0;
TF_CAPI_EXPORT extern TFE_MonitoringSampler0* TFE_MonitoringNewSampler0(
    const char* name, TFE_MonitoringBuckets* buckets, TF_Status* out_status,
    const char* description);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteSampler0(
    TFE_MonitoringSampler0* sampler);
TF_CAPI_EXPORT extern TFE_MonitoringSamplerCell* TFE_MonitoringGetCellSampler0(
    TFE_MonitoringSampler0* sampler);

// APIs for Sampler with 1 label.
typedef struct TFE_MonitoringSampler1 TFE_MonitoringSampler1;
TF_CAPI_EXPORT extern TFE_MonitoringSampler1* TFE_MonitoringNewSampler1(
    const char* name, TFE_MonitoringBuckets* buckets, TF_Status* out_status,
    const char* description, const char* label1);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteSampler1(
    TFE_MonitoringSampler1* sampler);
TF_CAPI_EXPORT extern TFE_MonitoringSamplerCell* TFE_MonitoringGetCellSampler1(
    TFE_MonitoringSampler1* sampler, const char* label1);

// APIs for Sampler with 2 label.
typedef struct TFE_MonitoringSampler2 TFE_MonitoringSampler2;
TF_CAPI_EXPORT extern TFE_MonitoringSampler2* TFE_MonitoringNewSampler2(
    const char* name, TFE_MonitoringBuckets* buckets, TF_Status* out_status,
    const char* description, const char* label1, const char* label2);
TF_CAPI_EXPORT extern void TFE_MonitoringDeleteSampler2(
    TFE_MonitoringSampler2* sampler);
TF_CAPI_EXPORT extern TFE_MonitoringSamplerCell* TFE_MonitoringGetCellSampler2(
    TFE_MonitoringSampler2* sampler, const char* label1, const char* label2);

// Sets whether to use TFRT
TF_CAPI_EXPORT extern void TFE_ContextOptionsSetTfrt(TFE_ContextOptions*,
                                                     bool use_tfrt);

// Sets whether to use TFRT distributed runtime
TF_CAPI_EXPORT extern void TFE_ContextOptionsSetTfrtDistributedRuntime(
    TFE_ContextOptions* options, bool use_tfrt_distributed_runtime);

// Returns the context_id from the EagerContext which is used by the
// EagerService to maintain consistency between client and worker. The
// context_id is initialized with a dummy value and is later set when the worker
// is initialized (either locally or remotely). The context_id can change during
// the process lifetime although this should cause the worker to be
// reinitialized (e.g. cleared caches) as well.
TF_CAPI_EXPORT extern uint64_t TFE_GetContextId(TFE_Context* ctx);

// -----------------------------------------------------------------------------
// Cancellation APIs.

typedef struct TFE_CancellationManager TFE_CancellationManager;
TF_CAPI_EXPORT extern TFE_CancellationManager* TFE_NewCancellationManager();
TF_CAPI_EXPORT extern bool TFE_CancellationManagerIsCancelled(
    TFE_CancellationManager*);
TF_CAPI_EXPORT extern void TFE_CancellationManagerStartCancel(
    TFE_CancellationManager*);
TF_CAPI_EXPORT extern void TFE_DeleteCancellationManager(
    TFE_CancellationManager*);

// Associates the given `cancellation_manager` with `op`, so that invoking
// `TFE_CancellationManagerStartCancel(cancellation_manager)` will cancel the
// execution of `op`.
typedef struct TFE_CancellationManager TFE_CancellationManager;
TF_CAPI_EXPORT extern void TFE_OpSetCancellationManager(
    TFE_Op* op, TFE_CancellationManager* cancellation_manager,
    TF_Status* status);

// -----------------------------------------------------------------------------
// Eager Executor APIs.
typedef struct TFE_Executor TFE_Executor;

// Creates a new eager Executor. Nodes in one executor are guaranteed to be
// executed in sequence. Assigning nodes to different executors allows executing
// nodes in parallel.
TF_CAPI_EXPORT extern TFE_Executor* TFE_NewExecutor(
    bool is_async, bool enable_streaming_enqueue);

// Deletes the eager Executor without waiting for enqueued nodes. Please call
// TFE_ExecutorWaitForAllPendingNodes before calling this API if you want to
// make sure all nodes are finished.
TF_CAPI_EXPORT extern void TFE_DeleteExecutor(TFE_Executor*);

// Returns true if the executor is in async mode.
TF_CAPI_EXPORT extern bool TFE_ExecutorIsAsync(TFE_Executor*);

// Causes the calling thread to block till all ops dispatched in this executor
// have been executed. Note that "execution" here refers to kernel execution /
// scheduling of copies, etc. Similar to sync execution, it doesn't guarantee
// that lower level device queues (like GPU streams) have been flushed.
//
// This call may not block for execution of ops enqueued concurrently with this
// call.
TF_CAPI_EXPORT extern void TFE_ExecutorWaitForAllPendingNodes(
    TFE_Executor*, TF_Status* status);

// When an error happens, any pending operations are discarded, and newly issued
// ops return an error. This call clears the error state and re-enables
// execution of newly issued ops.
//
// Note that outputs of discarded ops remain in a corrupt state and should not
// be used for future calls.
// TODO(agarwal): mark the affected handles and raise errors if they are used.
TF_CAPI_EXPORT extern void TFE_ExecutorClearError(TFE_Executor*);

// Sets a custom Executor for the current thread. All nodes created by this
// thread will be added to this Executor. It will override the current executor.
TF_CAPI_EXPORT extern void TFE_ContextSetExecutorForThread(TFE_Context*,
                                                           TFE_Executor*);

// Returns the Executor for the current thread.
TF_CAPI_EXPORT extern TFE_Executor* TFE_ContextGetExecutorForThread(
    TFE_Context*);

// -----------------------------------------------------------------------------
// Dynamic cluster API.

// Update an existing context with a new set of servers defined in a ServerDef
// proto. Servers can be added to and removed from the list of remote workers
// in the context. A New set of servers identified by the ServerDef must be up
// when the context is updated.
//
// This API is for experimental usage and may be subject to change.
TF_CAPI_EXPORT extern void TFE_ContextUpdateServerDef(TFE_Context* ctx,
                                                      int keep_alive_secs,
                                                      const void* proto,
                                                      size_t proto_len,
                                                      TF_Status* status);

// Checks whether a remote worker is alive or not. This will return true even if
// the context doesn't exist on the remote worker.
TF_CAPI_EXPORT extern bool TFE_ContextCheckAlive(TFE_Context* ctx,
                                                 const char* worker_name,
                                                 TF_Status* status);

// Sync pending nodes in local executors (including the context default executor
// and thread executors) and streaming requests to remote executors, and get the
// combined status.
TF_CAPI_EXPORT extern void TFE_ContextAsyncWait(TFE_Context* ctx,
                                                TF_Status* status);

// This function will block till the operation that produces `h` has
// completed. This is only valid on local TFE_TensorHandles. The pointer
// returned will be on the device in which the TFE_TensorHandle resides (so e.g.
// for a GPU tensor this will return a pointer to GPU memory). The pointer is
// only guaranteed to be valid until TFE_DeleteTensorHandle is called on this
// TensorHandle. Only supports POD data types.
TF_CAPI_EXPORT extern void* TFE_TensorHandleDevicePointer(TFE_TensorHandle*,
                                                          TF_Status*);

// This function will block till the operation that produces `h` has
// completed. This is only valid on local TFE_TensorHandles. Returns the size in
// bytes of the memory pointed to by the device pointer returned above.
TF_CAPI_EXPORT extern size_t TFE_TensorHandleDeviceMemorySize(TFE_TensorHandle*,
                                                              TF_Status*);

// Creates a new TensorHandle from memory residing in the physical device
// device_name. Takes ownership of the memory, and will call deleter to release
// it after TF no longer needs it or in case of error.
//
// Custom devices must use TFE_NewCustomDeviceTensorHandle instead.
TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_NewTensorHandleFromDeviceMemory(
    TFE_Context* ctx, const char* device_name, TF_DataType, const int64_t* dims,
    int num_dims, void* data, size_t len,
    void (*deallocator)(void* data, size_t len, void* arg),
    void* deallocator_arg, TF_Status* status);

// Retrieves the address space (i.e. job, replia, task) of the local host and
// saves it in the buffer.
TF_CAPI_EXPORT extern void TFE_HostAddressSpace(TFE_Context* ctx,
                                                TF_Buffer* buf);

// APIs for generically dealing with op attributes (e.g. when forwarding them
// through custom device implementations).
//
// TODO(allenl): Currently these are black boxes, but we should have some way to
// inspect values. This would let people e.g. copy over most attributes and then
// modify some based on their values.

// A reference to an op's name -> attribute mapping
typedef struct TFE_OpAttrs TFE_OpAttrs;

// Fetch a reference to `op`'s attributes. The returned reference is only valid
// while `op` is alive.
TF_CAPI_EXPORT extern const TFE_OpAttrs* TFE_OpGetAttrs(const TFE_Op* op);
// Add attributes in `attrs` to `op`.
//
// Does not overwrite or update existing attributes, but adds new ones.
TF_CAPI_EXPORT extern void TFE_OpAddAttrs(TFE_Op* op, const TFE_OpAttrs* attrs);

// Serialize `attrs` as a tensorflow::NameAttrList protocol buffer (into `buf`),
// containing the op name and a map of its attributes.
TF_CAPI_EXPORT extern void TFE_OpAttrsSerialize(const TFE_OpAttrs* attrs,
                                                TF_Buffer* buf,
                                                TF_Status* status);

// Set an op's attribute from a serialized AttrValue protocol buffer.
//
// Analogous to TF_SetAttrValueProto for building graph operations.
TF_CAPI_EXPORT extern void TFE_OpSetAttrValueProto(const TFE_Op* op,
                                                   const char* attr_name,
                                                   const void* proto,
                                                   size_t proto_len,
                                                   TF_Status* status);

// TODO(b/166642410): It would be nice, for custom devices and for other users,
// to have a non-string representation of devices (TF_Device) extracted from
// tensors/ops/etc. and usable in APIs like OpSetDevice/ResetOp/etc.

#define TFE_CUSTOM_DEVICE_VERSION 4

// Struct to be filled in. Functions are required except where indicated.
typedef struct TFE_CustomDevice {
  int version = TFE_CUSTOM_DEVICE_VERSION;
  // Method to copy a tensor to the custom device.
  TFE_TensorHandle* (*copy_tensor_to_device)(TFE_Context* context,
                                             TFE_TensorHandle* tensor,
                                             TF_Status* status,
                                             void* device_info);

  // Method to copy a tensor from the custom device to a target device.
  TFE_TensorHandle* (*copy_tensor_from_device)(TFE_Context* context,
                                               TFE_TensorHandle* tensor,
                                               const char* target_device_name,
                                               TF_Status* status,
                                               void* device_info);

  // Method to execute an operation.
  //
  // Arguments provide enough information to reconstruct the original `TFE_Op`,
  // or construct a transformed version, by inspecting the passed `op`.
  //
  // TFE_OpGetDevice(op) records the original placement of the operation. It may
  // be an empty string if no device was explicitly requested, but will
  // otherwise be the name of this custom device. Ops are placed onto a custom
  // device if any of their inputs are on that custom device, but custom devices
  // are free to set a bad status in order to require explicit placement.
  void (*execute)(const TFE_Op* op, int* num_outputs,
                  TFE_TensorHandle** outputs, TF_Status* s, void* device_info);

  // Method to delete a device.
  void (*delete_device)(void* device_info);

  // Implements TFE_CreatePackedTensorHandle when one of `handles` is on this
  // custom device.
  //
  // Many devices will want to simply return an "unimplemented" status
  // here. This is the default behavior if `pack` is null when passed to
  // TFE_RegisterCustomDevice.
  TFE_TensorHandle* (*pack)(TFE_Context* context, TFE_TensorHandle** handles,
                            int num_handles, TF_Status* s,
                            void* device_info) = nullptr;

  // Pins the op to `device` based on inputs to `op`. Returns true
  // signifying to pin to the current custom device. Returns false
  // to pin to the physical device.
  //
  // This function is guaranteed to be called only when all of the custom-device
  // inputs are on this device.
  bool (*shall_pin_to_this_device)(const TFE_Op* op, TF_Status* s) = nullptr;
} TFE_CustomDevice;

// Registers a custom device for use with eager execution.
//
// Eager operations may be placed on this device, e.g.  `with
// tf.device("CUSTOM"):` from Python if `device_name` for this call is
// "/job:localhost/replica:0/task:0/device:CUSTOM:0".
//
// The custom device defines copy operations for moving TensorHandles on and
// off, and an execution operation for named operations. Often execution will
// simply wrap op execution on one or more physical devices.
//
// device_info is an opaque caller-defined type stored with the custom device
// which is passed to the functions referenced in the TFE_CustomDevice struct
// `device` (execute, delete_device, etc.). It can for example contain the
// names of wrapped devices.
//
// There are currently no graph semantics implemented for registered custom
// devices, so executing tf.functions which contain operations placed on the
// custom devices will fail.
//
// `device_name` must not name an existing physical or custom device. It must
// follow the format:
//
//    /job:<name>/replica:<replica>/task:<task>/device:<type>:<device_num>
//
// If the device is successfully registered, `status` is set to TF_OK. Otherwise
// the device is not usable. In case of a bad status, `device.delete_device` is
// still called on `device_info` (i.e. the caller does not retain ownership).
//
// This API is highly experimental, and in particular is expected to change when
// it starts supporting operations with attributes and when tf.function support
// is added.
TF_CAPI_EXPORT extern void TFE_RegisterCustomDevice(TFE_Context* ctx,
                                                    TFE_CustomDevice device,
                                                    const char* device_name,
                                                    void* device_info,
                                                    TF_Status* status);

// Returns whether `device_name` maps to a registered custom device.
TF_CAPI_EXPORT extern bool TFE_IsCustomDevice(TFE_Context* ctx,
                                              const char* device_name);

// Struct to be filled in to define a custom device tensor handle. Fields are
// required except where indicated.
typedef struct TFE_CustomDeviceTensorHandleMethods {
  int version = TFE_CUSTOM_DEVICE_VERSION;

  // Computes the rank of the tensor handle.
  //
  // Shapes are specified via callbacks because retrieving the shape of a tensor
  // is a blocking operation for async eager; custom devices should avoid
  // retrieving shapes of tensors they wrap until the custom device tensor's
  // shape is explicitly requested where possible.
  int (*num_dims)(void* data, TF_Status* status);

  // Computes the axis length at `dim_index`.
  int64_t (*dim)(void* data, int dim_index, TF_Status* status);

  void (*deallocator)(void* data);

  // Summarizes the value of this tensor. The caller takes ownership of the
  // returned buffer. If `status` is not TF_OK, instead returns a null pointer.
  //
  // Does not include the shape and dtype of the tensor (which is generally
  // appended later), but should include any information specific to this custom
  // device which would be useful for debugging.
  //
  // Optional. If null, defaults to resolving the TFE_TensorHandle into a
  // TF_Tensor and summarizing that.
  TF_Buffer* (*summarize)(void* data, TF_Status* status) = nullptr;
} TFE_CustomDeviceTensorHandle;

// Creates a new TensorHandle from memory residing in a custom device. Takes
// ownership of the memory pointed to by `tensor_handle_data`, and calls
// `methods.deallocator` to release it after TF no longer needs it or in case of
// an error.
//
// This call is similar to `TFE_NewTensorHandleFromDeviceMemory`, but supports
// custom devices instead of physical devices and does not require blocking
// waiting for exact shapes.
TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_NewCustomDeviceTensorHandle(
    TFE_Context*, const char* device_name, TF_DataType, void* data,
    TFE_CustomDeviceTensorHandle methods, TF_Status* status);

TF_CAPI_EXPORT extern void TFE_ContextGetFunctionDef(TFE_Context* ctx,
                                                     const char* function_name,
                                                     TF_Buffer* buf,
                                                     TF_Status* status);

// Allocate and return a new Tensor on the host.
//
// The caller must set the Tensor values by writing them to the pointer returned
// by TF_TensorData with length TF_TensorByteSize.
TF_CAPI_EXPORT extern TF_Tensor* TFE_AllocateHostTensor(TFE_Context* ctx,
                                                        TF_DataType dtype,
                                                        const int64_t* dims,
                                                        int num_dims,
                                                        TF_Status* status);

// Given a Tensor, wrap it with a TensorHandle
//
// Similar to TFE_NewTensorHandle, but includes a pointer to the TFE_Context.
// The context should be identical to that of the Tensor.
TF_CAPI_EXPORT TFE_TensorHandle* TFE_NewTensorHandleFromTensor(
    TFE_Context* ctx, TF_Tensor* t, TF_Status* status);

// Create a packed TensorHandle with the given list of TensorHandles.
// If `handles` are on the same device, assign the same device to the packed
// handle; if `handles` are on different deivces, assign a CompositeDevice to
// it.
TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_CreatePackedTensorHandle(
    TFE_Context* ctx, TFE_TensorHandle** handles, int* num_handles,
    TF_Status* status);

// Configure soft device placement policy for the eager executor. Note this
// policy is applied to any subsequent op executions.
TF_CAPI_EXPORT void TFE_ContextSetSoftDevicePlacement(TFE_Context* ctx,
                                                      unsigned char enable,
                                                      TF_Status* status);

// Configure device placement policy logging for the eager executor. Note this
// policy is applied to any subsequent op executions.
TF_CAPI_EXPORT void TFE_ContextSetLogDevicePlacement(TFE_Context* ctx,
                                                     unsigned char enable,
                                                     TF_Status* status);

// Enables running eager ops as function.
TF_CAPI_EXPORT void TFE_ContextSetRunEagerOpAsFunction(TFE_Context* ctx,
                                                       unsigned char enable,
                                                       TF_Status* status);

// Enables rewrite jit_compile functions.
TF_CAPI_EXPORT void TFE_ContextSetJitCompileRewrite(TFE_Context* ctx,
                                                    unsigned char enable,
                                                    TF_Status* status);

// Returns the device type of the operation that produced `h`.
TF_CAPI_EXPORT extern const char* TFE_TensorHandleDeviceType(
    TFE_TensorHandle* h, TF_Status* status);

// Returns the device ID of the operation that produced `h`.
TF_CAPI_EXPORT extern int TFE_TensorHandleDeviceID(TFE_TensorHandle* h,
                                                   TF_Status* status);

// Returns the status for the tensor handle. In TFRT, a tensor handle can carry
// error info if error happens. If so, the status will be set with the error
// info. If not, status will be set as OK.
TF_CAPI_EXPORT extern void TFE_TensorHandleGetStatus(TFE_TensorHandle* h,
                                                     TF_Status* status);

// Get a comma-separated list of op names executed in graph functions dispatched
// to `ctx`. This feature is currently only enabled for TFRT debug builds, for
// performance and simplicity reasons.
TF_CAPI_EXPORT extern void TFE_GetExecutedOpNames(TFE_Context* ctx,
                                                  TF_Buffer* buf,
                                                  TF_Status* status);

// Set logical devices to the context's device manager.
// If logical devices are already configured at context initialization
// through TFE_ContextOptions, this method should not be called.
TF_CAPI_EXPORT extern void TFE_SetLogicalCpuDevices(TFE_Context* ctx,
                                                    int num_cpus,
                                                    const char* prefix,
                                                    TF_Status* status);

// Set configuration key and value using coordination service.
// If coordination service is enabled, the key-value will be stored on the
// leader and become accessible to all workers in the cluster.
// Currently, a config key can only be set with one value, and subsequently
// setting the same key will lead to errors.
//
// Note that the key-values are only expected to be used for cluster
// configuration data, and should not be used for storing a large amount of data
// or being accessed very frequently.
TF_CAPI_EXPORT extern void TFE_InsertConfigKeyValue(TFE_Context* ctx,
                                                    const char* key,
                                                    const char* value,
                                                    TF_Status* status);

// Get configuration key and value using coordination service.
// The config key must be set before getting its value. Getting value of
// non-existing config keys will result in errors.
TF_CAPI_EXPORT extern void TFE_GetConfigKeyValue(TFE_Context* ctx,
                                                 const char* key,
                                                 TF_Buffer* value_buf,
                                                 TF_Status* status);

// Delete configuration key-value. If `key` is a directory, recursively clean up
// all key-values under the path specified by `key`.
TF_CAPI_EXPORT extern void TFE_DeleteConfigKeyValue(TFE_Context* ctx,
                                                    const char* key,
                                                    TF_Status* status);

// Report error (specified by error_code and error_message) to other tasks in
// the cluster.
TF_CAPI_EXPORT extern void TFE_ReportErrorToCluster(TFE_Context* ctx,
                                                    int error_code,
                                                    const char* error_message,
                                                    TF_Status* status);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_EAGER_C_API_EXPERIMENTAL_H_
