/* Copyright 2015 Google Inc. All Rights Reserved.

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

// TODO(jeff,sanjay): Rename to tensorflow/public/c_api.h
#ifndef TENSORFLOW_PUBLIC_TENSOR_C_API_H_
#define TENSORFLOW_PUBLIC_TENSOR_C_API_H_

#include <stddef.h>

// --------------------------------------------------------------------------
// C API for TensorFlow.
//
// The API leans towards simplicity and uniformity instead of convenience
// since most usage will be by language specific wrappers.
//
// Conventions:
// * We use the prefix TF_ for everything in the API.
// * Objects are always passed around as pointers to opaque structs
//   and these structs are allocated/deallocated via the API.
// * TF_Status holds error information.  It is an object type
//   and therefore is passed around as a pointer to an opaque
//   struct as mentioned above.
// * Every call that has a TF_Status* argument clears it on success
//   and fills it with error info on failure.
//
// Questions left to address:
// * Might need to add stride info to TF_Tensor?
// * Might at some point need a way for callers to provide their own Env.
// * Should we remove the TF_Status arg from TF_AddProto calls and only
//   report errors later (e.g., on Run call).
// * Should dimensions be unsigned instead of signed?
// * Maybe add TF_TensorShape that encapsulates dimension info.
//
// Design decisions made:
// * Backing store for tensor memory has an associated deallocation
//   function.  This deallocation function will point to client code
//   for tensors populated by the client.  So the client can do things
//   like shadowing a numpy array.
// * We do not provide TF_OK since it is not strictly necessary and we
//   are not optimizing for convenience.
// * We make assumption that one session has one graph.  This should be
//   fine since we have the ability to run sub-graphs.
// * We are not providing TF_AddNode/TF_AddNodes to better support
//   languages/platforms where proto is not available.  This is because
//   we can just point authors of bindings at the .proto file and the
//   proto serialization spec and they can do the right thing for
//   their language.
// * We could allow NULL for some arguments (e.g., NULL options arg).
//   However since convenience is not a primary goal, we don't do this.
// * Devices are not in this API.  Instead, they are created/used internally
//   and the API just provides high level controls over the number of
//   devices of each type.

#ifdef __cplusplus
extern "C" {
#endif

// --------------------------------------------------------------------------
// TF_DataType holds the type for a scalar value.  E.g., one slot in a tensor.
// The enum values here are identical to corresponding values in types.proto.
typedef enum {
  TF_FLOAT = 1,
  TF_DOUBLE = 2,
  TF_INT32 = 3,  // Int32 tensors are always in 'host' memory.
  TF_UINT8 = 4,
  TF_INT16 = 5,
  TF_INT8 = 6,
  TF_STRING = 7,
  TF_COMPLEX64 = 8,  // Single-precision complex
  TF_COMPLEX = 8,    // Old identifier kept for API backwards compatibility
  TF_INT64 = 9,
  TF_BOOL = 10,
  TF_QINT8 = 11,     // Quantized int8
  TF_QUINT8 = 12,    // Quantized uint8
  TF_QINT32 = 13,    // Quantized int32
  TF_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  TF_QINT16 = 15,    // Quantized int16
  TF_QUINT16 = 16,   // Quantized uint16
  TF_UINT16 = 17,
  TF_COMPLEX128 = 18,  // Double-precision complex
  TF_HALF = 19,
} TF_DataType;

// --------------------------------------------------------------------------
// TF_Code holds an error code.  The enum values here are identical to
// corresponding values in error_codes.proto.
typedef enum {
  TF_OK = 0,
  TF_CANCELLED = 1,
  TF_UNKNOWN = 2,
  TF_INVALID_ARGUMENT = 3,
  TF_DEADLINE_EXCEEDED = 4,
  TF_NOT_FOUND = 5,
  TF_ALREADY_EXISTS = 6,
  TF_PERMISSION_DENIED = 7,
  TF_UNAUTHENTICATED = 16,
  TF_RESOURCE_EXHAUSTED = 8,
  TF_FAILED_PRECONDITION = 9,
  TF_ABORTED = 10,
  TF_OUT_OF_RANGE = 11,
  TF_UNIMPLEMENTED = 12,
  TF_INTERNAL = 13,
  TF_UNAVAILABLE = 14,
  TF_DATA_LOSS = 15,
} TF_Code;

// --------------------------------------------------------------------------
// TF_Status holds error information.  It either has an OK code, or
// else an error code with an associated error message.
typedef struct TF_Status TF_Status;

// --------------------------------------------------------------------------
// TF_Buffer holds a pointer to a block of data and its associated length.
// Typically, the data consists of a serialized protocol buffer, but other data
// may also be held in a buffer.
//
// By default, TF_Buffer itself does not do any memory management of the
// pointed-to block.  If need be, users of this struct should specify how to
// deallocate the block by setting the `data_deallocator` function pointer.
typedef struct {
  const void* data;
  size_t length;
  void (*data_deallocator)(void* data, size_t length);
} TF_Buffer;

// Makes a copy of the input and sets an appropriate deallocator.  Useful for
// passing in read-only, input protobufs.
extern TF_Buffer* TF_NewBufferFromString(const void* proto, size_t proto_len);

// Useful for passing *out* a protobuf.
extern TF_Buffer* TF_NewBuffer();

extern void TF_DeleteBuffer(TF_Buffer*);

extern TF_Buffer TF_GetBuffer(TF_Buffer* buffer);

// --------------------------------------------------------------------------
// TF_Library holds information about dynamically loaded TensorFlow plugins.
typedef struct TF_Library TF_Library;

// Return a new status object.
extern TF_Status* TF_NewStatus();

// Delete a previously created status object.
extern void TF_DeleteStatus(TF_Status*);

// Record <code, msg> in *s.  Any previous information is lost.
// A common use is to clear a status: TF_SetStatus(s, TF_OK, "");
extern void TF_SetStatus(TF_Status* s, TF_Code code, const char* msg);

// Return the code record in *s.
extern TF_Code TF_GetCode(const TF_Status* s);

// Return a pointer to the error message in *s.  The return value
// points to memory that is only usable until the next mutation to *s.
// Always returns an empty string if TF_GetCode(s) is TF_OK.
extern const char* TF_Message(const TF_Status* s);

// --------------------------------------------------------------------------
// TF_Tensor holds a multi-dimensional array of elements of a single data type.
// For all types other than TF_STRING, the data buffer stores elements
// in row major order.  E.g. if data is treated as a vector of TF_DataType:
//
//   element 0:   index (0, ..., 0)
//   element 1:   index (0, ..., 1)
//   ...
//
// TODO(jeff,sanjay): Define format for TF_STRING tensors.  Perhaps:
//   start_offset: array[uint64]
//   data:         byte[...]
//
//   String length is encoded (varint?) starting at data[start_offset[i]]
//   String contents follow immediately after string length.

typedef struct TF_Tensor TF_Tensor;

// Return a new tensor that holds the bytes data[0,len-1].
//
// The data will be deallocated by a subsequent call to TF_DeleteTensor via:
//      (*deallocator)(data, len, deallocator_arg)
// Clients must provide a custom deallocator function so they can pass in
// memory managed by something like numpy.
extern TF_Tensor* TF_NewTensor(TF_DataType, long long* dims, int num_dims,
                               void* data, size_t len,
                               void (*deallocator)(void* data, size_t len,
                                                   void* arg),
                               void* deallocator_arg);

// Destroy a tensor.
extern void TF_DeleteTensor(TF_Tensor*);

// Return the type of a tensor element.
extern TF_DataType TF_TensorType(const TF_Tensor*);

// Return the number of dimensions that the tensor has.
extern int TF_NumDims(const TF_Tensor*);

// Return the length of the tensor in the "dim_index" dimension.
// REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
extern long long TF_Dim(const TF_Tensor* tensor, int dim_index);

// Return the size of the underlying data in bytes.
extern size_t TF_TensorByteSize(const TF_Tensor*);

// Return a pointer to the underlying data buffer.
extern void* TF_TensorData(const TF_Tensor*);

// --------------------------------------------------------------------------
// TF_SessionOptions holds options that can be passed during session creation.
typedef struct TF_SessionOptions TF_SessionOptions;

// Return a new options object.
extern TF_SessionOptions* TF_NewSessionOptions();

// Set the target in TF_SessionOptions.options.
// target can be empty, a single entry, or a comma separated list of entries.
// Each entry is in one of the following formats :
// "local"
// ip:port
// host:port
extern void TF_SetTarget(TF_SessionOptions* options, const char* target);

// Set the config in TF_SessionOptions.options.
// config should be a serialized brain.ConfigProto proto.
// If config was not parsed successfully as a ConfigProto, record the
// error information in *status.
extern void TF_SetConfig(TF_SessionOptions* options, const void* proto,
                         size_t proto_len, TF_Status* status);

// Destroy an options object.
extern void TF_DeleteSessionOptions(TF_SessionOptions*);

// TODO(jeff,sanjay):
// - export functions to set Config fields

// --------------------------------------------------------------------------
// TF_Session manages a single graph and execution.
typedef struct TF_Session TF_Session;

// Return a new execution session, or NULL on error.
extern TF_Session* TF_NewSession(const TF_SessionOptions*, TF_Status* status);

// Close a session.
extern void TF_CloseSession(TF_Session*, TF_Status* status);

// Destroy a session.  Even if error information is recorded in *status,
// this call discards all resources associated with the session.
extern void TF_DeleteSession(TF_Session*, TF_Status* status);

// Treat the bytes proto[0,proto_len-1] as a serialized GraphDef and
// add the nodes in that GraphDef to the graph for the session.
extern void TF_ExtendGraph(TF_Session*, const void* proto, size_t proto_len,
                           TF_Status*);

// Run the graph associated with the session starting with the
// supplied inputs (inputs[0,ninputs-1]).  Regardless of success or
// failure, inputs[] become the property of the implementation (the
// implementation will eventually call TF_DeleteTensor on each input).
//
// Any NULL and non-NULL value combinations for (`run_options`,
// `run_metadata`) are valid.
//
//    - `run_options` may be NULL, in which case it will be ignored; or
//      non-NULL, in which case it must point to a `TF_Buffer` containing the
//      serialized representation of a `RunOptions` protocol buffer.
//    - `run_metadata` may be NULL, in which case it will be ignored; or
//      non-NULL, in which case it must point to an empty, freshly allocated
//      `TF_Buffer` that may be updated to contain the serialized representation
//      of a `RunMetadata` protocol buffer.
//
// The caller retains the ownership of `run_options` and/or `run_metadata` (when
// not NULL) and should manually call TF_DeleteBuffer on them.
//
// On success, the tensors corresponding to output_names[0,noutputs-1]
// are placed in outputs[], and these outputs[] become the property
// of the caller (the caller must eventually call TF_DeleteTensor on
// them).
//
// On failure, outputs[] contains NULLs.
extern void TF_Run(TF_Session*,
                   // RunOptions
                   const TF_Buffer* run_options,
                   // Input tensors
                   const char** input_names, TF_Tensor** inputs, int ninputs,
                   // Output tensors
                   const char** output_tensor_names, TF_Tensor** outputs,
                   int noutputs,
                   // Target nodes
                   const char** target_node_names, int ntargets,
                   // RunMetadata
                   TF_Buffer* run_metadata,
                   // Output status
                   TF_Status*);

// Set up the graph with the intended feeds and fetches for a sequence
// of partial run calls.
//
// On success, returns a handle that is used for subsequent PRun calls.
//
// On failure, out_status contains a tensorflow::Status with an error
// message.
// NOTE: This is EXPERIMENTAL and subject to change.
extern void TF_PRunSetup(TF_Session*,
                         // Input names
                         const char** input_names, int ninputs,
                         // Output names
                         const char** output_tensor_names, int noutputs,
                         // Target nodes
                         const char** target_node_names, int ntargets,
                         // Output handle
                         char** handle,
                         // Output status
                         TF_Status*);

// Continue to run the graph with additional feeds and fetches. The
// execution state is uniquely identified by the handle.
// NOTE: This is EXPERIMENTAL and subject to change.
extern void TF_PRun(TF_Session*, const char* handle,
                    // Input tensors
                    const char** input_names, TF_Tensor** inputs, int ninputs,
                    // Output tensors
                    const char** output_tensor_names, TF_Tensor** outputs,
                    int noutputs,
                    // Target nodes
                    const char** target_node_names, int ntargets,
                    // Output status
                    TF_Status*);

// --------------------------------------------------------------------------
// Load plugins containing custom ops and kernels

// Load the library specified by library_filename and register the ops and
// kernels present in that library.
//
// Pass "library_filename" to a platform-specific mechanism for dynamically
// loading a library. The rules for determining the exact location of the
// library are platform-specific and are not documented here.
// Expects the symbols "RegisterOps", "RegisterKernels", and "GetOpList", to be
// defined in the library.
//
// On success, place OK in status and return the newly created library handle.
// The caller owns the library handle.
//
// On failure, place an error status in status and return NULL.
extern TF_Library* TF_LoadLibrary(const char* library_filename,
                                  TF_Status* status);

// Get the OpList of OpDefs defined in the library pointed by lib_handle.
//
// Returns a TF_Buffer. The memory pointed to by the result is owned by
// lib_handle. The data in the buffer will be the serialized OpList proto for
// ops defined in the library.
extern TF_Buffer TF_GetOpList(TF_Library* lib_handle);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_PUBLIC_TENSOR_C_API_H_
