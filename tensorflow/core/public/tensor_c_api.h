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
//   and threfore is passed around as a pointer to an opaque
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
  TF_COMPLEX = 8,  // Single-precision complex
  TF_INT64 = 9,
  TF_BOOL = 10,
  TF_QINT8 = 11,     // Quantized int8
  TF_QUINT8 = 12,    // Quantized uint8
  TF_QINT32 = 13,    // Quantized int32
  TF_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
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
//      (*deallocator_fn)(data, len, deallocator_arg)
// Clients can provide a custom deallocator function so they can pass in
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
extern void TF_SetConfig(TF_SessionOptions* options, const char* config,
                         size_t config_len, TF_Status* status);

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
// On success, the tensors corresponding to output_names[0,noutputs-1]
// are placed in outputs[].  and these outputs[] become the property
// of the caller (the caller must eventually call TF_DeleteTensor on
// them).
//
// On failure, outputs[] contains nulls.
extern void TF_Run(TF_Session*,
                   // Input tensors
                   const char** input_names, TF_Tensor** inputs, int ninputs,
                   // Output tensors
                   const char** output_tensor_names, TF_Tensor** outputs,
                   int noutputs,
                   // Target nodes
                   const char** target_node_names, int ntargets,
                   // Output status
                   TF_Status*);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_PUBLIC_TENSOR_C_API_H_
