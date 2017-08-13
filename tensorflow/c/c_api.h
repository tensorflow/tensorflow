/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_C_API_H_
#define TENSORFLOW_C_C_API_H_

#include <stddef.h>
#include <stdint.h>

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
// * unsigned char is used for booleans (instead of the 'bool' type).
//   In C++ bool is a keyword while in C99 bool is a macro defined
//   in stdbool.h. It is possible for the two to be inconsistent.
//   For example, neither the C99 nor the C++11 standard force a byte
//   size on the bool type, so the macro defined in stdbool.h could
//   be inconsistent with the bool keyword in C++. Thus, the use
//   of stdbool.h is avoided and unsigned char is used instead.
// * size_t is used to represent byte sizes of objects that are
//   materialized in the address space of the calling process.
// * int is used as an index into arrays.
//
// Questions left to address:
// * Might at some point need a way for callers to provide their own Env.
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
// * We could allow NULL for some arguments (e.g., NULL options arg).
//   However since convenience is not a primary goal, we don't do this.
// * Devices are not in this API.  Instead, they are created/used internally
//   and the API just provides high level controls over the number of
//   devices of each type.

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

// --------------------------------------------------------------------------
// TF_Version returns a string describing version information of the
// TensorFlow library. TensorFlow using semantic versioning.
TF_CAPI_EXPORT extern const char* TF_Version();

// --------------------------------------------------------------------------
// TF_DataType holds the type for a scalar value.  E.g., one slot in a tensor.
// The enum values here are identical to corresponding values in types.proto.
typedef enum TF_DataType {
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
  TF_RESOURCE = 20,
  TF_VARIANT = 21,
} TF_DataType;

// TF_DataTypeSize returns the sizeof() for the underlying type corresponding
// to the given TF_DataType enum value. Returns 0 for variable length types
// (eg. TF_STRING) or on failure.
TF_CAPI_EXPORT extern size_t TF_DataTypeSize(TF_DataType dt);

// --------------------------------------------------------------------------
// TF_Code holds an error code.  The enum values here are identical to
// corresponding values in error_codes.proto.
typedef enum TF_Code {
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
TF_CAPI_EXPORT extern TF_Status* TF_NewStatus();

// Delete a previously created status object.
TF_CAPI_EXPORT extern void TF_DeleteStatus(TF_Status*);

// Record <code, msg> in *s.  Any previous information is lost.
// A common use is to clear a status: TF_SetStatus(s, TF_OK, "");
TF_CAPI_EXPORT extern void TF_SetStatus(TF_Status* s, TF_Code code,
                                        const char* msg);

// Return the code record in *s.
TF_CAPI_EXPORT extern TF_Code TF_GetCode(const TF_Status* s);

// Return a pointer to the (null-terminated) error message in *s.  The
// return value points to memory that is only usable until the next
// mutation to *s.  Always returns an empty string if TF_GetCode(s) is
// TF_OK.
TF_CAPI_EXPORT extern const char* TF_Message(const TF_Status* s);

// --------------------------------------------------------------------------
// TF_Buffer holds a pointer to a block of data and its associated length.
// Typically, the data consists of a serialized protocol buffer, but other data
// may also be held in a buffer.
//
// By default, TF_Buffer itself does not do any memory management of the
// pointed-to block.  If need be, users of this struct should specify how to
// deallocate the block by setting the `data_deallocator` function pointer.
typedef struct TF_Buffer {
  const void* data;
  size_t length;
  void (*data_deallocator)(void* data, size_t length);
} TF_Buffer;

// Makes a copy of the input and sets an appropriate deallocator.  Useful for
// passing in read-only, input protobufs.
TF_CAPI_EXPORT extern TF_Buffer* TF_NewBufferFromString(const void* proto,
                                                        size_t proto_len);

// Useful for passing *out* a protobuf.
TF_CAPI_EXPORT extern TF_Buffer* TF_NewBuffer();

TF_CAPI_EXPORT extern void TF_DeleteBuffer(TF_Buffer*);

TF_CAPI_EXPORT extern TF_Buffer TF_GetBuffer(TF_Buffer* buffer);

// --------------------------------------------------------------------------
// TF_Tensor holds a multi-dimensional array of elements of a single data type.
// For all types other than TF_STRING, the data buffer stores elements
// in row major order.  E.g. if data is treated as a vector of TF_DataType:
//
//   element 0:   index (0, ..., 0)
//   element 1:   index (0, ..., 1)
//   ...
//
// The format for TF_STRING tensors is:
//   start_offset: array[uint64]
//   data:         byte[...]
//
//   The string length (as a varint), followed by the contents of the string
//   is encoded at data[start_offset[i]]]. TF_StringEncode and TF_StringDecode
//   facilitate this encoding.

typedef struct TF_Tensor TF_Tensor;

// Return a new tensor that holds the bytes data[0,len-1].
//
// The data will be deallocated by a subsequent call to TF_DeleteTensor via:
//      (*deallocator)(data, len, deallocator_arg)
// Clients must provide a custom deallocator function so they can pass in
// memory managed by something like numpy.
TF_CAPI_EXPORT extern TF_Tensor* TF_NewTensor(
    TF_DataType, const int64_t* dims, int num_dims, void* data, size_t len,
    void (*deallocator)(void* data, size_t len, void* arg),
    void* deallocator_arg);

// Allocate and return a new Tensor.
//
// This function is an alternative to TF_NewTensor and should be used when
// memory is allocated to pass the Tensor to the C API. The allocated memory
// satisfies TensorFlow's memory alignment preferences and should be preferred
// over calling malloc and free.
//
// The caller must set the Tensor values by writing them to the pointer returned
// by TF_TensorData with length TF_TensorByteSize.
TF_CAPI_EXPORT extern TF_Tensor* TF_AllocateTensor(TF_DataType,
                                                   const int64_t* dims,
                                                   int num_dims, size_t len);

// Deletes `tensor` and returns a new TF_Tensor with the same content if
// possible. Returns nullptr and leaves `tensor` untouched if not.
TF_CAPI_EXPORT extern TF_Tensor* TF_TensorMaybeMove(TF_Tensor* tensor);

// Destroy a tensor.
TF_CAPI_EXPORT extern void TF_DeleteTensor(TF_Tensor*);

// Return the type of a tensor element.
TF_CAPI_EXPORT extern TF_DataType TF_TensorType(const TF_Tensor*);

// Return the number of dimensions that the tensor has.
TF_CAPI_EXPORT extern int TF_NumDims(const TF_Tensor*);

// Return the length of the tensor in the "dim_index" dimension.
// REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
TF_CAPI_EXPORT extern int64_t TF_Dim(const TF_Tensor* tensor, int dim_index);

// Return the size of the underlying data in bytes.
TF_CAPI_EXPORT extern size_t TF_TensorByteSize(const TF_Tensor*);

// Return a pointer to the underlying data buffer.
TF_CAPI_EXPORT extern void* TF_TensorData(const TF_Tensor*);

// --------------------------------------------------------------------------
// Encode the string `src` (`src_len` bytes long) into `dst` in the format
// required by TF_STRING tensors. Does not write to memory more than `dst_len`
// bytes beyond `*dst`. `dst_len` should be at least
// TF_StringEncodedSize(src_len).
//
// On success returns the size in bytes of the encoded string.
// Returns an error into `status` otherwise.
TF_CAPI_EXPORT extern size_t TF_StringEncode(const char* src, size_t src_len,
                                             char* dst, size_t dst_len,
                                             TF_Status* status);

// Decode a string encoded using TF_StringEncode.
//
// On success, sets `*dst` to the start of the decoded string and `*dst_len` to
// its length. Returns the number of bytes starting at `src` consumed while
// decoding. `*dst` points to memory within the encoded buffer.  On failure,
// `*dst` and `*dst_len` are undefined and an error is set in `status`.
//
// Does not read memory more than `src_len` bytes beyond `src`.
TF_CAPI_EXPORT extern size_t TF_StringDecode(const char* src, size_t src_len,
                                             const char** dst, size_t* dst_len,
                                             TF_Status* status);

// Return the size in bytes required to encode a string `len` bytes long into a
// TF_STRING tensor.
TF_CAPI_EXPORT extern size_t TF_StringEncodedSize(size_t len);

// --------------------------------------------------------------------------
// TF_SessionOptions holds options that can be passed during session creation.
typedef struct TF_SessionOptions TF_SessionOptions;

// Return a new options object.
TF_CAPI_EXPORT extern TF_SessionOptions* TF_NewSessionOptions();

// Set the target in TF_SessionOptions.options.
// target can be empty, a single entry, or a comma separated list of entries.
// Each entry is in one of the following formats :
// "local"
// ip:port
// host:port
TF_CAPI_EXPORT extern void TF_SetTarget(TF_SessionOptions* options,
                                        const char* target);

// Set the config in TF_SessionOptions.options.
// config should be a serialized tensorflow.ConfigProto proto.
// If config was not parsed successfully as a ConfigProto, record the
// error information in *status.
TF_CAPI_EXPORT extern void TF_SetConfig(TF_SessionOptions* options,
                                        const void* proto, size_t proto_len,
                                        TF_Status* status);

// Destroy an options object.
TF_CAPI_EXPORT extern void TF_DeleteSessionOptions(TF_SessionOptions*);

// TODO(jeff,sanjay):
// - export functions to set Config fields

// --------------------------------------------------------------------------
// The new graph construction API, still under development.

// Represents a computation graph.  Graphs may be shared between sessions.
// Graphs are thread-safe when used as directed below.
typedef struct TF_Graph TF_Graph;

// Return a new graph object.
TF_CAPI_EXPORT extern TF_Graph* TF_NewGraph();

// Destroy an options object.  Graph will be deleted once no more
// TFSession's are referencing it.
TF_CAPI_EXPORT extern void TF_DeleteGraph(TF_Graph*);

// Operation being built. The underlying graph must outlive this.
typedef struct TF_OperationDescription TF_OperationDescription;

// Operation that has been added to the graph. Valid until the graph is
// deleted -- in particular adding a new operation to the graph does not
// invalidate old TF_Operation* pointers.
typedef struct TF_Operation TF_Operation;

// Represents a specific input of an operation.
typedef struct TF_Input {
  TF_Operation* oper;
  int index;  // The index of the input within oper.
} TF_Input;

// Represents a specific output of an operation.
typedef struct TF_Output {
  TF_Operation* oper;
  int index;  // The index of the output within oper.
} TF_Output;

// Sets the shape of the Tensor referenced by `output` in `graph` to
// the shape described by `dims` and `num_dims`.
//
// If the number of dimensions is unknown, `num_dims` must be set to
// -1 and `dims` can be null. If a dimension is unknown, the
// corresponding entry in the `dims` array must be -1.
//
// This does not overwrite the existing shape associated with `output`,
// but merges the input shape with the existing shape.  For example,
// setting a shape of [-1, 2] with an existing shape [2, -1] would set
// a final shape of [2, 2] based on shape merging semantics.
//
// Returns an error into `status` if:
//   * `output` is not in `graph`.
//   * An invalid shape is being set (e.g., the shape being set
//     is incompatible with the existing shape).
TF_CAPI_EXPORT extern void TF_GraphSetTensorShape(TF_Graph* graph,
                                                  TF_Output output,
                                                  const int64_t* dims,
                                                  const int num_dims,
                                                  TF_Status* status);

// Returns the number of dimensions of the Tensor referenced by `output`
// in `graph`.
//
// If the number of dimensions in the shape is unknown, returns -1.
//
// Returns an error into `status` if:
//   * `output` is not in `graph`.
TF_CAPI_EXPORT extern int TF_GraphGetTensorNumDims(TF_Graph* graph,
                                                   TF_Output output,
                                                   TF_Status* status);

// Returns the shape of the Tensor referenced by `output` in `graph`
// into `dims`. `dims` must be an array large enough to hold `num_dims`
// entries (e.g., the return value of TF_GraphGetTensorNumDims).
//
// If the number of dimensions in the shape is unknown or the shape is
// a scalar, `dims` will remain untouched. Otherwise, each element of
// `dims` will be set corresponding to the size of the dimension. An
// unknown dimension is represented by `-1`.
//
// Returns an error into `status` if:
//   * `output` is not in `graph`.
//   * `num_dims` does not match the actual number of dimensions.
TF_CAPI_EXPORT extern void TF_GraphGetTensorShape(TF_Graph* graph,
                                                  TF_Output output,
                                                  int64_t* dims, int num_dims,
                                                  TF_Status* status);

// Operation will only be added to *graph when TF_FinishOperation() is
// called (assuming TF_FinishOperation() does not return an error).
// *graph must not be deleted until after TF_FinishOperation() is
// called.
TF_CAPI_EXPORT extern TF_OperationDescription* TF_NewOperation(
    TF_Graph* graph, const char* op_type, const char* oper_name);

// Specify the device for `desc`.  Defaults to empty, meaning unconstrained.
TF_CAPI_EXPORT extern void TF_SetDevice(TF_OperationDescription* desc,
                                        const char* device);

// The calls to TF_AddInput and TF_AddInputList must match (in number,
// order, and type) the op declaration.  For example, the "Concat" op
// has registration:
//   REGISTER_OP("Concat")
//       .Input("concat_dim: int32")
//       .Input("values: N * T")
//       .Output("output: T")
//       .Attr("N: int >= 2")
//       .Attr("T: type");
// that defines two inputs, "concat_dim" and "values" (in that order).
// You must use TF_AddInput() for the first input (since it takes a
// single tensor), and TF_AddInputList() for the second input (since
// it takes a list, even if you were to pass a list with a single
// tensor), as in:
//   TF_OperationDescription* desc = TF_NewOperation(graph, "Concat", "c");
//   TF_Output concat_dim_input = {...};
//   TF_AddInput(desc, concat_dim_input);
//   TF_Output values_inputs[5] = {{...}, ..., {...}};
//   TF_AddInputList(desc, values_inputs, 5);

// For inputs that take a single tensor.
TF_CAPI_EXPORT extern void TF_AddInput(TF_OperationDescription* desc,
                                       TF_Output input);

// For inputs that take a list of tensors.
// inputs must point to TF_Output[num_inputs].
TF_CAPI_EXPORT extern void TF_AddInputList(TF_OperationDescription* desc,
                                           const TF_Output* inputs,
                                           int num_inputs);

// Call once per control input to `desc`.
TF_CAPI_EXPORT extern void TF_AddControlInput(TF_OperationDescription* desc,
                                              TF_Operation* input);

// Request that `desc` be co-located on the device where `op`
// is placed.
//
// Use of this is discouraged since the implementation of device placement is
// subject to change. Primarily intended for internal libraries
TF_CAPI_EXPORT extern void TF_ColocateWith(TF_OperationDescription* desc,
                                           TF_Operation* op);

// Call some TF_SetAttr*() function for every attr that is not
// inferred from an input and doesn't have a default value you wish to
// keep.

// `value` must point to a string of length `length` bytes.
TF_CAPI_EXPORT extern void TF_SetAttrString(TF_OperationDescription* desc,
                                            const char* attr_name,
                                            const void* value, size_t length);
// `values` and `lengths` each must have lengths `num_values`.
// `values[i]` must point to a string of length `lengths[i]` bytes.
TF_CAPI_EXPORT extern void TF_SetAttrStringList(TF_OperationDescription* desc,
                                                const char* attr_name,
                                                const void* const* values,
                                                const size_t* lengths,
                                                int num_values);
TF_CAPI_EXPORT extern void TF_SetAttrInt(TF_OperationDescription* desc,
                                         const char* attr_name, int64_t value);
TF_CAPI_EXPORT extern void TF_SetAttrIntList(TF_OperationDescription* desc,
                                             const char* attr_name,
                                             const int64_t* values,
                                             int num_values);
TF_CAPI_EXPORT extern void TF_SetAttrFloat(TF_OperationDescription* desc,
                                           const char* attr_name, float value);
TF_CAPI_EXPORT extern void TF_SetAttrFloatList(TF_OperationDescription* desc,
                                               const char* attr_name,
                                               const float* values,
                                               int num_values);
TF_CAPI_EXPORT extern void TF_SetAttrBool(TF_OperationDescription* desc,
                                          const char* attr_name,
                                          unsigned char value);
TF_CAPI_EXPORT extern void TF_SetAttrBoolList(TF_OperationDescription* desc,
                                              const char* attr_name,
                                              const unsigned char* values,
                                              int num_values);
TF_CAPI_EXPORT extern void TF_SetAttrType(TF_OperationDescription* desc,
                                          const char* attr_name,
                                          TF_DataType value);
TF_CAPI_EXPORT extern void TF_SetAttrTypeList(TF_OperationDescription* desc,
                                              const char* attr_name,
                                              const TF_DataType* values,
                                              int num_values);

// Set `num_dims` to -1 to represent "unknown rank".  Otherwise,
// `dims` points to an array of length `num_dims`.  `dims[i]` must be
// >= -1, with -1 meaning "unknown dimension".
TF_CAPI_EXPORT extern void TF_SetAttrShape(TF_OperationDescription* desc,
                                           const char* attr_name,
                                           const int64_t* dims, int num_dims);
// `dims` and `num_dims` must point to arrays of length `num_shapes`.
// Set `num_dims[i]` to -1 to represent "unknown rank".  Otherwise,
// `dims[i]` points to an array of length `num_dims[i]`.  `dims[i][j]`
// must be >= -1, with -1 meaning "unknown dimension".
TF_CAPI_EXPORT extern void TF_SetAttrShapeList(TF_OperationDescription* desc,
                                               const char* attr_name,
                                               const int64_t* const* dims,
                                               const int* num_dims,
                                               int num_shapes);
// `proto` must point to an array of `proto_len` bytes representing a
// binary-serialized TensorShapeProto.
TF_CAPI_EXPORT extern void TF_SetAttrTensorShapeProto(
    TF_OperationDescription* desc, const char* attr_name, const void* proto,
    size_t proto_len, TF_Status* status);
// `protos` and `proto_lens` must point to arrays of length `num_shapes`.
// `protos[i]` must point to an array of `proto_lens[i]` bytes
// representing a binary-serialized TensorShapeProto.
TF_CAPI_EXPORT extern void TF_SetAttrTensorShapeProtoList(
    TF_OperationDescription* desc, const char* attr_name,
    const void* const* protos, const size_t* proto_lens, int num_shapes,
    TF_Status* status);

TF_CAPI_EXPORT extern void TF_SetAttrTensor(TF_OperationDescription* desc,
                                            const char* attr_name,
                                            TF_Tensor* value,
                                            TF_Status* status);
TF_CAPI_EXPORT extern void TF_SetAttrTensorList(TF_OperationDescription* desc,
                                                const char* attr_name,
                                                TF_Tensor* const* values,
                                                int num_values,
                                                TF_Status* status);

// `proto` should point to a sequence of bytes of length `proto_len`
// representing a binary serialization of an AttrValue protocol
// buffer.
TF_CAPI_EXPORT extern void TF_SetAttrValueProto(TF_OperationDescription* desc,
                                                const char* attr_name,
                                                const void* proto,
                                                size_t proto_len,
                                                TF_Status* status);

// If this function succeeds:
//   * *status is set to an OK value,
//   * a TF_Operation is added to the graph,
//   * a non-null value pointing to the added operation is returned --
//     this value is valid until the underlying graph is deleted.
// Otherwise:
//   * *status is set to a non-OK value,
//   * the graph is not modified,
//   * a null value is returned.
// In either case, it deletes `desc`.
TF_CAPI_EXPORT extern TF_Operation* TF_FinishOperation(
    TF_OperationDescription* desc, TF_Status* status);

// TF_Operation functions.  Operations are immutable once created, so
// these are all query functions.

TF_CAPI_EXPORT extern const char* TF_OperationName(TF_Operation* oper);
TF_CAPI_EXPORT extern const char* TF_OperationOpType(TF_Operation* oper);
TF_CAPI_EXPORT extern const char* TF_OperationDevice(TF_Operation* oper);

TF_CAPI_EXPORT extern int TF_OperationNumOutputs(TF_Operation* oper);
TF_CAPI_EXPORT extern TF_DataType TF_OperationOutputType(TF_Output oper_out);
TF_CAPI_EXPORT extern int TF_OperationOutputListLength(TF_Operation* oper,
                                                       const char* arg_name,
                                                       TF_Status* status);

TF_CAPI_EXPORT extern int TF_OperationNumInputs(TF_Operation* oper);
TF_CAPI_EXPORT extern TF_DataType TF_OperationInputType(TF_Input oper_in);
TF_CAPI_EXPORT extern int TF_OperationInputListLength(TF_Operation* oper,
                                                      const char* arg_name,
                                                      TF_Status* status);

// In this code:
//   TF_Output producer = TF_OperationInput(consumer);
// There is an edge from producer.oper's output (given by
// producer.index) to consumer.oper's input (given by consumer.index).
TF_CAPI_EXPORT extern TF_Output TF_OperationInput(TF_Input oper_in);

// Get the number of current consumers of a specific output of an
// operation.  Note that this number can change when new operations
// are added to the graph.
TF_CAPI_EXPORT extern int TF_OperationOutputNumConsumers(TF_Output oper_out);

// Get list of all current consumers of a specific output of an
// operation.  `consumers` must point to an array of length at least
// `max_consumers` (ideally set to
// TF_OperationOutputNumConsumers(oper_out)).  Beware that a concurrent
// modification of the graph can increase the number of consumers of
// an operation.  Returns the number of output consumers (should match
// TF_OperationOutputNumConsumers(oper_out)).
TF_CAPI_EXPORT extern int TF_OperationOutputConsumers(TF_Output oper_out,
                                                      TF_Input* consumers,
                                                      int max_consumers);

// Get the number of control inputs to an operation.
TF_CAPI_EXPORT extern int TF_OperationNumControlInputs(TF_Operation* oper);

// Get list of all control inputs to an operation.  `control_inputs` must
// point to an array of length `max_control_inputs` (ideally set to
// TF_OperationNumControlInputs(oper)).  Returns the number of control
// inputs (should match TF_OperationNumControlInputs(oper)).
TF_CAPI_EXPORT extern int TF_OperationGetControlInputs(
    TF_Operation* oper, TF_Operation** control_inputs, int max_control_inputs);

// Get the number of operations that have `*oper` as a control input.
// Note that this number can change when new operations are added to
// the graph.
TF_CAPI_EXPORT extern int TF_OperationNumControlOutputs(TF_Operation* oper);

// Get the list of operations that have `*oper` as a control input.
// `control_outputs` must point to an array of length at least
// `max_control_outputs` (ideally set to
// TF_OperationNumControlOutputs(oper)). Beware that a concurrent
// modification of the graph can increase the number of control
// outputs.  Returns the number of control outputs (should match
// TF_OperationNumControlOutputs(oper)).
TF_CAPI_EXPORT extern int TF_OperationGetControlOutputs(
    TF_Operation* oper, TF_Operation** control_outputs,
    int max_control_outputs);

// TF_AttrType describes the type of the value of an attribute on an operation.
typedef enum TF_AttrType {
  TF_ATTR_STRING = 0,
  TF_ATTR_INT = 1,
  TF_ATTR_FLOAT = 2,
  TF_ATTR_BOOL = 3,
  TF_ATTR_TYPE = 4,
  TF_ATTR_SHAPE = 5,
  TF_ATTR_TENSOR = 6,
  TF_ATTR_PLACEHOLDER = 7,
  TF_ATTR_FUNC = 8,
} TF_AttrType;

// TF_AttrMetadata describes the value of an attribute on an operation.
typedef struct TF_AttrMetadata {
  // A boolean: 1 if the attribute value is a list, 0 otherwise.
  unsigned char is_list;

  // Length of the list if is_list is true. Undefined otherwise.
  int64_t list_size;

  // Type of elements of the list if is_list != 0.
  // Type of the single value stored in the attribute if is_list == 0.
  TF_AttrType type;

  // Total size the attribute value.
  // The units of total_size depend on is_list and type.
  // (1) If type == TF_ATTR_STRING and is_list == 0
  //     then total_size is the byte size of the string
  //     valued attribute.
  // (2) If type == TF_ATTR_STRING and is_list == 1
  //     then total_size is the cumulative byte size
  //     of all the strings in the list.
  // (3) If type == TF_ATTR_SHAPE and is_list == 0
  //     then total_size is the number of dimensions
  //     of the shape valued attribute, or -1
  //     if its rank is unknown.
  // (4) If type == TF_ATTR_SHAPE and is_list == 1
  //     then total_size is the cumulative number
  //     of dimensions of all shapes in the list.
  // (5) Otherwise, total_size is undefined.
  int64_t total_size;
} TF_AttrMetadata;

// Returns metadata about the value of the attribute `attr_name` of `oper`.
TF_CAPI_EXPORT extern TF_AttrMetadata TF_OperationGetAttrMetadata(
    TF_Operation* oper, const char* attr_name, TF_Status* status);

// Fills in `value` with the value of the attribute `attr_name`.  `value` must
// point to an array of length at least `max_length` (ideally set to
// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
TF_CAPI_EXPORT extern void TF_OperationGetAttrString(TF_Operation* oper,
                                                     const char* attr_name,
                                                     void* value,
                                                     size_t max_length,
                                                     TF_Status* status);

// Get the list of strings in the value of the attribute `attr_name`.  Fills in
// `values` and `lengths`, each of which must point to an array of length at
// least `max_values`.
//
// The elements of values will point to addresses in `storage` which must be at
// least `storage_size` bytes in length.  Ideally, max_values would be set to
// TF_AttrMetadata.list_size and `storage` would be at least
// TF_AttrMetadata.total_size, obtained from TF_OperationGetAttrMetadata(oper,
// attr_name).
//
// Fails if storage_size is too small to hold the requested number of strings.
TF_CAPI_EXPORT extern void TF_OperationGetAttrStringList(
    TF_Operation* oper, const char* attr_name, void** values, size_t* lengths,
    int max_values, void* storage, size_t storage_size, TF_Status* status);

TF_CAPI_EXPORT extern void TF_OperationGetAttrInt(TF_Operation* oper,
                                                  const char* attr_name,
                                                  int64_t* value,
                                                  TF_Status* status);

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
TF_CAPI_EXPORT extern void TF_OperationGetAttrIntList(TF_Operation* oper,
                                                      const char* attr_name,
                                                      int64_t* values,
                                                      int max_values,
                                                      TF_Status* status);

TF_CAPI_EXPORT extern void TF_OperationGetAttrFloat(TF_Operation* oper,
                                                    const char* attr_name,
                                                    float* value,
                                                    TF_Status* status);

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
TF_CAPI_EXPORT extern void TF_OperationGetAttrFloatList(TF_Operation* oper,
                                                        const char* attr_name,
                                                        float* values,
                                                        int max_values,
                                                        TF_Status* status);

TF_CAPI_EXPORT extern void TF_OperationGetAttrBool(TF_Operation* oper,
                                                   const char* attr_name,
                                                   unsigned char* value,
                                                   TF_Status* status);

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
TF_CAPI_EXPORT extern void TF_OperationGetAttrBoolList(TF_Operation* oper,
                                                       const char* attr_name,
                                                       unsigned char* values,
                                                       int max_values,
                                                       TF_Status* status);

TF_CAPI_EXPORT extern void TF_OperationGetAttrType(TF_Operation* oper,
                                                   const char* attr_name,
                                                   TF_DataType* value,
                                                   TF_Status* status);

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
TF_CAPI_EXPORT extern void TF_OperationGetAttrTypeList(TF_Operation* oper,
                                                       const char* attr_name,
                                                       TF_DataType* values,
                                                       int max_values,
                                                       TF_Status* status);

// Fills in `value` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `num_dims` (ideally set to
// TF_Attr_Meta.size from TF_OperationGetAttrMetadata(oper, attr_name)).
TF_CAPI_EXPORT extern void TF_OperationGetAttrShape(TF_Operation* oper,
                                                    const char* attr_name,
                                                    int64_t* value,
                                                    int num_dims,
                                                    TF_Status* status);

// Fills in `dims` with the list of shapes in the attribute `attr_name` of
// `oper` and `num_dims` with the corresponding number of dimensions. On return,
// for every i where `num_dims[i]` > 0, `dims[i]` will be an array of
// `num_dims[i]` elements. A value of -1 for `num_dims[i]` indicates that the
// i-th shape in the list is unknown.
//
// The elements of `dims` will point to addresses in `storage` which must be
// large enough to hold at least `storage_size` int64_ts.  Ideally, `num_shapes`
// would be set to TF_AttrMetadata.list_size and `storage_size` would be set to
// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
// attr_name).
//
// Fails if storage_size is insufficient to hold the requested shapes.
TF_CAPI_EXPORT extern void TF_OperationGetAttrShapeList(
    TF_Operation* oper, const char* attr_name, int64_t** dims, int* num_dims,
    int num_shapes, int64_t* storage, int storage_size, TF_Status* status);

// Sets `value` to the binary-serialized TensorShapeProto of the value of
// `attr_name` attribute of `oper`'.
TF_CAPI_EXPORT extern void TF_OperationGetAttrTensorShapeProto(
    TF_Operation* oper, const char* attr_name, TF_Buffer* value,
    TF_Status* status);

// Fills in `values` with binary-serialized TensorShapeProto values of the
// attribute `attr_name` of `oper`. `values` must point to an array of length at
// least `num_values` (ideally set to TF_AttrMetadata.list_size from
// TF_OperationGetAttrMetadata(oper, attr_name)).
TF_CAPI_EXPORT extern void TF_OperationGetAttrTensorShapeProtoList(
    TF_Operation* oper, const char* attr_name, TF_Buffer** values,
    int max_values, TF_Status* status);

// Gets the TF_Tensor valued attribute of `attr_name` of `oper`.
//
// Allocates a new TF_Tensor which the caller is expected to take
// ownership of (and can deallocate using TF_DeleteTensor).
TF_CAPI_EXPORT extern void TF_OperationGetAttrTensor(TF_Operation* oper,
                                                     const char* attr_name,
                                                     TF_Tensor** value,
                                                     TF_Status* status);

// Fills in `values` with the TF_Tensor values of the attribute `attr_name` of
// `oper`. `values` must point to an array of TF_Tensor* of length at least
// `max_values` (ideally set to TF_AttrMetadata.list_size from
// TF_OperationGetAttrMetadata(oper, attr_name)).
//
// The caller takes ownership of all the non-null TF_Tensor* entries in `values`
// (which can be deleted using TF_DeleteTensor(values[i])).
TF_CAPI_EXPORT extern void TF_OperationGetAttrTensorList(TF_Operation* oper,
                                                         const char* attr_name,
                                                         TF_Tensor** values,
                                                         int max_values,
                                                         TF_Status* status);

// Sets `output_attr_value` to the binary-serialized AttrValue proto
// representation of the value of the `attr_name` attr of `oper`.
TF_CAPI_EXPORT extern void TF_OperationGetAttrValueProto(
    TF_Operation* oper, const char* attr_name, TF_Buffer* output_attr_value,
    TF_Status* status);

// Returns the operation in the graph with `oper_name`. Returns nullptr if
// no operation found.
TF_CAPI_EXPORT extern TF_Operation* TF_GraphOperationByName(
    TF_Graph* graph, const char* oper_name);

// Iterate through the operations of a graph.  To use:
// size_t pos = 0;
// TF_Operation* oper;
// while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
//   DoSomethingWithOperation(oper);
// }
TF_CAPI_EXPORT extern TF_Operation* TF_GraphNextOperation(TF_Graph* graph,
                                                          size_t* pos);

// Write out a serialized representation of `graph` (as a GraphDef protocol
// message) to `output_graph_def` (allocated by TF_NewBuffer()).
// `output_graph_def`'s underlying buffer will be freed when TF_DeleteBuffer()
// is called.
//
// May fail on very large graphs in the future.
TF_CAPI_EXPORT extern void TF_GraphToGraphDef(TF_Graph* graph,
                                              TF_Buffer* output_graph_def,
                                              TF_Status* status);

// TF_ImportGraphDefOptions holds options that can be passed to
// TF_GraphImportGraphDef.
typedef struct TF_ImportGraphDefOptions TF_ImportGraphDefOptions;

TF_CAPI_EXPORT extern TF_ImportGraphDefOptions* TF_NewImportGraphDefOptions();
TF_CAPI_EXPORT extern void TF_DeleteImportGraphDefOptions(
    TF_ImportGraphDefOptions* opts);

// Set the prefix to be prepended to the names of nodes in `graph_def` that will
// be imported into `graph`.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsSetPrefix(
    TF_ImportGraphDefOptions* opts, const char* prefix);

// Set any imported nodes with input `src_name:src_index` to have that input
// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
// `dst` references a node already existing in the graph being imported into.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsAddInputMapping(
    TF_ImportGraphDefOptions* opts, const char* src_name, int src_index,
    TF_Output dst);

// Set any imported nodes with control input `src_name` to have that input
// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
// `dst` references an operation already existing in the graph being imported
// into.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsRemapControlDependency(
    TF_ImportGraphDefOptions* opts, const char* src_name, TF_Operation* dst);

// Cause the imported graph to have a control dependency on `oper`. `oper`
// should exist in the graph being imported into.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsAddControlDependency(
    TF_ImportGraphDefOptions* opts, TF_Operation* oper);

// Add an output in `graph_def` to be returned via the `return_outputs` output
// parameter of TF_GraphImportGraphDef(). If the output is remapped via an input
// mapping, the corresponding existing tensor in `graph` will be returned.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsAddReturnOutput(
    TF_ImportGraphDefOptions* opts, const char* oper_name, int index);

// Returns the number of return outputs added via
// TF_ImportGraphDefOptionsAddReturnOutput().
TF_CAPI_EXPORT extern int TF_ImportGraphDefOptionsNumReturnOutputs(
    const TF_ImportGraphDefOptions* opts);

// Import the graph serialized in `graph_def` into `graph`.
//
// `num_return_outputs` must be the number of return outputs added (i.e. the
// result of TF_ImportGraphDefOptionsNumReturnOutputs()).  If
// `num_return_outputs` is non-zero, `return_outputs` must be of length
// `num_return_outputs`. Otherwise it can be null.
TF_CAPI_EXPORT extern void TF_GraphImportGraphDefWithReturnOutputs(
    TF_Graph* graph, const TF_Buffer* graph_def,
    const TF_ImportGraphDefOptions* options, TF_Output* return_outputs,
    int num_return_outputs, TF_Status* status);

// Import the graph serialized in `graph_def` into `graph`.
// Convenience function for when no return outputs have been added.
TF_CAPI_EXPORT extern void TF_GraphImportGraphDef(
    TF_Graph* graph, const TF_Buffer* graph_def,
    const TF_ImportGraphDefOptions* options, TF_Status* status);

// Note: The following function may fail on very large protos in the future.

TF_CAPI_EXPORT extern void TF_OperationToNodeDef(TF_Operation* oper,
                                                 TF_Buffer* output_node_def,
                                                 TF_Status* status);

typedef struct TF_WhileParams {
  // The number of inputs to the while loop, i.e. the number of loop variables.
  // This is the size of cond_inputs, body_inputs, and body_outputs.
  const int ninputs;

  // The while condition graph. The inputs are the current values of the loop
  // variables. The output should be a scalar boolean.
  TF_Graph* const cond_graph;
  const TF_Output* const cond_inputs;
  TF_Output cond_output;

  // The loop body graph. The inputs are the current values of the loop
  // variables. The outputs are the updated values of the loop variables.
  TF_Graph* const body_graph;
  const TF_Output* const body_inputs;
  TF_Output* const body_outputs;

  // Unique null-terminated name for this while loop. This is used as a prefix
  // for created operations.
  const char* name;
} TF_WhileParams;

// Creates a TF_WhileParams for creating a while loop in `g`. `inputs` are
// outputs that already exist in `g` used as initial values for the loop
// variables.
//
// The returned TF_WhileParams will have all fields initialized except
// `cond_output`, `body_outputs`, and `name`. The `body_outputs` buffer will be
// allocated to size `ninputs`. The caller should build `cond_graph` and
// `body_graph` starting from the inputs, and store the final outputs in
// `cond_output` and `body_outputs`.
//
// If `status` is OK, the caller must call either TF_FinishWhile or
// TF_AbortWhile on the returned TF_WhileParams. If `status` isn't OK, the
// returned TF_WhileParams is not valid, and the caller should not call
// TF_FinishWhile() or TF_AbortWhile().
//
// Missing functionality (TODO):
// - Gradients
// - Reference-type inputs
// - Directly referencing external tensors from the cond/body graphs (this is
//   possible in the Python API)
TF_CAPI_EXPORT extern TF_WhileParams TF_NewWhile(TF_Graph* g, TF_Output* inputs,
                                                 int ninputs,
                                                 TF_Status* status);

// Builds the while loop specified by `params` and returns the output tensors of
// the while loop in `outputs`. `outputs` should be allocated to size
// `params.ninputs`.
//
// `params` is no longer valid once this returns.
//
// Either this or TF_AbortWhile() must be called after a successful
// TF_NewWhile() call.
TF_CAPI_EXPORT extern void TF_FinishWhile(const TF_WhileParams* params,
                                          TF_Status* status,
                                          TF_Output* outputs);

// Frees `params`s resources without building a while loop. `params` is no
// longer valid after this returns. Either this or TF_FinishWhile() must be
// called after a successful TF_NewWhile() call.
TF_CAPI_EXPORT extern void TF_AbortWhile(const TF_WhileParams* params);

// Adds operations to compute the partial derivatives of sum of `y`s w.r.t `x`s,
// i.e., d(y_1 + y_2 + ...)/dx_1, d(y_1 + y_2 + ...)/dx_2...
// `dx` are used as initial gradients (which represent the symbolic partial
// derivatives of some loss function `L` w.r.t. `y`).
// `dx` must be nullptr or have size `ny`.
// If `dx` is nullptr, the implementation will use dx of `OnesLike` for all
// shapes in `y`.
// The partial derivatives are returned in `dy`. `dy` should be allocated to
// size `nx`.
//
// WARNING: This function does not yet support all the gradients that python
// supports. See
// https://www.tensorflow.org/code/tensorflow/cc/gradients/README.md
// for instructions on how to add C++ more gradients.
TF_CAPI_EXPORT void TF_AddGradients(TF_Graph* g, TF_Output* y, int ny,
                                    TF_Output* x, int nx, TF_Output* dx,
                                    TF_Status* status, TF_Output* dy);

// TODO(josh11b): Register OpDef, available to all operations added
// to this graph.

// The following two may both benefit from a subgraph-definition API
// that re-uses most of the graph-definition API.
// TODO(andydavis): Add functions to a graph.

// --------------------------------------------------------------------------
// API for driving Graph execution.

typedef struct TF_Session TF_Session;

// Return a new execution session with the associated graph, or NULL on error.
//
// *graph must be a valid graph (not deleted or nullptr).  This function will
// prevent the graph from being deleted until TF_DeleteSession() is called.
// Does not take ownership of opts.
TF_CAPI_EXPORT extern TF_Session* TF_NewSession(TF_Graph* graph,
                                                const TF_SessionOptions* opts,
                                                TF_Status* status);

// This function creates a new TF_Session (which is created on success) using
// `session_options`, and then initializes state (restoring tensors and other
// assets) using `run_options`.
//
// Any NULL and non-NULL value combinations for (`run_options, `meta_graph_def`)
// are valid.
//
// - `export_dir` must be set to the path of the exported SavedModel.
// - `tags` must include the set of tags used to identify one MetaGraphDef in
//    the SavedModel.
// - `graph` must be a graph newly allocated with TF_NewGraph().
//
// If successful, populates `graph` with the contents of the Graph and
// `meta_graph_def` with the MetaGraphDef of the loaded model.
TF_CAPI_EXPORT extern TF_Session* TF_LoadSessionFromSavedModel(
    const TF_SessionOptions* session_options, const TF_Buffer* run_options,
    const char* export_dir, const char* const* tags, int tags_len,
    TF_Graph* graph, TF_Buffer* meta_graph_def, TF_Status* status);

// Close a session.
//
// Contacts any other processes associated with the session, if applicable.
// May not be called after TF_DeleteSession().
TF_CAPI_EXPORT extern void TF_CloseSession(TF_Session*, TF_Status* status);

// Destroy a session object.
//
// Even if error information is recorded in *status, this call discards all
// local resources associated with the session.  The session may not be used
// during or after this call (and the session drops its reference to the
// corresponding graph).
TF_CAPI_EXPORT extern void TF_DeleteSession(TF_Session*, TF_Status* status);

// Run the graph associated with the session starting with the supplied inputs
// (inputs[0,ninputs-1] with corresponding values in input_values[0,ninputs-1]).
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
// The caller retains ownership of `input_values` (which can be deleted using
// TF_DeleteTensor). The caller also retains ownership of `run_options` and/or
// `run_metadata` (when not NULL) and should manually call TF_DeleteBuffer on
// them.
//
// On success, the tensors corresponding to outputs[0,noutputs-1] are placed in
// output_values[]. Ownership of the elements of output_values[] is transferred
// to the caller, which must eventually call TF_DeleteTensor on them.
//
// On failure, output_values[] contains NULLs.
TF_CAPI_EXPORT extern void TF_SessionRun(
    TF_Session* session,
    // RunOptions
    const TF_Buffer* run_options,
    // Input tensors
    const TF_Output* inputs, TF_Tensor* const* input_values, int ninputs,
    // Output tensors
    const TF_Output* outputs, TF_Tensor** output_values, int noutputs,
    // Target operations
    const TF_Operation* const* target_opers, int ntargets,
    // RunMetadata
    TF_Buffer* run_metadata,
    // Output status
    TF_Status*);

// Set up the graph with the intended feeds (inputs) and fetches (outputs) for a
// sequence of partial run calls.
//
// On success, returns a handle that is used for subsequent PRun calls. The
// handle should be deleted with TF_DeletePRunHandle when it is no longer
// needed.
//
// On failure, out_status contains a tensorflow::Status with an error
// message. *handle is set to nullptr.
TF_CAPI_EXPORT extern void TF_SessionPRunSetup(
    TF_Session*,
    // Input names
    const TF_Output* inputs, int ninputs,
    // Output names
    const TF_Output* outputs, int noutputs,
    // Target operations
    const TF_Operation* const* target_opers, int ntargets,
    // Output handle
    const char** handle,
    // Output status
    TF_Status*);

// Continue to run the graph with additional feeds and fetches. The
// execution state is uniquely identified by the handle.
TF_CAPI_EXPORT extern void TF_SessionPRun(
    TF_Session*, const char* handle,
    // Input tensors
    const TF_Output* inputs, TF_Tensor* const* input_values, int ninputs,
    // Output tensors
    const TF_Output* outputs, TF_Tensor** output_values, int noutputs,
    // Target operations
    const TF_Operation* const* target_opers, int ntargets,
    // Output status
    TF_Status*);

// Deletes a handle allocated by TF_SessionPRunSetup.
// Once called, no more calls to TF_SessionPRun should be made.
TF_CAPI_EXPORT extern void TF_DeletePRunHandle(const char* handle);

// --------------------------------------------------------------------------
// The deprecated session API.  Please switch to the above instead of
// TF_ExtendGraph(). This deprecated API can be removed at any time without
// notice.

typedef struct TF_DeprecatedSession TF_DeprecatedSession;

TF_CAPI_EXPORT extern TF_DeprecatedSession* TF_NewDeprecatedSession(
    const TF_SessionOptions*, TF_Status* status);
TF_CAPI_EXPORT extern void TF_CloseDeprecatedSession(TF_DeprecatedSession*,
                                                     TF_Status* status);
TF_CAPI_EXPORT extern void TF_DeleteDeprecatedSession(TF_DeprecatedSession*,
                                                      TF_Status* status);
TF_CAPI_EXPORT extern void TF_Reset(const TF_SessionOptions* opt,
                                    const char** containers, int ncontainers,
                                    TF_Status* status);
// Treat the bytes proto[0,proto_len-1] as a serialized GraphDef and
// add the nodes in that GraphDef to the graph for the session.
//
// Prefer use of TF_Session and TF_GraphImportGraphDef over this.
TF_CAPI_EXPORT extern void TF_ExtendGraph(TF_DeprecatedSession*,
                                          const void* proto, size_t proto_len,
                                          TF_Status*);

// See TF_SessionRun() above.
TF_CAPI_EXPORT extern void TF_Run(TF_DeprecatedSession*,
                                  const TF_Buffer* run_options,
                                  const char** input_names, TF_Tensor** inputs,
                                  int ninputs, const char** output_names,
                                  TF_Tensor** outputs, int noutputs,
                                  const char** target_oper_names, int ntargets,
                                  TF_Buffer* run_metadata, TF_Status*);

// See TF_SessionPRunSetup() above.
TF_CAPI_EXPORT extern void TF_PRunSetup(TF_DeprecatedSession*,
                                        const char** input_names, int ninputs,
                                        const char** output_names, int noutputs,
                                        const char** target_oper_names,
                                        int ntargets, const char** handle,
                                        TF_Status*);

// See TF_SessionPRun above.
TF_CAPI_EXPORT extern void TF_PRun(TF_DeprecatedSession*, const char* handle,
                                   const char** input_names, TF_Tensor** inputs,
                                   int ninputs, const char** output_names,
                                   TF_Tensor** outputs, int noutputs,
                                   const char** target_oper_names, int ntargets,
                                   TF_Status*);

typedef struct TF_DeviceList TF_DeviceList;

// Lists all devices in a TF_Session.
//
// Caller takes ownership of the returned TF_DeviceList* which must eventually
// be freed with a call to TF_DeleteDeviceList.
TF_CAPI_EXPORT extern TF_DeviceList* TF_SessionListDevices(TF_Session* session,
                                                           TF_Status* status);

// Lists all devices in a TF_Session.
//
// Caller takes ownership of the returned TF_DeviceList* which must eventually
// be freed with a call to TF_DeleteDeviceList.
TF_CAPI_EXPORT extern TF_DeviceList* TF_DeprecatedSessionListDevices(
    TF_DeprecatedSession* session, TF_Status* status);

// Deallocates the device list.
TF_CAPI_EXPORT extern void TF_DeleteDeviceList(TF_DeviceList* list);

// Counts the number of elements in the device list.
TF_CAPI_EXPORT extern int TF_DeviceListCount(const TF_DeviceList* list);

// Retrieves the full name of the device (e.g. /job:worker/replica:0/...)
// The return value will be a pointer to a null terminated string. The caller
// must not modify or delete the string. It will be deallocated upon a call to
// TF_DeleteDeviceList.
//
// If index is out of bounds, an error code will be set in the status object,
// and a null pointer will be returned.
TF_CAPI_EXPORT extern const char* TF_DeviceListName(const TF_DeviceList* list,
                                                    int index, TF_Status*);

// Retrieves the type of the device at the given index.
//
// The caller must not modify or delete the string. It will be deallocated upon
// a call to TF_DeleteDeviceList.
//
// If index is out of bounds, an error code will be set in the status object,
// and a null pointer will be returned.
TF_CAPI_EXPORT extern const char* TF_DeviceListType(const TF_DeviceList* list,
                                                    int index, TF_Status*);

// Retrieve the amount of memory associated with a given device.
//
// If index is out of bounds, an error code will be set in the status object,
// and -1 will be returned.
TF_CAPI_EXPORT extern int64_t TF_DeviceListMemoryBytes(
    const TF_DeviceList* list, int index, TF_Status*);

// --------------------------------------------------------------------------
// Load plugins containing custom ops and kernels

// TF_Library holds information about dynamically loaded TensorFlow plugins.
typedef struct TF_Library TF_Library;

// Load the library specified by library_filename and register the ops and
// kernels present in that library.
//
// Pass "library_filename" to a platform-specific mechanism for dynamically
// loading a library. The rules for determining the exact location of the
// library are platform-specific and are not documented here.
//
// On success, place OK in status and return the newly created library handle.
// The caller owns the library handle.
//
// On failure, place an error status in status and return NULL.
TF_CAPI_EXPORT extern TF_Library* TF_LoadLibrary(const char* library_filename,
                                                 TF_Status* status);

// Get the OpList of OpDefs defined in the library pointed by lib_handle.
//
// Returns a TF_Buffer. The memory pointed to by the result is owned by
// lib_handle. The data in the buffer will be the serialized OpList proto for
// ops defined in the library.
TF_CAPI_EXPORT extern TF_Buffer TF_GetOpList(TF_Library* lib_handle);

// Frees the memory associated with the library handle.
// Does NOT unload the library.
TF_CAPI_EXPORT extern void TF_DeleteLibraryHandle(TF_Library* lib_handle);

// Get the OpList of all OpDefs defined in this address space.
// Returns a TF_Buffer, ownership of which is transferred to the caller
// (and can be freed using TF_DeleteBuffer).
//
// The data in the buffer will be the serialized OpList proto for ops registered
// in this address space.
TF_CAPI_EXPORT extern TF_Buffer* TF_GetAllOpList();

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_C_API_H_
