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

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/tf_attrtype.h"
#include "tensorflow/c/tf_buffer.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tstring.h"
#include "tensorflow/core/framework/full_type.pb.h"

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
// * Deletion functions are safe to call on nullptr.
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

#ifdef __cplusplus
extern "C" {
#endif

// --------------------------------------------------------------------------
// TF_Version returns a string describing version information of the
// TensorFlow library. TensorFlow uses semantic versioning.
TF_CAPI_EXPORT extern const char* TF_Version(void);

// Parsing a serialized TensorProto into a TF_Tensor.
TF_CAPI_EXPORT extern void TF_TensorFromProto(const TF_Buffer* from,
                                              TF_Tensor* to, TF_Status* status);

// --------------------------------------------------------------------------
// Used to return strings across the C API. The caller does not take ownership
// of the underlying data pointer and is not responsible for freeing it.
typedef struct TF_StringView {
  const char* data;
  size_t len;
} TF_StringView;

// --------------------------------------------------------------------------
// TF_SessionOptions holds options that can be passed during session creation.
typedef struct TF_SessionOptions TF_SessionOptions;

// Return a new options object.
TF_CAPI_EXPORT extern TF_SessionOptions* TF_NewSessionOptions(void);

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
TF_CAPI_EXPORT extern TF_Graph* TF_NewGraph(void);

// Destroy an options object. Graph will be deleted once no more
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

// TF_Function is a grouping of operations with defined inputs and outputs.
// Once created and added to graphs, functions can be invoked by creating an
// operation whose operation type matches the function name.
typedef struct TF_Function TF_Function;

// Function definition options. TODO(iga): Define and implement
typedef struct TF_FunctionOptions TF_FunctionOptions;

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

// Creates a new operation - see `TF_NewOperation` for more details.
//
// The lock for `graph` must be held when calling this function.
//
// Unless implementing advanced behavior, like custom gradient functions, you
// most likely need to call `TF_NewOperation` instead.
TF_CAPI_EXPORT extern TF_OperationDescription* TF_NewOperationLocked(
    TF_Graph* graph, const char* op_type, const char* oper_name);

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
TF_CAPI_EXPORT extern void TF_SetAttrPlaceholder(TF_OperationDescription* desc,
                                                 const char* attr_name,
                                                 const char* placeholder);

// Set a 'func' attribute to the specified name.
// `value` must point to a string of length `length` bytes.
TF_CAPI_EXPORT extern void TF_SetAttrFuncName(TF_OperationDescription* desc,
                                              const char* attr_name,
                                              const char* value, size_t length);

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

// Adds this operation to the graph - see `TF_FinishOperation` for more details.
//
// The lock for `graph` must be held when calling this function.
//
// Unless implementing advanced behavior, like custom gradient functions, you
// most likely need to call `TF_FinishOperation` instead.
TF_CAPI_EXPORT extern TF_Operation* TF_FinishOperationLocked(
    TF_OperationDescription* desc, TF_Status* status);

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

// Get list of all inputs of a specific operation.  `inputs` must point to
// an array of length at least `max_inputs` (ideally set to
// TF_OperationNumInputs(oper)).  Beware that a concurrent
// modification of the graph can increase the number of inputs of
// an operation.
TF_CAPI_EXPORT extern void TF_OperationAllInputs(TF_Operation* oper,
                                                 TF_Output* inputs,
                                                 int max_inputs);

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
// `attr_name` attribute of `oper`.
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

// Get the number of attributes the operation has.
TF_CAPI_EXPORT extern int TF_OperationGetNumAttrs(TF_Operation* oper);

// Get the length of the name of the ith attribute, or -1 if there is not an
// ith attribute.
TF_CAPI_EXPORT extern int TF_OperationGetAttrNameLength(TF_Operation* oper,
                                                        int i);

// Get the name of the ith attribute.  output should have the size of
// TF_OperationGetAttrNameLength(oper, i).
TF_CAPI_EXPORT extern void TF_OperationGetAttrName(TF_Operation* oper, int i,
                                                   char* output,
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

// Returns the serialized OpDef proto with name `op_name`, or a bad status if no
// such op exists. This can return OpDefs of functions copied into the graph.
TF_CAPI_EXPORT extern void TF_GraphGetOpDef(TF_Graph* graph,
                                            const char* op_name,
                                            TF_Buffer* output_op_def,
                                            TF_Status* status);

// Returns the serialized VersionDef proto for this graph.
TF_CAPI_EXPORT extern void TF_GraphVersions(TF_Graph* graph,
                                            TF_Buffer* output_version_def,
                                            TF_Status* status);

// TF_ImportGraphDefOptions holds options that can be passed to
// TF_GraphImportGraphDef.
typedef struct TF_ImportGraphDefOptions TF_ImportGraphDefOptions;

TF_CAPI_EXPORT extern TF_ImportGraphDefOptions* TF_NewImportGraphDefOptions(
    void);
TF_CAPI_EXPORT extern void TF_DeleteImportGraphDefOptions(
    TF_ImportGraphDefOptions* opts);

// Set the prefix to be prepended to the names of nodes in `graph_def` that will
// be imported into `graph`. `prefix` is copied and has no lifetime
// requirements.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsSetPrefix(
    TF_ImportGraphDefOptions* opts, const char* prefix);

// Set the execution device for nodes in `graph_def`.
// Only applies to nodes where a device was not already explicitly specified.
// `device` is copied and has no lifetime requirements.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsSetDefaultDevice(
    TF_ImportGraphDefOptions* opts, const char* device);

// Set whether to uniquify imported operation names. If true, imported operation
// names will be modified if their name already exists in the graph. If false,
// conflicting names will be treated as an error. Note that this option has no
// effect if a prefix is set, since the prefix will guarantee all names are
// unique. Defaults to false.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsSetUniquifyNames(
    TF_ImportGraphDefOptions* opts, unsigned char uniquify_names);

// If true, the specified prefix will be modified if it already exists as an
// operation name or prefix in the graph. If false, a conflicting prefix will be
// treated as an error. This option has no effect if no prefix is specified.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsSetUniquifyPrefix(
    TF_ImportGraphDefOptions* opts, unsigned char uniquify_prefix);

// Set any imported nodes with input `src_name:src_index` to have that input
// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
// `dst` references a node already existing in the graph being imported into.
// `src_name` is copied and has no lifetime requirements.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsAddInputMapping(
    TF_ImportGraphDefOptions* opts, const char* src_name, int src_index,
    TF_Output dst);

// Set any imported nodes with control input `src_name` to have that input
// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
// `dst` references an operation already existing in the graph being imported
// into. `src_name` is copied and has no lifetime requirements.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsRemapControlDependency(
    TF_ImportGraphDefOptions* opts, const char* src_name, TF_Operation* dst);

// Cause the imported graph to have a control dependency on `oper`. `oper`
// should exist in the graph being imported into.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsAddControlDependency(
    TF_ImportGraphDefOptions* opts, TF_Operation* oper);

// Add an output in `graph_def` to be returned via the `return_outputs` output
// parameter of TF_GraphImportGraphDef(). If the output is remapped via an input
// mapping, the corresponding existing tensor in `graph` will be returned.
// `oper_name` is copied and has no lifetime requirements.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsAddReturnOutput(
    TF_ImportGraphDefOptions* opts, const char* oper_name, int index);

// Returns the number of return outputs added via
// TF_ImportGraphDefOptionsAddReturnOutput().
TF_CAPI_EXPORT extern int TF_ImportGraphDefOptionsNumReturnOutputs(
    const TF_ImportGraphDefOptions* opts);

// Add an operation in `graph_def` to be returned via the `return_opers` output
// parameter of TF_GraphImportGraphDef(). `oper_name` is copied and has no
// lifetime requirements.
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsAddReturnOperation(
    TF_ImportGraphDefOptions* opts, const char* oper_name);

// Returns the number of return operations added via
// TF_ImportGraphDefOptionsAddReturnOperation().
TF_CAPI_EXPORT extern int TF_ImportGraphDefOptionsNumReturnOperations(
    const TF_ImportGraphDefOptions* opts);

// TF_ImportGraphDefResults holds results that are generated by
// TF_GraphImportGraphDefWithResults().
typedef struct TF_ImportGraphDefResults TF_ImportGraphDefResults;

// Fetches the return outputs requested via
// TF_ImportGraphDefOptionsAddReturnOutput(). The number of fetched outputs is
// returned in `num_outputs`. The array of return outputs is returned in
// `outputs`. `*outputs` is owned by and has the lifetime of `results`.
TF_CAPI_EXPORT extern void TF_ImportGraphDefResultsReturnOutputs(
    TF_ImportGraphDefResults* results, int* num_outputs, TF_Output** outputs);

// Fetches the return operations requested via
// TF_ImportGraphDefOptionsAddReturnOperation(). The number of fetched
// operations is returned in `num_opers`. The array of return operations is
// returned in `opers`. `*opers` is owned by and has the lifetime of `results`.
TF_CAPI_EXPORT extern void TF_ImportGraphDefResultsReturnOperations(
    TF_ImportGraphDefResults* results, int* num_opers, TF_Operation*** opers);

// Fetches any input mappings requested via
// TF_ImportGraphDefOptionsAddInputMapping() that didn't appear in the GraphDef
// and weren't used as input to any node in the imported graph def. The number
// of fetched mappings is returned in `num_missing_unused_input_mappings`. The
// array of each mapping's source node name is returned in `src_names`, and the
// array of each mapping's source index is returned in `src_indexes`.
//
// `*src_names`, `*src_indexes`, and the memory backing each string in
// `src_names` are owned by and have the lifetime of `results`.
TF_CAPI_EXPORT extern void TF_ImportGraphDefResultsMissingUnusedInputMappings(
    TF_ImportGraphDefResults* results, int* num_missing_unused_input_mappings,
    const char*** src_names, int** src_indexes);

// Deletes a results object returned by TF_GraphImportGraphDefWithResults().
TF_CAPI_EXPORT extern void TF_DeleteImportGraphDefResults(
    TF_ImportGraphDefResults* results);

// Import the graph serialized in `graph_def` into `graph`.  Returns nullptr and
// a bad status on error. Otherwise, returns a populated
// TF_ImportGraphDefResults instance. The returned instance must be deleted via
// TF_DeleteImportGraphDefResults().
TF_CAPI_EXPORT extern TF_ImportGraphDefResults*
TF_GraphImportGraphDefWithResults(TF_Graph* graph, const TF_Buffer* graph_def,
                                  const TF_ImportGraphDefOptions* options,
                                  TF_Status* status);

// Import the graph serialized in `graph_def` into `graph`.
// Convenience function for when only return outputs are needed.
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
// Convenience function for when no results are needed.
TF_CAPI_EXPORT extern void TF_GraphImportGraphDef(
    TF_Graph* graph, const TF_Buffer* graph_def,
    const TF_ImportGraphDefOptions* options, TF_Status* status);

// Adds a copy of function `func` and optionally its gradient function `grad`
// to `g`. Once `func`/`grad` is added to `g`, it can be called by creating
// an operation using the function's name.
// Any changes to `func`/`grad` (including deleting it) done after this method
// returns, won't affect the copy of `func`/`grad` in `g`.
// If `func` or `grad` are already in `g`, TF_GraphCopyFunction has no
// effect on them, but can establish the function->gradient relationship
// between them if `func` does not already have a gradient. If `func` already
// has a gradient different from `grad`, an error is returned.
//
// `func` must not be null.
// If `grad` is null and `func` is not in `g`, `func` is added without a
// gradient.
// If `grad` is null and `func` is in `g`, TF_GraphCopyFunction is a noop.
// `grad` must have appropriate signature as described in the doc of
// GradientDef in tensorflow/core/framework/function.proto.
//
// If successful, status is set to OK and `func` and `grad` are added to `g`.
// Otherwise, status is set to the encountered error and `g` is unmodified.
TF_CAPI_EXPORT extern void TF_GraphCopyFunction(TF_Graph* g,
                                                const TF_Function* func,
                                                const TF_Function* grad,
                                                TF_Status* status);

// Returns the number of TF_Functions registered in `g`.
TF_CAPI_EXPORT extern int TF_GraphNumFunctions(TF_Graph* g);

// Fills in `funcs` with the TF_Function* registered in `g`.
// `funcs` must point to an array of TF_Function* of length at least
// `max_func`. In usual usage, max_func should be set to the result of
// TF_GraphNumFunctions(g). In this case, all the functions registered in
// `g` will be returned. Else, an unspecified subset.
//
// If successful, returns the number of TF_Function* successfully set in
// `funcs` and sets status to OK. The caller takes ownership of
// all the returned TF_Functions. They must be deleted with TF_DeleteFunction.
// On error, returns 0, sets status to the encountered error, and the contents
// of funcs will be undefined.
TF_CAPI_EXPORT extern int TF_GraphGetFunctions(TF_Graph* g, TF_Function** funcs,
                                               int max_func, TF_Status* status);

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
//
// `dx` are used as initial gradients (which represent the symbolic partial
// derivatives of some loss function `L` w.r.t. `y`).
// `dx` must be nullptr or have size `ny`.
// If `dx` is nullptr, the implementation will use dx of `OnesLike` for all
// shapes in `y`.
// The partial derivatives are returned in `dy`. `dy` should be allocated to
// size `nx`.
//
// Gradient nodes are automatically named under the "gradients/" prefix. To
// guarantee name uniqueness, subsequent calls to the same graph will
// append an incremental tag to the prefix: "gradients_1/", "gradients_2/", ...
// See TF_AddGradientsWithPrefix, which provides a means to specify a custom
// name prefix for operations added to a graph to compute the gradients.
//
// WARNING: This function does not yet support all the gradients that python
// supports. See
// https://www.tensorflow.org/code/tensorflow/cc/gradients/README.md
// for instructions on how to add C++ more gradients.
TF_CAPI_EXPORT void TF_AddGradients(TF_Graph* g, TF_Output* y, int ny,
                                    TF_Output* x, int nx, TF_Output* dx,
                                    TF_Status* status, TF_Output* dy);

// Adds operations to compute the partial derivatives of sum of `y`s w.r.t `x`s,
// i.e., d(y_1 + y_2 + ...)/dx_1, d(y_1 + y_2 + ...)/dx_2...
// This is a variant of TF_AddGradients that allows to caller to pass a custom
// name prefix to the operations added to a graph to compute the gradients.
//
// `dx` are used as initial gradients (which represent the symbolic partial
// derivatives of some loss function `L` w.r.t. `y`).
// `dx` must be nullptr or have size `ny`.
// If `dx` is nullptr, the implementation will use dx of `OnesLike` for all
// shapes in `y`.
// The partial derivatives are returned in `dy`. `dy` should be allocated to
// size `nx`.
// `prefix` names the scope into which all gradients operations are being added.
// `prefix` must be unique within the provided graph otherwise this operation
// will fail. If `prefix` is nullptr, the default prefixing behaviour takes
// place, see TF_AddGradients for more details.
//
// WARNING: This function does not yet support all the gradients that python
// supports. See
// https://www.tensorflow.org/code/tensorflow/cc/gradients/README.md
// for instructions on how to add C++ more gradients.
TF_CAPI_EXPORT void TF_AddGradientsWithPrefix(TF_Graph* g, const char* prefix,
                                              TF_Output* y, int ny,
                                              TF_Output* x, int nx,
                                              TF_Output* dx, TF_Status* status,
                                              TF_Output* dy);

// Create a TF_Function from a TF_Graph
//
// Params:
//  fn_body - the graph whose operations (or subset of whose operations) will be
//            converted to TF_Function.
//  fn_name - the name of the new TF_Function. Should match the operation
//            name (OpDef.name) regexp [A-Z][A-Za-z0-9_.\\-/]*.
//            If `append_hash_to_fn_name` is false, `fn_name` must be distinct
//            from other function and operation names (at least those
//            registered in graphs where this function will be used).
//  append_hash_to_fn_name - Must be 0 or 1. If set to 1, the actual name
//                           of the function will be `fn_name` appended with
//                           '_<hash_of_this_function's_definition>'.
//                           If set to 0, the function's name will be `fn_name`.
//  num_opers - `num_opers` contains the number of elements in the `opers` array
//              or a special value of -1 meaning that no array is given.
//              The distinction between an empty array of operations and no
//              array of operations is necessary to distinguish the case of
//              creating a function with no body (e.g. identity or permutation)
//              and the case of creating a function whose body contains all
//              the nodes in the graph (except for the automatic skipping, see
//              below).
//  opers - Array of operations to become the body of the function or null.
//          - If no array is given (`num_opers` = -1), all the
//          operations in `fn_body` will become part of the function
//          except operations referenced in `inputs`. These operations
//          must have a single output (these operations are typically
//          placeholders created for the sole purpose of representing
//          an input. We can relax this constraint if there are
//          compelling use cases).
//          - If an array is given (`num_opers` >= 0), all operations
//          in it will become part of the function. In particular, no
//          automatic skipping of dummy input operations is performed.
//  ninputs - number of elements in `inputs` array
//  inputs - array of TF_Outputs that specify the inputs to the function.
//           If `ninputs` is zero (the function takes no inputs), `inputs`
//           can be null. The names used for function inputs are normalized
//           names of the operations (usually placeholders) pointed to by
//           `inputs`. These operation names should start with a letter.
//           Normalization will convert all letters to lowercase and
//           non-alphanumeric characters to '_' to make resulting names match
//           the "[a-z][a-z0-9_]*" pattern for operation argument names.
//           `inputs` cannot contain the same tensor twice.
//  noutputs - number of elements in `outputs` array
//  outputs - array of TF_Outputs that specify the outputs of the function.
//            If `noutputs` is zero (the function returns no outputs), `outputs`
//            can be null. `outputs` can contain the same tensor more than once.
//  output_names - The names of the function's outputs. `output_names` array
//                 must either have the same length as `outputs`
//                 (i.e. `noutputs`) or be null. In the former case,
//                 the names should match the regular expression for ArgDef
//                 names - "[a-z][a-z0-9_]*". In the latter case,
//                 names for outputs will be generated automatically.
//  opts - various options for the function, e.g. XLA's inlining control.
//  description - optional human-readable description of this function.
//  status - Set to OK on success and an appropriate error on failure.
//
// Note that when the same TF_Output is listed as both an input and an output,
// the corresponding function's output will equal to this input,
// instead of the original node's output.
//
// Callers must also satisfy the following constraints:
// - `inputs` cannot refer to TF_Outputs within a control flow context. For
//   example, one cannot use the output of "switch" node as input.
// - `inputs` and `outputs` cannot have reference types. Reference types are
//   not exposed through C API and are being replaced with Resources. We support
//   reference types inside function's body to support legacy code. Do not
//   use them in new code.
// - Every node in the function's body must have all of its inputs (including
//   control inputs). In other words, for every node in the body, each input
//   must be either listed in `inputs` or must come from another node in
//   the body. In particular, it is an error to have a control edge going from
//   a node outside of the body into a node in the body. This applies to control
//   edges going from nodes referenced in `inputs` to nodes in the body when
//   the former nodes are not in the body (automatically skipped or not
//   included in explicitly specified body).
//
// Returns:
//  On success, a newly created TF_Function instance. It must be deleted by
//  calling TF_DeleteFunction.
//
//  On failure, null.
TF_CAPI_EXPORT extern TF_Function* TF_GraphToFunction(
    const TF_Graph* fn_body, const char* fn_name,
    unsigned char append_hash_to_fn_name, int num_opers,
    const TF_Operation* const* opers, int ninputs, const TF_Output* inputs,
    int noutputs, const TF_Output* outputs, const char* const* output_names,
    const TF_FunctionOptions* opts, const char* description, TF_Status* status);

// Similar to TF_GraphToFunction but allows specifying control outputs of the
// function.
//
//  The arguments of TF_GraphToFunction have the same meaning, but the new
//  arguments are as follows:
//
//    ncontrol_outputs: Number of control outputs of the function.
//    control_outputs: vector of TF_Operation objects to be marked as control
//      outputs of the function. Operations marked as control outputs are
//      guaranteed to execute.
//    control_output_names: Optional. If not nullptr, vector of strings, one
//      per control output, with their names to be added to the function's
//      OpDef.
TF_CAPI_EXPORT extern TF_Function* TF_GraphToFunctionWithControlOutputs(
    const TF_Graph* fn_body, const char* fn_name,
    unsigned char append_hash_to_fn_name, int num_opers,
    const TF_Operation* const* opers, int ninputs, const TF_Output* inputs,
    int noutputs, const TF_Output* outputs, const char* const* output_names,
    int ncontrol_outputs, const TF_Operation* const* control_outputs,
    const char* const* control_output_names, const TF_FunctionOptions* opts,
    const char* description, TF_Status* status);

// Returns the name of the graph function.
// The return value points to memory that is only usable until the next
// mutation to *func.
TF_CAPI_EXPORT extern const char* TF_FunctionName(TF_Function* func);

// Write out a serialized representation of `func` (as a FunctionDef protocol
// message) to `output_func_def` (allocated by TF_NewBuffer()).
// `output_func_def`'s underlying buffer will be freed when TF_DeleteBuffer()
// is called.
//
// May fail on very large graphs in the future.
TF_CAPI_EXPORT extern void TF_FunctionToFunctionDef(TF_Function* func,
                                                    TF_Buffer* output_func_def,
                                                    TF_Status* status);

// Construct and return the function whose FunctionDef representation is
// serialized in `proto`. `proto_len` must equal the number of bytes
// pointed to by `proto`.
// Returns:
//  On success, a newly created TF_Function instance. It must be deleted by
//  calling TF_DeleteFunction.
//
//  On failure, null.
TF_CAPI_EXPORT extern TF_Function* TF_FunctionImportFunctionDef(
    const void* proto, size_t proto_len, TF_Status* status);

// Sets function attribute named `attr_name` to value stored in `proto`.
// If this attribute is already set to another value, it is overridden.
// `proto` should point to a sequence of bytes of length `proto_len`
// representing a binary serialization of an AttrValue protocol
// buffer.
TF_CAPI_EXPORT extern void TF_FunctionSetAttrValueProto(TF_Function* func,
                                                        const char* attr_name,
                                                        const void* proto,
                                                        size_t proto_len,
                                                        TF_Status* status);

// Sets `output_attr_value` to the binary-serialized AttrValue proto
// representation of the value of the `attr_name` attr of `func`.
// If `attr_name` attribute is not present, status is set to an error.
TF_CAPI_EXPORT extern void TF_FunctionGetAttrValueProto(
    TF_Function* func, const char* attr_name, TF_Buffer* output_attr_value,
    TF_Status* status);

// Frees the memory used by the `func` struct.
// TF_DeleteFunction is a noop if `func` is null.
// Deleting a function does not remove it from any graphs it was copied to.
TF_CAPI_EXPORT extern void TF_DeleteFunction(TF_Function* func);

// Attempts to evaluate `output`. This will only be possible if `output` doesn't
// depend on any graph inputs (this function is safe to call if this isn't the
// case though).
//
// If the evaluation is successful, this function returns true and `output`s
// value is returned in `result`. Otherwise returns false. An error status is
// returned if something is wrong with the graph or input. Note that this may
// return false even if no error status is set.
TF_CAPI_EXPORT extern unsigned char TF_TryEvaluateConstant(TF_Graph* graph,
                                                           TF_Output output,
                                                           TF_Tensor** result,
                                                           TF_Status* status);

// TODO(josh11b): Register OpDef, available to all operations added
// to this graph.

// --------------------------------------------------------------------------
// API for driving Graph execution.

typedef struct TF_Session TF_Session;

// Return a new execution session with the associated graph, or NULL on
// error. Does not take ownership of any input parameters.
//
// *`graph` must be a valid graph (not deleted or nullptr). `graph` will be
// kept alive for the lifetime of the returned TF_Session. New nodes can still
// be added to `graph` after this call.
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
                                                    int index,
                                                    TF_Status* status);

// Retrieves the type of the device at the given index.
//
// The caller must not modify or delete the string. It will be deallocated upon
// a call to TF_DeleteDeviceList.
//
// If index is out of bounds, an error code will be set in the status object,
// and a null pointer will be returned.
TF_CAPI_EXPORT extern const char* TF_DeviceListType(const TF_DeviceList* list,
                                                    int index,
                                                    TF_Status* status);

// Retrieve the amount of memory associated with a given device.
//
// If index is out of bounds, an error code will be set in the status object,
// and -1 will be returned.
TF_CAPI_EXPORT extern int64_t TF_DeviceListMemoryBytes(
    const TF_DeviceList* list, int index, TF_Status* status);

// Retrieve the incarnation number of a given device.
//
// If index is out of bounds, an error code will be set in the status object,
// and 0 will be returned.
TF_CAPI_EXPORT extern uint64_t TF_DeviceListIncarnation(
    const TF_DeviceList* list, int index, TF_Status* status);

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
TF_CAPI_EXPORT extern TF_Buffer* TF_GetAllOpList(void);

// TF_ApiDefMap encapsulates a collection of API definitions for an operation.
//
// This object maps the name of a TensorFlow operation to a description of the
// API to generate for it, as defined by the ApiDef protocol buffer (
// https://www.tensorflow.org/code/tensorflow/core/framework/api_def.proto)
//
// The ApiDef messages are typically used to generate convenience wrapper
// functions for TensorFlow operations in various language bindings.
typedef struct TF_ApiDefMap TF_ApiDefMap;

// Creates a new TF_ApiDefMap instance.
//
// Params:
//  op_list_buffer - TF_Buffer instance containing serialized OpList
//    protocol buffer. (See
//    https://www.tensorflow.org/code/tensorflow/core/framework/op_def.proto
//    for the OpList proto definition).
//  status - Set to OK on success and an appropriate error on failure.
TF_CAPI_EXPORT extern TF_ApiDefMap* TF_NewApiDefMap(TF_Buffer* op_list_buffer,
                                                    TF_Status* status);

// Deallocates a TF_ApiDefMap.
TF_CAPI_EXPORT extern void TF_DeleteApiDefMap(TF_ApiDefMap* apimap);

// Add ApiDefs to the map.
//
// `text` corresponds to a text representation of an ApiDefs protocol message.
// (https://www.tensorflow.org/code/tensorflow/core/framework/api_def.proto).
//
// The provided ApiDefs will be merged with existing ones in the map, with
// precedence given to the newly added version in case of conflicts with
// previous calls to TF_ApiDefMapPut.
TF_CAPI_EXPORT extern void TF_ApiDefMapPut(TF_ApiDefMap* api_def_map,
                                           const char* text, size_t text_len,
                                           TF_Status* status);

// Returns a serialized ApiDef protocol buffer for the TensorFlow operation
// named `name`.
TF_CAPI_EXPORT extern TF_Buffer* TF_ApiDefMapGet(TF_ApiDefMap* api_def_map,
                                                 const char* name,
                                                 size_t name_len,
                                                 TF_Status* status);

// --------------------------------------------------------------------------
// Kernel definition information.

// Returns a serialized KernelList protocol buffer containing KernelDefs for all
// registered kernels.
TF_CAPI_EXPORT extern TF_Buffer* TF_GetAllRegisteredKernels(TF_Status* status);

// Returns a serialized KernelList protocol buffer containing KernelDefs for all
// kernels registered for the operation named `name`.
TF_CAPI_EXPORT extern TF_Buffer* TF_GetRegisteredKernelsForOp(
    const char* name, TF_Status* status);

// Update edge, switch input/ output in a node
TF_CAPI_EXPORT extern void TF_UpdateEdge(TF_Graph* graph, TF_Output new_src,
                                         TF_Input dst, TF_Status* status);

// --------------------------------------------------------------------------
// In-process TensorFlow server functionality, for use in distributed training.
// A Server instance encapsulates a set of devices and a Session target that
// can participate in distributed training. A server belongs to a cluster
// (specified by a ClusterSpec), and corresponds to a particular task in a
// named job. The server can communicate with any other server in the same
// cluster.

// In-process TensorFlow server.
typedef struct TF_Server TF_Server;

// Creates a new in-process TensorFlow server configured using a serialized
// ServerDef protocol buffer provided via `proto` and `proto_len`.
//
// The server will not serve any requests until TF_ServerStart is invoked.
// The server will stop serving requests once TF_ServerStop or
// TF_DeleteServer is invoked.
TF_CAPI_EXPORT extern TF_Server* TF_NewServer(const void* proto,
                                              size_t proto_len,
                                              TF_Status* status);

// Starts an in-process TensorFlow server.
TF_CAPI_EXPORT extern void TF_ServerStart(TF_Server* server, TF_Status* status);

// Stops an in-process TensorFlow server.
TF_CAPI_EXPORT extern void TF_ServerStop(TF_Server* server, TF_Status* status);

// Blocks until the server has been successfully stopped (via TF_ServerStop or
// TF_ServerClose).
TF_CAPI_EXPORT extern void TF_ServerJoin(TF_Server* server, TF_Status* status);

// Returns the target string that can be provided to TF_SetTarget() to connect
// a TF_Session to `server`.
//
// The returned string is valid only until TF_DeleteServer is invoked.
TF_CAPI_EXPORT extern const char* TF_ServerTarget(TF_Server* server);

// Destroy an in-process TensorFlow server, frees memory. If server is running
// it will be stopped and joined.
TF_CAPI_EXPORT extern void TF_DeleteServer(TF_Server* server);

// Register a listener method that processes printed messages.
//
// If any listeners are registered, the print operator will call all listeners
// with the printed messages and immediately return without writing to the
// logs.
TF_CAPI_EXPORT extern void TF_RegisterLogListener(
    void (*listener)(const char*));

// Register a FileSystem plugin from filename `plugin_filename`.
//
// On success, place OK in status.
// On failure, place an error status in status.
TF_CAPI_EXPORT extern void TF_RegisterFilesystemPlugin(
    const char* plugin_filename, TF_Status* status);

// Apis that are correponding to python c api. --------------------

TF_CAPI_EXPORT extern void TF_AddControlInput(TF_Graph* graph, TF_Operation* op, TF_Operation* input);

TF_CAPI_EXPORT extern void TF_SetAttr(TF_Graph* graph, TF_Operation* op,
                                       const char* attr_name,
                                       TF_Buffer* attr_value_proto,
                                       TF_Status* status);

TF_CAPI_EXPORT extern void TF_ClearAttr(TF_Graph* graph, TF_Operation* op,
                                         const char* attr_name,
                                         TF_Status* status);

TF_CAPI_EXPORT extern void TF_SetFullType(TF_Graph* graph, TF_Operation* op,
                                           const tensorflow::FullTypeDef& full_type);

TF_CAPI_EXPORT extern void TF_SetRequestedDevice(TF_Graph* graph,
                                                  TF_Operation* op,
                                                  const char* device);

TF_CAPI_EXPORT extern void TF_UpdateEdge(TF_Graph* graph, TF_Output new_src,
                                          TF_Input dst, TF_Status* status);

TF_CAPI_EXPORT extern void TF_RemoveAllControlInputs(TF_Graph* graph, TF_Operation* op);

TF_CAPI_EXPORT extern void TF_SetRequireShapeInferenceFns(TF_Graph* graph, bool require);

TF_CAPI_EXPORT extern void TF_ExtendSession(TF_Session* session, TF_Status* status);

TF_CAPI_EXPORT extern const char* TF_GetHandleShapeAndType(TF_Graph* graph, TF_Output output);

TF_CAPI_EXPORT extern void TF_SetHandleShapeAndType(TF_Graph* graph,
                                                     TF_Output output,
                                                     const void* proto,
                                                     size_t proto_len,
                                                     TF_Status* status);

void TFC_AddWhileInputHack(TF_Graph* graph, TF_Output new_src, TF_Operation* dst,
                       TF_Status* status);

// ----------------------------------------------------------------

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_C_API_H_
