/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_CORE_C_COMMON_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_CORE_C_COMMON_H_

#include "absl/log/log.h"
#include "tensorflow/compiler/mlir/lite/core/c/c_api_types.h"  // IWYU pragma: export

// LINT.IfChange

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Check whether the value `a` is true, and if not return
/// absl::InvalidArgumentError from the current function, while also
/// reporting the location of the error.
#define TFL_MIGRATION_ENSURE(a)                                              \
  do {                                                                       \
    if (!(a)) {                                                              \
      auto error_message =                                                   \
          absl::StrFormat("%s:%d %s was not true.", __FILE__, __LINE__, #a); \
      LOG(ERROR) << error_message;                                           \
      return absl::InvalidArgumentError(error_message);                      \
    }                                                                        \
  } while (0)

#define TFL_MIGRATION_ENSURE_STATUS(a) \
  do {                                 \
    const absl::Status s = (a);        \
    if (!s.ok()) {                     \
      return s;                        \
    }                                  \
  } while (0)

/// The list of external context types known to TF Lite. This list exists solely
/// to avoid conflicts and to ensure ops can share the external contexts they
/// need. Access to the external contexts is controlled by one of the
/// corresponding support files.
typedef enum TfLiteExternalContextType {
  kTfLiteEigenContext = 0,       /// include eigen_support.h to use.
  kTfLiteGemmLowpContext = 1,    /// include gemm_support.h to use.
  kTfLiteEdgeTpuContext = 2,     /// Placeholder for Edge TPU support.
  kTfLiteCpuBackendContext = 3,  /// include cpu_backend_context.h to use.
  kTfLiteMaxExternalContexts = 4
} TfLiteExternalContextType;

// Forward declare so dependent structs and methods can reference these types
// prior to the struct definitions.
struct TfLiteContext;
struct TfLiteDelegate;
struct TfLiteRegistration;

/// An external context is a collection of information unrelated to the TF Lite
/// framework, but useful to a subset of the ops. TF Lite knows very little
/// about the actual contexts, but it keeps a list of them, and is able to
/// refresh them if configurations like the number of recommended threads
/// change.
typedef struct TfLiteExternalContext {
  TfLiteExternalContextType type;
  TfLiteStatus (*Refresh)(struct TfLiteContext* context);
} TfLiteExternalContext;

/// Fixed size list of integers. Used for dimensions and inputs/outputs tensor
/// indices
typedef struct TfLiteIntArray {
  int size;

#if defined(_MSC_VER)
  // Context for why this is needed is in http://b/189926408#comment21
  int data[1];
#elif (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
       __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON) ||                                             \
    (defined(__clang__) && __clang_major__ == 7 && __clang_minor__ == 1)
  // gcc 6.1+ have a bug where flexible members aren't properly handled
  // https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
  int data[0];
#else
  int data[];
#endif
} TfLiteIntArray;

/// Single-precision complex data type compatible with the C99 definition.
typedef struct TfLiteComplex64 {
  float re, im;  /// real and imaginary parts, respectively.
} TfLiteComplex64;

/// Double-precision complex data type compatible with the C99 definition.
typedef struct TfLiteComplex128 {
  double re, im;  /// real and imaginary parts, respectively.
} TfLiteComplex128;

/// Half precision data type compatible with the C99 definition.
typedef struct TfLiteFloat16 {
  uint16_t data;
} TfLiteFloat16;

/// Return the name of a given type, for error reporting purposes.
const char* TfLiteTypeGetName(TfLiteType type);

/// SupportedQuantizationTypes.
typedef enum TfLiteQuantizationType {
  /// No quantization.
  kTfLiteNoQuantization = 0,
  /// Affine quantization (with support for per-channel quantization).
  /// Corresponds to TfLiteAffineQuantization.
  kTfLiteAffineQuantization = 1,
} TfLiteQuantizationType;

/// Structure specifying the quantization used by the tensor, if-any.
typedef struct TfLiteQuantization {
  /// The type of quantization held by params.
  TfLiteQuantizationType type;
  /// Holds an optional reference to a quantization param structure. The actual
  /// type depends on the value of the `type` field (see the comment there for
  /// the values and corresponding types).
  void* params;
} TfLiteQuantization;

/// A union of pointers that points to memory for a given tensor.
///
/// Do not access these members directly, if possible, use
/// `GetTensorData<TYPE>(tensor)` instead, otherwise only access `.data`, as
/// other members are deprecated.
typedef union TfLitePtrUnion {
  int32_t* i32;
  uint32_t* u32;
  int64_t* i64;
  uint64_t* u64;
  float* f;
  TfLiteFloat16* f16;
  double* f64;
  char* raw;
  const char* raw_const;
  uint8_t* uint8;
  bool* b;
  int16_t* i16;
  uint16_t* ui16;
  TfLiteComplex64* c64;
  TfLiteComplex128* c128;
  int8_t* int8;
  /// Only use this member.
  void* data;
} TfLitePtrUnion;

/// Memory allocation strategies.
///  * `kTfLiteMmapRo`: Read-only memory-mapped data, or data externally
///        allocated.
///  * `kTfLiteArenaRw`: Arena allocated with no guarantees about persistence,
///        and available during eval.
///  * `kTfLiteArenaRwPersistent`: Arena allocated but persistent across eval,
///  and only available during eval.
///  * `kTfLiteDynamic`: Allocated during eval, or for string tensors.
///  * `kTfLitePersistentRo`: Allocated and populated during prepare. This is
///        useful for tensors that can be computed during prepare and treated
///        as constant inputs for downstream ops (also in prepare).
///  * `kTfLiteCustom`: Custom memory allocation provided by the user. See
///        TfLiteCustomAllocation below.
///  * `kTfLiteVariantObject`: Allocation is an arbitrary type-erased C++
///  object.
///        Allocation and deallocation are done through `new` and `delete`.
typedef enum TfLiteAllocationType {
  kTfLiteMemNone = 0,
  kTfLiteMmapRo,
  kTfLiteArenaRw,
  kTfLiteArenaRwPersistent,
  kTfLiteDynamic,
  kTfLitePersistentRo,
  kTfLiteCustom,
  kTfLiteVariantObject,
} TfLiteAllocationType;

/// The delegates should use zero or positive integers to represent handles.
/// -1 is reserved from unallocated status.
typedef int TfLiteBufferHandle;
enum {
  kTfLiteNullBufferHandle = -1,
};

/// Storage format of each dimension in a sparse tensor.
typedef enum TfLiteDimensionType {
  kTfLiteDimDense = 0,
  kTfLiteDimSparseCSR,
} TfLiteDimensionType;

/// Metadata to encode each dimension in a sparse tensor.
typedef struct TfLiteDimensionMetadata {
  TfLiteDimensionType format;
  int dense_size;
  TfLiteIntArray* array_segments;
  TfLiteIntArray* array_indices;
} TfLiteDimensionMetadata;

/// Parameters used to encode a sparse tensor. For detailed explanation of each
/// field please refer to lite/schema/schema.fbs.
typedef struct TfLiteSparsity {
  TfLiteIntArray* traversal_order;
  TfLiteIntArray* block_map;
  TfLiteDimensionMetadata* dim_metadata;
  int dim_metadata_size;
} TfLiteSparsity;

/// A tensor in the interpreter system which is a wrapper around a buffer of
/// data including a dimensionality (or NULL if not currently defined).
#ifndef TF_LITE_STATIC_MEMORY
typedef struct TfLiteTensor {
  /// The data type specification for data stored in `data`. This affects
  /// what member of `data` union should be used.
  TfLiteType type;
  /// A union of data pointers. The appropriate type should be used for a typed
  /// tensor based on `type`.
  TfLitePtrUnion data;
  /// A pointer to a structure representing the dimensionality interpretation
  /// that the buffer should have. NOTE: the product of elements of `dims`
  /// and the element datatype size should be equal to `bytes` below.
  TfLiteIntArray* dims;
  /// Quantization information.
  TfLiteQuantizationParams params;
  /// How memory is mapped
  ///  kTfLiteMmapRo: Memory mapped read only.
  ///  i.e. weights
  ///  kTfLiteArenaRw: Arena allocated read write memory
  ///  (i.e. temporaries, outputs).
  TfLiteAllocationType allocation_type;
  /// The number of bytes required to store the data of this Tensor. I.e.
  /// (bytes of each element) * dims[0] * ... * dims[n-1].  For example, if
  /// type is kTfLiteFloat32 and dims = {3, 2} then
  /// bytes = sizeof(float) * 3 * 2 = 4 * 3 * 2 = 24.
  size_t bytes;

  /// An opaque pointer to a tflite::MMapAllocation
  const void* allocation;

  /// Null-terminated name of this tensor.
  const char* name;

  /// The delegate which knows how to handle `buffer_handle`.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  struct TfLiteDelegate* delegate;

  /// An integer buffer handle that can be handled by `delegate`.
  /// The value is valid only when delegate is not null.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteBufferHandle buffer_handle;

  /// If the delegate uses its own buffer (e.g. GPU memory), the delegate is
  /// responsible to set data_is_stale to true.
  /// `delegate->CopyFromBufferHandle` can be called to copy the data from
  /// delegate buffer.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  bool data_is_stale;

  /// True if the tensor is a variable.
  bool is_variable;

  /// Quantization information. Replaces params field above.
  TfLiteQuantization quantization;

  /// Parameters used to encode a sparse tensor.
  /// This is optional. The field is NULL if a tensor is dense.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteSparsity* sparsity;

  /// Optional. Encodes shapes with unknown dimensions with -1. This field is
  /// only populated when unknown dimensions exist in a read-write tensor (i.e.
  /// an input or output tensor). (e.g.  `dims` contains [1, 1, 1, 3] and
  /// `dims_signature` contains [1, -1, -1, 3]).  If no unknown dimensions exist
  /// then `dims_signature` is either null, or set to an empty array.  Note that
  /// this field only exists when TF_LITE_STATIC_MEMORY is not defined.
  const TfLiteIntArray* dims_signature;
} TfLiteTensor;

/// A structure representing an instance of a node.
/// This structure only exhibits the inputs, outputs, user defined data and some
/// node properties (like statefulness), not other features like the type.
typedef struct TfLiteNode {
  /// Inputs to this node expressed as indices into the simulator's tensors.
  TfLiteIntArray* inputs;

  /// Outputs to this node expressed as indices into the simulator's tensors.
  TfLiteIntArray* outputs;

  /// intermediate tensors to this node expressed as indices into the
  /// simulator's tensors.
  TfLiteIntArray* intermediates;

  /// Temporary tensors uses during the computations. This usually contains no
  /// tensors, but ops are allowed to change that if they need scratch space of
  /// any sort.
  TfLiteIntArray* temporaries;

  /// Opaque data provided by the node implementer through `Registration.init`.
  void* user_data;

  /// Opaque data provided to the node if the node is a builtin. This is usually
  /// a structure defined in builtin_op_data.h
  void* builtin_data;

  /// Custom initial data. This is the opaque data provided in the flatbuffer.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  const void* custom_initial_data;
  int custom_initial_data_size;

  /// The pointer to the delegate. This is non-null only when the node is
  /// created by calling `interpreter.ModifyGraphWithDelegate`.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  struct TfLiteDelegate* delegate;

  /// Whether this op might have side effect (e.g. stateful op).
  bool might_have_side_effect;
} TfLiteNode;

#else   // defined(TF_LITE_STATIC_MEMORY)?
// NOTE: This flag is opt-in only at compile time.
//
// Specific reduced TfLiteTensor struct for TF Micro runtime. This struct
// contains only the minimum fields required to initialize and prepare a micro
// inference graph. The fields in this struct have been ordered from
// largest-to-smallest for optimal struct sizeof.
//
// This struct does not use:
// - allocation
// - buffer_handle
// - data_is_stale
// - delegate
// - dims_signature
// - name
// - sparsity
typedef struct TfLiteTensor {
  // TODO(b/155784997): Consider consolidating these quantization fields:
  // Quantization information. Replaces params field above.
  TfLiteQuantization quantization;

  // Quantization information.
  TfLiteQuantizationParams params;

  // A union of data pointers. The appropriate type should be used for a typed
  // tensor based on `type`.
  TfLitePtrUnion data;

  // A pointer to a structure representing the dimensionality interpretation
  // that the buffer should have. NOTE: the product of elements of `dims`
  // and the element datatype size should be equal to `bytes` below.
  TfLiteIntArray* dims;

  // The number of bytes required to store the data of this Tensor. I.e.
  // (bytes of each element) * dims[0] * ... * dims[n-1].  For example, if
  // type is kTfLiteFloat32 and dims = {3, 2} then
  // bytes = sizeof(float) * 3 * 2 = 4 * 3 * 2 = 24.
  size_t bytes;

  // The data type specification for data stored in `data`. This affects
  // what member of `data` union should be used.
  TfLiteType type;

  // How memory is mapped
  //  kTfLiteMmapRo: Memory mapped read only.
  //  i.e. weights
  //  kTfLiteArenaRw: Arena allocated read write memory
  //  (i.e. temporaries, outputs).
  TfLiteAllocationType allocation_type;

  // True if the tensor is a variable.
  bool is_variable;
} TfLiteTensor;

// Specific reduced TfLiteNode struct for TF Micro runtime. This struct contains
// only the minimum fields required to represent a node.
//
// This struct does not use:
// - delegate
// - intermediates
// - temporaries
typedef struct TfLiteNode {
  // Inputs to this node expressed as indices into the simulator's tensors.
  TfLiteIntArray* inputs;

  // Outputs to this node expressed as indices into the simulator's tensors.
  TfLiteIntArray* outputs;

  // intermediate tensors to this node expressed as indices into the simulator's
  // tensors.
  TfLiteIntArray* intermediates;

  // Opaque data provided by the node implementer through `Registration.init`.
  void* user_data;

  // Opaque data provided to the node if the node is a builtin. This is usually
  // a structure defined in builtin_op_data.h
  void* builtin_data;

  // Custom initial data. This is the opaque data provided in the flatbuffer.
  //
  // WARNING: This is an experimental interface that is subject to change.
  const void* custom_initial_data;
  int custom_initial_data_size;
} TfLiteNode;
#endif  // TF_LITE_STATIC_MEMORY

/// Light-weight tensor struct for TF Micro runtime. Provides the minimal amount
/// of information required for a kernel to run during TfLiteRegistration::Eval.
// TODO(b/160955687): Move this field into TF_LITE_STATIC_MEMORY when TFLM
// builds with this flag by default internally.
typedef struct TfLiteEvalTensor {
  /// A union of data pointers. The appropriate type should be used for a typed
  /// tensor based on `type`.
  TfLitePtrUnion data;

  /// A pointer to a structure representing the dimensionality interpretation
  /// that the buffer should have.
  TfLiteIntArray* dims;

  /// The data type specification for data stored in `data`. This affects
  /// what member of `data` union should be used.
  TfLiteType type;
} TfLiteEvalTensor;

/// WARNING: This is an experimental interface that is subject to change.
///
/// Currently, TfLiteDelegateParams has to be allocated in a way that it's
/// trivially destructable. It will be stored as `builtin_data` field in
/// `TfLiteNode` of the delegate node.
///
/// See also the `CreateDelegateParams` function in `interpreter.cc` details.
typedef struct TfLiteDelegateParams {
  struct TfLiteDelegate* delegate;
  TfLiteIntArray* nodes_to_replace;
  TfLiteIntArray* input_tensors;
  TfLiteIntArray* output_tensors;
} TfLiteDelegateParams;

/// `TfLiteContext` allows an op to access the tensors.
///
/// `TfLiteContext` is a struct that is created by the TF Lite runtime
/// and passed to the "methods" (C function pointers) in the
/// `TfLiteRegistration` struct that are used to define custom ops and custom
/// delegate kernels. It contains information and methods (C function pointers)
/// that can be called by the code implementing a custom op or a custom delegate
/// kernel. These methods provide access to the context in which that custom op
/// or custom delegate kernel occurs, such as access to the input and output
/// tensors for that op, as well as methods for allocating memory buffers
/// and intermediate tensors, etc.
///
/// See also `TfLiteOpaqueContext`, which is an more ABI-stable equivalent.
typedef struct TfLiteContext {
  /// Number of tensors in the context.
  size_t tensors_size;

  /// The execution plan contains a list of the node indices in execution
  /// order. execution_plan->size is the current number of nodes. And,
  /// execution_plan->data[0] is the first node that needs to be run.
  /// TfLiteDelegates can traverse the current execution plan by iterating
  /// through each member of this array and using GetNodeAndRegistration() to
  /// access details about a node. i.e.
  ///
  ///
  ///     TfLiteIntArray* execution_plan;
  ///     TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context,
  ///                                                     &execution_plan));
  ///     for (int exec_index = 0; exec_index < execution_plan->size;
  ///           exec_index++) {
  ///        int node_index = execution_plan->data[exec_index];
  ///        TfLiteNode* node;
  ///        TfLiteRegistration* reg;
  ///        context->GetNodeAndRegistration(context, node_index, &node, &reg);
  ///     }
  ///
  /// Note: the memory pointed by '`*execution_plan` is OWNED by TfLite runtime.
  /// Future calls to GetExecutionPlan invalidates earlier outputs. The
  /// following code snippet shows the issue of such an invocation pattern.
  /// After calling CheckNode, subsequent access to `plan_1st` is undefined.
  ///
  ///     void CheckNode(const TfLiteNode* node) {
  ///       ...
  ///       TfLiteIntArray* plan_2nd;
  ///       TF_LITE_ENSURE_STATUS(
  ///           context->GetExecutionPlan(context, &plan_2nd)
  ///       );
  ///       ...
  ///     }
  ///
  ///     TfLiteIntArray* plan_1st;
  ///     TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan_1st));
  ///     for (int exec_index = 0; exec_index < plan_1st->size; exec_index++) {
  ///        int node_index = plan_1st->data[exec_index];
  ///        TfLiteNode* node;
  ///        TfLiteRegistration* reg;
  ///        context->GetNodeAndRegistration(context, node_index, &node, &reg);
  ///        CheckNode(node);
  ///     }
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*GetExecutionPlan)(struct TfLiteContext* context,
                                   TfLiteIntArray** execution_plan);

  /// An array of tensors in the interpreter context (of length `tensors_size`)
  TfLiteTensor* tensors;

  /// opaque full context ptr (an opaque c++ data structure)
  void* impl_;

  /// Request memory pointer be resized. Updates dimensions on the tensor.
  /// NOTE: ResizeTensor takes ownership of newSize.
  TfLiteStatus (*ResizeTensor)(struct TfLiteContext*, TfLiteTensor* tensor,
                               TfLiteIntArray* new_size);
  /// Request that an error be reported with format string msg.
  void (*ReportError)(struct TfLiteContext*, const char* msg, ...);

  /// Add `tensors_to_add` tensors, preserving pre-existing Tensor entries.  If
  /// non-null, the value pointed to by `first_new_tensor_index` will be set to
  /// the index of the first new tensor.
  TfLiteStatus (*AddTensors)(struct TfLiteContext*, int tensors_to_add,
                             int* first_new_tensor_index);

  /// Get a Tensor node by node_index.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*GetNodeAndRegistration)(
      struct TfLiteContext*, int node_index, TfLiteNode** node,
      struct TfLiteRegistration** registration);

  /// Replace ops with one or more stub delegate operations. This function
  /// does not take ownership of `nodes_to_replace`.
  TfLiteStatus (*ReplaceNodeSubsetsWithDelegateKernels)(
      struct TfLiteContext*, struct TfLiteRegistration registration,
      const TfLiteIntArray* nodes_to_replace, struct TfLiteDelegate* delegate);

  /// Number of threads that are recommended to subsystems like gemmlowp and
  /// eigen.
  int recommended_num_threads;

  /// Access external contexts by type.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteExternalContext* (*GetExternalContext)(struct TfLiteContext*,
                                               TfLiteExternalContextType);
  /// Set the value of a external context. Does not take ownership of the
  /// pointer.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  void (*SetExternalContext)(struct TfLiteContext*, TfLiteExternalContextType,
                             TfLiteExternalContext*);

  /// Flag for allowing float16 precision for FP32 calculation.
  /// default: false.
  ///
  /// WARNING: This is an experimental API and subject to change.
  bool allow_fp32_relax_to_fp16;

  /// Pointer to the op-level profiler, if set; nullptr otherwise.
  void* profiler;

  /// Allocate persistent buffer which has the same life time as the
  /// interpreter. Returns `nullptr` on failure. The memory is allocated from
  /// heap for TFL, and from tail in TFLM. This method is only available in
  /// `Init` or `Prepare` stage.
  ///
  /// WARNING: This is an experimental interface that is subject
  /// to change.
  void* (*AllocatePersistentBuffer)(struct TfLiteContext* ctx, size_t bytes);

  /// Allocate a buffer which will be deallocated right after invoke phase.
  /// The memory is allocated from heap in TFL, and from volatile arena in TFLM.
  /// This method is only available in invoke stage.
  ///
  /// NOTE: If possible use `RequestScratchBufferInArena` method to avoid memory
  /// allocation during inference time.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*AllocateBufferForEval)(struct TfLiteContext* ctx, size_t bytes,
                                        void** ptr);

  /// Request a scratch buffer in the arena through static memory planning.
  /// This method is only available in `Prepare` stage and the buffer is
  /// allocated by the interpreter between Prepare and Eval stage. In `Eval`
  /// stage, `GetScratchBuffer` API can be used to fetch the address.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*RequestScratchBufferInArena)(struct TfLiteContext* ctx,
                                              size_t bytes, int* buffer_idx);

  /// Get the scratch buffer pointer.
  /// This method is only available in Eval stage.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  void* (*GetScratchBuffer)(struct TfLiteContext* ctx, int buffer_idx);

  /// Resize the memory pointer of the `tensor`. This method behaves the same as
  /// `ResizeTensor`, except that it makes a copy of the shape array internally
  /// so the shape array could be deallocated right afterwards.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*ResizeTensorExplicit)(struct TfLiteContext* ctx,
                                       TfLiteTensor* tensor, int dims,
                                       const int* shape);

  /// This method provides a preview of post-delegation partitioning. Each
  /// TfLiteDelegateParams in the referenced array corresponds to one instance
  /// of the delegate kernel. Example usage:
  ///
  ///     TfLiteIntArray* nodes_to_replace = ...;
  ///     TfLiteDelegateParams* params_array;
  ///     int num_partitions = 0;
  ///     TF_LITE_ENSURE_STATUS(context->PreviewDelegatePartitioning(
  ///        context, delegate, nodes_to_replace, &params_array,
  ///        &num_partitions));
  ///     for (int idx = 0; idx < num_partitions; idx++) {
  ///        const auto& partition_params = params_array[idx];
  ///        ...
  ///     }
  ///
  /// NOTE: The context owns the memory referenced by partition_params_array. It
  /// will be cleared with another call to PreviewDelegatePartitioning, or after
  /// TfLiteDelegateParams::Prepare returns.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*PreviewDelegatePartitioning)(
      struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
      TfLiteDelegateParams** partition_params_array, int* num_partitions);

  /// Returns a TfLiteTensor struct for a given index.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  ///
  /// WARNING: This method may not be available on all platforms.
  TfLiteTensor* (*GetTensor)(const struct TfLiteContext* context,
                             int tensor_idx);

  /// Returns a TfLiteEvalTensor struct for a given index.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  ///
  /// WARNING: This method may not be available on all platforms.
  TfLiteEvalTensor* (*GetEvalTensor)(const struct TfLiteContext* context,
                                     int tensor_idx);

  /// Retrieves named metadata buffer from the TFLite model.
  /// Returns kTfLiteOk if metadata is successfully obtained from the flatbuffer
  /// Model: that is, there exists a `metadata` entry with given `name` string.
  /// (see TFLite's schema.fbs).
  /// The corresponding `buffer` information is populated in `ptr` & `bytes`.
  /// The data from `ptr` is valid for the lifetime of the Interpreter.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*GetModelMetadata)(const struct TfLiteContext* context,
                                   const char* name, const char** ptr,
                                   size_t* bytes);

  /// Retrieves the corresponding TfLiteContext of a subgraph that the given
  /// subgraph_index points to and switches to the delegate context for that
  /// subgraph. If an invalid subgraph index is given, returns kTfLiteError.
  ///
  /// NOTE: This function is expected to be paired with ReleaseSubgraphContext()
  /// once the delegate preparation is done and/or the delegate context
  /// functions are no longer needed.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*AcquireSubgraphContext)(
      struct TfLiteContext* context, int subgraph_index,
      struct TfLiteContext** acquired_context);
  /// Releases the subgraph context by switching back to the TFLite kernel
  /// context for the subgraph that the given subgraph_index points to.
  ///
  /// NOTE: This function is expected to be used after AcquireSubgraphContext()
  /// once the delegate preparation is done and/or the delegate context
  /// functions are no longer needed.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*ReleaseSubgraphContext)(struct TfLiteContext* context,
                                         int subgraph_index);
} TfLiteContext;

/// `TfLiteOperator` is an external version of `TfLiteRegistration`
/// for C API which doesn't use internal types (such as `TfLiteContext`) but
/// only uses stable API types (such as `TfLiteOpaqueContext`). The purpose of
/// each field is the exactly the same as with `TfLiteRegistration`.
typedef struct TfLiteOperator TfLiteOperator;

#ifndef DOXYGEN_SKIP
// For backwards compatibility.
// Deprecated. Use TfLiteOperator instead.
typedef TfLiteOperator TfLiteRegistrationExternal;
#endif

/// The valid values of the `inplace_operator` field in `TfLiteRegistration`.
/// This allow an op to signal to the runtime that the same data pointer
/// may be passed as an input and output without impacting the result.
/// This does not mean that the memory can safely be reused, it is up to the
/// runtime to determine this, e.g. if another op consumes the same input or not
/// or if an input tensor has sufficient memory allocated to store the output
/// data.
///
/// Setting these flags authorizes the runtime to set the data pointers of an
/// input and output tensor to the same value. In such cases, the memory
/// required by the output must be less than or equal to that required by the
/// shared input, never greater. If kTfLiteInplaceOpDataUnmodified is set, then
/// the runtime can share the same input tensor with multiple operator's
/// outputs, provided that kTfLiteInplaceOpDataUnmodified is set for all of
/// them. Otherwise, if an input tensor is consumed by multiple operators, it
/// may only be shared with the operator which is the last to consume it.
///
/// Note that this is a bitmask, so the values should be 1, 2, 4, 8, ...etc.
typedef enum {
  /// The default value. This indicates that the same data pointer cannot safely
  /// be passed as an op's input and output.
  kTfLiteInplaceOpNone = 0,
  /// This indicates that an op's first output's data is identical to its first
  /// input's data, for example Reshape.
  kTfLiteInplaceOpDataUnmodified = 1,
  /// Setting kTfLiteInplaceInputCanBeSharedWithCorrespondingOutput means
  /// that InputN may be shared with OutputN instead of with the first output.
  /// This flag requires one or more of kTfLiteInplaceOpInputNShared to be set.
  kTfLiteInplaceInputCanBeSharedWithCorrespondingOutput = 2,
  /// kTfLiteInplaceOpInputNShared indicates that it is safe for an op to share
  /// InputN's data pointer with an output tensor. If
  /// kTfLiteInplaceInputCanBeSharedWithCorrespondingOutput is set then
  /// kTfLiteInplaceOpInputNShared indicates that InputN may be shared
  /// with OutputN, otherwise kTfLiteInplaceOpInputNShared indicates that InputN
  /// may be shared with the first output.
  ///
  /// Indicates that an op's first input may be shared with the first output
  /// tensor. kTfLiteInplaceInputCanBeSharedWithCorrespondingOutput has
  /// no impact on the behavior allowed by this flag.
  kTfLiteInplaceOpInput0Shared = 4,
  /// Indicates that an op's second input may be shared with the first output
  /// if kTfLiteInplaceInputCanBeSharedWithCorrespondingOutput is not set
  /// or second output if kTfLiteInplaceInputCanBeSharedWithCorrespondingOutput
  /// is set.
  kTfLiteInplaceOpInput1Shared = 8,
  /// Indicates that an op's third input may be shared with the first output
  /// if kTfLiteInplaceInputCanBeSharedWithCorrespondingOutput is not set
  /// or third output if kTfLiteInplaceInputCanBeSharedWithCorrespondingOutput
  /// is
  /// set.
  kTfLiteInplaceOpInput2Shared = 16,
  /// Placeholder to ensure that enum can hold 64 bit values to accommodate
  /// future fields.
  kTfLiteInplaceOpMaxValue = UINT64_MAX,
} TfLiteInPlaceOp;

/// `TfLiteRegistration` defines the implementation of an operation
/// (a built-in op, custom op, or custom delegate kernel).
///
/// It is a struct containing "methods" (C function pointers) that will be
/// invoked by the TF Lite runtime to evaluate instances of the operation.
///
/// See also `TfLiteOperator` which is a more ABI-stable equivalent.
typedef struct TfLiteRegistration {
  /// Initializes the op from serialized data.
  /// Called only *once* for the lifetime of the op, so any one-time allocations
  /// should be made here (unless they depend on tensor sizes).
  ///
  /// * If a built-in op:
  ///       * `buffer` is the op's params data (TfLiteLSTMParams*).
  ///       * `length` is zero.
  /// * If custom op:
  ///       * `buffer` is the op's `custom_options`.
  ///       * `length` is the size of the buffer.
  ///
  /// Returns a type-punned (i.e. void*) opaque data (e.g. a primitive pointer
  /// or an instance of a struct).
  ///
  /// The returned pointer will be stored with the node in the `user_data`
  /// field, accessible within prepare and invoke functions below.
  ///
  /// NOTE: if the data is already in the desired format, simply implement this
  /// function to return `nullptr` and implement the free function to be a
  /// no-op.
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);

  /// The pointer `buffer` is the data previously returned by an init
  /// invocation.
  void (*free)(TfLiteContext* context, void* buffer);

  /// prepare is called when the inputs this node depends on have been resized.
  /// `context->ResizeTensor()` can be called to request output tensors to be
  /// resized.
  /// Can be called multiple times for the lifetime of the op.
  ///
  /// Returns `kTfLiteOk` on success.
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);

  /// Execute the node (should read `node->inputs` and output to
  /// `node->outputs`).
  ///
  /// Returns `kTfLiteOk` on success.
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);

  /// `profiling_string` is called during summarization of profiling information
  /// in order to group executions together. Providing a value here will cause a
  /// given op to appear multiple times is the profiling report. This is
  /// particularly useful for custom ops that can perform significantly
  /// different calculations depending on their `user-data`.
  const char* (*profiling_string)(const TfLiteContext* context,
                                  const TfLiteNode* node);

  /// Builtin codes. If this kernel refers to a builtin this is the code
  /// of the builtin. This is so we can do marshaling to other frameworks like
  /// NN API.
  ///
  /// Note: It is the responsibility of the registration binder to set this
  /// properly.
  int32_t builtin_code;

  /// Custom op name. If the op is a builtin, this will be `null`.
  ///
  /// Note: It is the responsibility of the registration binder to set this
  /// properly.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  const char* custom_name;

  /// The version of the op.
  /// Note: It is the responsibility of the registration binder to set this
  /// properly.
  int version;

  /// The external (i.e. ABI-stable) version of `TfLiteRegistration`.
  /// Since we can't use internal types (such as `TfLiteContext`) for C API to
  /// maintain ABI stability.  C API user will provide `TfLiteOperator` to
  /// implement custom ops.  We keep it inside of `TfLiteRegistration` and use
  /// it to route callbacks properly.
  TfLiteOperator* registration_external;

  /// Retrieves asynchronous kernel.
  ///
  /// If the `async_kernel` field is nullptr, it means the operation described
  /// by this TfLiteRegistration object does not support asynchronous execution.
  /// Otherwise, the function that the field points to should only be called for
  /// delegate kernel nodes, i.e. `node` should be a delegate kernel node
  /// created by applying a delegate. If the function returns nullptr, that
  /// means that the underlying delegate does not support asynchronous execution
  /// for this `node`.
  struct TfLiteAsyncKernel* (*async_kernel)(TfLiteContext* context,
                                            TfLiteNode* node);

  /// Indicates if an operator's output may safely overwrite its inputs.
  /// See the comments in `TfLiteInPlaceOp`.
  uint64_t inplace_operator;
} TfLiteRegistration;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_CORE_C_COMMON_H_

// LINT.ThenChange(//tensorflow/lite/tools/versioning/op_signature.cc)
