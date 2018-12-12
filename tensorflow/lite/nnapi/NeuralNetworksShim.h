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
#ifndef TENSORFLOW_LITE_NNAPI_NEURALNETWORKSSHIM_H_
#define TENSORFLOW_LITE_NNAPI_NEURALNETWORKSSHIM_H_

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// helpers

#define NNAPI_LOG(format, ...) fprintf(stderr, format "\n", __VA_ARGS__);
#define LOAD_FUNCTION(name) \
  static name##_fn fn = reinterpret_cast<name##_fn>(loadFunction(#name));
#define EXECUTE_FUNCTION(...) \
  if (fn != nullptr) {        \
    fn(__VA_ARGS__);          \
  }
#define EXECUTE_FUNCTION_RETURN(...) return fn != nullptr ? fn(__VA_ARGS__) : 0;

inline void* loadLibrary(const char* name) {
  // TODO: change RTLD_LOCAL? Assumes there can be multiple instances of nn
  // api RT
  void* handle = nullptr;
#ifdef __ANDROID__
  handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    NNAPI_LOG("nnapi error: unable to open library %s", name);
  }
#endif
  return handle;
}

typedef int (*ASharedMemory_create_fn)(const char* name, size_t size);

// ASharedMemory_create was added in Android 8.0, so safe to use with NNAPI
// which was added in 8.1.
inline int ASharedMemory_create(const char* name, size_t size) {
  static void* handle = loadLibrary("libandroid.so");
  static ASharedMemory_create_fn fn =
      handle != nullptr ? reinterpret_cast<ASharedMemory_create_fn>(
                              dlsym(handle, "ASharedMemory_create"))
                        : nullptr;
  return fn(name, size);
}

inline void* getLibraryHandle() {
  static void* handle = loadLibrary("libneuralnetworks.so");
  return handle;
}

inline void* loadFunction(const char* name) {
  void* fn = nullptr;
  if (getLibraryHandle() != nullptr) {
    fn = dlsym(getLibraryHandle(), name);
  }
  if (fn == nullptr) {
    NNAPI_LOG("nnapi error: unable to open function %s", name);
  }
  return fn;
}

inline bool NNAPIExists() {
  static bool nnapi_is_available = getLibraryHandle();
  return nnapi_is_available;
}

// NN api types based on NNAPI header file
// https://developer.android.com/ndk/reference/group/neural-networks

/**
 * Operand types.
 *
 * The type of operands that can be added to a model.
 *
 * Although we define many types, most operators accept just a few
 * types.  Most used are ANEURALNETWORKS_TENSOR_FLOAT32,
 * ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, and ANEURALNETWORKS_INT32.
 */
enum {
  ANEURALNETWORKS_FLOAT32 = 0,
  ANEURALNETWORKS_INT32 = 1,
  ANEURALNETWORKS_UINT32 = 2,
  ANEURALNETWORKS_TENSOR_FLOAT32 = 3,
  ANEURALNETWORKS_TENSOR_INT32 = 4,
  ANEURALNETWORKS_TENSOR_QUANT8_ASYMM = 5,
};

/**
 * Operation types.
 *
 * The type of operations that can be added to a model.
 */
enum {
  ANEURALNETWORKS_ADD = 0,
  ANEURALNETWORKS_AVERAGE_POOL_2D = 1,
  ANEURALNETWORKS_CONCATENATION = 2,
  ANEURALNETWORKS_CONV_2D = 3,
  ANEURALNETWORKS_DEPTHWISE_CONV_2D = 4,
  ANEURALNETWORKS_DEPTH_TO_SPACE = 5,
  ANEURALNETWORKS_DEQUANTIZE = 6,
  ANEURALNETWORKS_EMBEDDING_LOOKUP = 7,
  ANEURALNETWORKS_FLOOR = 8,
  ANEURALNETWORKS_FULLY_CONNECTED = 9,
  ANEURALNETWORKS_HASHTABLE_LOOKUP = 10,
  ANEURALNETWORKS_L2_NORMALIZATION = 11,
  ANEURALNETWORKS_L2_POOL_2D = 12,
  ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION = 13,
  ANEURALNETWORKS_LOGISTIC = 14,
  ANEURALNETWORKS_LSH_PROJECTION = 15,
  ANEURALNETWORKS_LSTM = 16,
  ANEURALNETWORKS_MAX_POOL_2D = 17,
  ANEURALNETWORKS_MUL = 18,
  ANEURALNETWORKS_RELU = 19,
  ANEURALNETWORKS_RELU1 = 20,
  ANEURALNETWORKS_RELU6 = 21,
  ANEURALNETWORKS_RESHAPE = 22,
  ANEURALNETWORKS_RESIZE_BILINEAR = 23,
  ANEURALNETWORKS_RNN = 24,
  ANEURALNETWORKS_SOFTMAX = 25,
  ANEURALNETWORKS_SPACE_TO_DEPTH = 26,
  ANEURALNETWORKS_SVDF = 27,
  ANEURALNETWORKS_TANH = 28,
  ANEURALNETWORKS_BATCH_TO_SPACE_ND = 29,
  ANEURALNETWORKS_DIV = 30,
  ANEURALNETWORKS_MEAN = 31,
  ANEURALNETWORKS_PAD = 32,
  ANEURALNETWORKS_SPACE_TO_BATCH_ND = 33,
  ANEURALNETWORKS_SQUEEZE = 34,
  ANEURALNETWORKS_STRIDED_SLICE = 35,
  ANEURALNETWORKS_SUB = 36,
  ANEURALNETWORKS_TRANSPOSE = 37,
  ANEURALNETWORKS_CEIL = 38,
};

/**
 * Fused activation function types.
 *
 */
enum {
  ANEURALNETWORKS_FUSED_NONE = 0,
  ANEURALNETWORKS_FUSED_RELU = 1,
  ANEURALNETWORKS_FUSED_RELU1 = 2,
  ANEURALNETWORKS_FUSED_RELU6 = 3,
};

/**
 * Execution preferences.
 */
enum {
  ANEURALNETWORKS_PREFER_LOW_POWER = 0,
  ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1,
  ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2,
};

/**
 * Result codes.
 */
enum {
  ANEURALNETWORKS_NO_ERROR = 0,
  ANEURALNETWORKS_OUT_OF_MEMORY = 1,
  ANEURALNETWORKS_INCOMPLETE = 2,
  ANEURALNETWORKS_UNEXPECTED_NULL = 3,
  ANEURALNETWORKS_BAD_DATA = 4,
  ANEURALNETWORKS_OP_FAILED = 5,
  ANEURALNETWORKS_UNMAPPABLE = 5,
  ANEURALNETWORKS_BAD_STATE = 6,
};

/**
 * Implicit padding algorithms.
 */
enum {
  ANEURALNETWORKS_PADDING_SAME = 1,
  ANEURALNETWORKS_PADDING_VALID = 2,
};

/**
 * ANeuralNetworksMemory is an opaque type that represents memory.
 *
 * This type is used to represent shared memory, memory mapped files,
 * and similar memories.
 *
 * By using shared memory, a program can efficiently communicate to the
 * runtime and drivers the tensors that define a model. See
 * {@link ANeuralNetworksModel_setOperandValueFromMemory}. An application
 * should typically create one shared memory object that contains every tensor
 * needed to define a model. {@link ANeuralNetworksMemory_createFromFd} can be
 * used to create shared memory from a file handle. {@link
 * ANeuralNetworksMemory_createShared} can be used to directly created shared
 * memory.
 *
 * Memory objects can also be used to specify the input and output arguments of
 * an execution. See {@link ANeuralNetworksExecution_setInputFromMemory}
 * and {@link ANeuralNetworksExecution_setOutputFromMemory}.
 */
typedef struct ANeuralNetworksMemory ANeuralNetworksMemory;

/**
 * ANeuralNetworksModel is an opaque type that contains a description of the
 * mathematical operations that constitute the model.
 *
 * <p>The model will be built by calling<ul>
 * <li>{@link ANeuralNetworksModel_create},</li>
 * <li>{@link ANeuralNetworksModel_addOperation},</li>
 * <li>{@link ANeuralNetworksModel_addOperand},</li>
 * </ul>
 *
 * A model is completed by calling {@link ANeuralNetworksModel_finish}.
 * A model is destroyed by calling {@link ANeuralNetworksModel_free}.
 *
 * <p>It is the application's responsibility to make sure that only one thread
 * modifies a model at a given time. It is however safe for more than one
 * thread to use the model once {@link ANeuralNetworksModel_finish} has
 * returned.</p>
 *
 * <p>It is also the application's responsibility to ensure that there are no
 * other uses of the model after calling {@link ANeuralNetworksModel_free}. This
 * includes any compilation or execution object created using the model.</p>
 */
typedef struct ANeuralNetworksModel ANeuralNetworksModel;

/**
 * ANeuralNetworksCompilation is an opaque type that can be used to compile
 * a machine learning model.
 *
 * <p>To use:<ul>
 *    <li>Create a new compilation instance by calling the
 *        {@link ANeuralNetworksCompilation_create} function.</li>
 *    <li>Perform the compilation with {@link
 * ANeuralNetworksCompilation_start}.</li> <li>Wait for the compilation to
 * complete with {@link ANeuralNetworksCompilation_wait}.</li> <li>Use the
 * compilation as many times as needed with {@link
 * ANeuralNetworksExecution_create}.</li> <li>Destroy the compilation with
 * {@link ANeuralNetworksCompilation_free} once all executions using the
 * compilation have completed.</li></ul></p>
 *
 * <p>A compilation cannot be modified once {@link
 * ANeuralNetworksCompilation_start} has been called on it.</p>
 *
 * <p>It is the application's responsibility to make sure that only one thread
 * modifies a compilation at a given time. It is however safe for more than one
 * thread to use {@link ANeuralNetworksCompilation_wait} at the same time.
 * It is also safe for multiple threads to use a compilation object once
 * {@link ANeuralNetworksCompilation_wait} has completed.</p>
 *
 * <p>It is also the application's responsibility to ensure that there are no
 * other uses of the compilation after calling {@link
 * ANeuralNetworksCompilation_free}. This includes any execution object created
 * using the compilation.</p>
 */
typedef struct ANeuralNetworksCompilation ANeuralNetworksCompilation;

/**
 * ANeuralNetworksExecution is an opaque type that can be used to apply a
 * machine learning model to a set of inputs.
 *
 * <p>To use:<ul>
 *    <li>Create a new execution instance by calling the
 *        {@link ANeuralNetworksExecution_create} function.</li>
 *    <li>Associate data to the model inputs with
 *        {@link ANeuralNetworksExecution_setInput} or
 *        {@link ANeuralNetworksExecution_setInputFromMemory}.</li>
 *    <li>Associate output buffers to the model outputs with
 *        {@link ANeuralNetworksExecution_setOutput} or
 *        {@link ANeuralNetworksExecution_setOutputFromMemory}.</li>
 *    <li>Apply the model with {@link
 * ANeuralNetworksExecution_startCompute}.</li> <li>Wait for the execution to
 * complete with {@link ANeuralNetworksExecution_wait}.</li> <li>Destroy the
 * execution with
 *        {@link ANeuralNetworksExecution_free}.</li></ul></p>
 *
 * <p>An execution cannot be modified once {@link
 * ANeuralNetworksExecution_start} has been called on it.</p>
 *
 * <p>An execution can be applied to a model with
 * {@link ANeuralNetworksExecution_startCompute} only once. Create new
 * executions to do new evaluations of the model.</p>
 *
 * <p>It is the application's responsibility to make sure that only one thread
 * modifies an execution at a given time. It is however safe for more than one
 * thread to use {@link ANeuralNetworksExecution_wait} at the same time.</p>
 *
 * <p>It is also the application's responsibility to ensure that there are no
 * other uses of the request after calling {@link
 * ANeuralNetworksRequest_free}.</p>
 */
typedef struct ANeuralNetworksExecution ANeuralNetworksExecution;

/**
 * ANeuralNetworksOperandType describes the type of an operand.
 * This structure is used to describe both scalars and tensors.
 */
typedef struct ANeuralNetworksOperandType {
  /** The data type, e.g ANEURALNETWORKS_INT8. */
  int32_t type;
  /** The number of dimensions. It should be 0 for scalars. */
  uint32_t dimensionCount;
  /** The dimensions of the tensor. It should be nullptr for scalars. */
  const uint32_t* dimensions;
  /** These two fields are only used for quantized tensors.
   * They should be zero for scalars and non-fixed point tensors.
   * The dequantized value of each entry is (value - offset) * scale.
   */
  float scale;
  int32_t zeroPoint;
} ANeuralNetworksOperandType;

/**
 * ANeuralNetworksEvent is an opaque type that represents an event
 * that will be signaled once an execution completes.
 */
typedef struct ANeuralNetworksEvent ANeuralNetworksEvent;

typedef int32_t ANeuralNetworksOperationType;

// nn api function types

typedef int (*ANeuralNetworksMemory_createFromFd_fn)(
    size_t size, int protect, int fd, size_t offset,
    ANeuralNetworksMemory** memory);

typedef void (*ANeuralNetworksMemory_free_fn)(ANeuralNetworksMemory* memory);

typedef int (*ANeuralNetworksModel_create_fn)(ANeuralNetworksModel** model);

typedef int (*ANeuralNetworksModel_finish_fn)(ANeuralNetworksModel* model);

typedef void (*ANeuralNetworksModel_free_fn)(ANeuralNetworksModel* model);

typedef int (*ANeuralNetworksCompilation_create_fn)(
    ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation);

typedef void (*ANeuralNetworksCompilation_free_fn)(
    ANeuralNetworksCompilation* compilation);

typedef int (*ANeuralNetworksCompilation_setPreference_fn)(
    ANeuralNetworksCompilation* compilation, int32_t preference);

typedef int (*ANeuralNetworksCompilation_finish_fn)(
    ANeuralNetworksCompilation* compilation);

typedef int (*ANeuralNetworksModel_addOperand_fn)(
    ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type);

typedef int (*ANeuralNetworksModel_setOperandValue_fn)(
    ANeuralNetworksModel* model, int32_t index, const void* buffer,
    size_t length);

typedef int (*ANeuralNetworksModel_setOperandValueFromMemory_fn)(
    ANeuralNetworksModel* model, int32_t index,
    const ANeuralNetworksMemory* memory, size_t offset, size_t length);

typedef int (*ANeuralNetworksModel_addOperation_fn)(
    ANeuralNetworksModel* model, ANeuralNetworksOperationType type,
    uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
    const uint32_t* outputs);

typedef int (*ANeuralNetworksModel_identifyInputsAndOutputs_fn)(
    ANeuralNetworksModel* model, uint32_t inputCount, const uint32_t* inputs,
    uint32_t outputCount, const uint32_t* outputs);

typedef int (*ANeuralNetworksModel_relaxComputationFloat32toFloat16_fn)(
    ANeuralNetworksModel* model, bool allow);

typedef int (*ANeuralNetworksExecution_create_fn)(
    ANeuralNetworksCompilation* compilation,
    ANeuralNetworksExecution** execution);

typedef void (*ANeuralNetworksExecution_free_fn)(
    ANeuralNetworksExecution* execution);

typedef int (*ANeuralNetworksExecution_setInput_fn)(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const void* buffer, size_t length);

typedef int (*ANeuralNetworksExecution_setInputFromMemory_fn)(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    size_t offset, size_t length);

typedef int (*ANeuralNetworksExecution_setOutput_fn)(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, void* buffer, size_t length);

typedef int (*ANeuralNetworksExecution_setOutputFromMemory_fn)(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    size_t offset, size_t length);

typedef int (*ANeuralNetworksExecution_startCompute_fn)(
    ANeuralNetworksExecution* execution, ANeuralNetworksEvent** event);

typedef int (*ANeuralNetworksEvent_wait_fn)(ANeuralNetworksEvent* event);

typedef void (*ANeuralNetworksEvent_free_fn)(ANeuralNetworksEvent* event);

/**
 * Creates a shared memory object from a file descriptor.
 *
 * The shared memory is backed by a file descriptor via mmap.
 * See {@link ANeuralNetworksMemory} for a description on how to use
 * this shared memory.
 *
 * @param size The requested size in bytes.
 *             Must not be larger than the file size.
 * @param prot The desired memory protection for the mapping.
 *             It is either PROT_NONE or the bitwise OR of one or
 *             more of the following flags: PROT_READ, PROT_WRITE.
 * @param fd The requested file descriptor.
 *           The file descriptor has to be mmap-able. The file
 *           descriptor will be duplicated.
 * @param offset The offset to the beginning of the file of the area to map.
 *               The offset has to be aligned to a page size.
 * @param memory The memory object to be created.
 *               Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the request completed normally.
 */
inline int ANeuralNetworksMemory_createFromFd(size_t size, int protect, int fd,
                                              size_t offset,
                                              ANeuralNetworksMemory** memory) {
  LOAD_FUNCTION(ANeuralNetworksMemory_createFromFd);
  EXECUTE_FUNCTION_RETURN(size, protect, fd, offset, memory);
}

/**
 * Delete a memory object.
 *
 * Destroys the object used by the run time to keep track of the memory.
 * This will free the underlying actual memory if no other code has open
 * handles to this memory.
 *
 * @param memory The memory object to be freed.
 */
inline void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory) {
  LOAD_FUNCTION(ANeuralNetworksMemory_free);
  EXECUTE_FUNCTION(memory);
}

/**
 * Create an empty {@link ANeuralNetworksModel}.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuralNetworksExecution_startCompute} is invoked.
 *
 * The model should be constructed with calls to
 * {@link ANeuralNetworksModel_addOperation} and
 * {@link ANeuralNetworksModel_addOperand}
 *
 * <p>{@link ANeuralNetworksModel_finish} should be called once the model
 * has been fully constructed.</p>
 *
 * <p>{@link ANeuralNetworksModel_free} should be called once the model
 * is no longer needed.</p>
 *
 * @param model The {@link ANeuralNetworksModel} to be created.
 *              Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_create(ANeuralNetworksModel** model) {
  LOAD_FUNCTION(ANeuralNetworksModel_create);
  EXECUTE_FUNCTION_RETURN(model);
}

/**
 * Destroy a model.
 *
 * The model need not have been finished by a call to
 * {@link ANeuralNetworksModel_finish}.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be destroyed. Passing NULL is acceptable and
 *              results in no operation.
 */
inline void ANeuralNetworksModel_free(ANeuralNetworksModel* model) {
  LOAD_FUNCTION(ANeuralNetworksModel_free);
  EXECUTE_FUNCTION(model);
}

/**
 * Indicate that we have finished modifying a model. Required before
 * calling {@link ANeuralNetworksCompilation_compile}.
 *
 * An application is responsible to make sure that no other thread uses
 * the model at the same time.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be finished.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_finish(ANeuralNetworksModel* model) {
  LOAD_FUNCTION(ANeuralNetworksModel_finish);
  EXECUTE_FUNCTION_RETURN(model);
}

/**
 * Add an operand to a model.
 *
 * The order in which the operands are added is important. The first one added
 * to a model will have the index value 0, the second 1, etc. These indexes are
 * used as operand identifiers in {@link ANeuralNetworksModel_addOperation},
 * {@link ANeuralNetworksExecution_setInput},
 * {@link ANeuralNetworksExecution_setInputFromMemory},
 * {@link ANeuralNetworksExecution_setOutput},
 * {@link ANeuralNetworksExecution_setOutputFromMemory} and
 * {@link ANeuralNetworksExecution_setOperandValue}.
 *
 * To build a model that can accommodate inputs of various sizes, as you may
 * want to do for a CNN, set the size of the dimensions that will vary at run
 * time to 0. If you do so, provide the full dimensions when calling
 * {@link ANeuralNetworksExecution_setInput} or {@link
 * ANeuralNetworksExecution_setInputFromMemory}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be modified.
 * @param type The {@link ANeuralNetworksOperandType} that describes the shape
 * of the operand.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_addOperand(
    ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type) {
  LOAD_FUNCTION(ANeuralNetworksModel_addOperand);
  EXECUTE_FUNCTION_RETURN(model, type);
}

/**
 * Sets an operand to a constant value.
 *
 * For scalar values, the content of buffer is copied into the model.
 *
 * For tensor values, a pointer to the buffer is stored within the model.
 * The application is responsible for not changing the content of this region
 * until all executions using this model have completed. As the data may
 * be copied during processing, modifying the data after this call yields
 * undefined results.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param buffer A pointer to the data to use.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model,
                                                int32_t index,
                                                const void* buffer,
                                                size_t length) {
  LOAD_FUNCTION(ANeuralNetworksModel_setOperandValue);
  EXECUTE_FUNCTION_RETURN(model, index, buffer, length);
}

/**
 * Sets an operand to a value stored in a memory object.
 *
 * The content of the memory is not copied. A reference to that memory is stored
 * inside the model. The application is responsible for not changing the content
 * of the memory region until all executions using this model have completed.
 * As the data may be copied during processing, modifying the data after this
 * call yields undefined results.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param buffer A pointer to the data to use.
 * @param memory The memory containing the data.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_setOperandValueFromMemory(
    ANeuralNetworksModel* model, int32_t index,
    const ANeuralNetworksMemory* memory, size_t offset, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksModel_setOperandValueFromMemory);
  EXECUTE_FUNCTION_RETURN(model, index, memory, offset, length);
}

/**
 * Add an operation to a model.
 *
 * @param model The model to be modified.
 * @param type The type of the operation.
 * @param inputCount The number of entries in the inputs array.
 * @param inputs An array of indexes identifying each operand.
 * @param outputCount The number of entries in the outputs array.
 * @param outputs An array of indexes identifying each operand.
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to {@link ANeuralNetworksModel_addOperand}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                             ANeuralNetworksOperationType type,
                                             uint32_t inputCount,
                                             const uint32_t* inputs,
                                             uint32_t outputCount,
                                             const uint32_t* outputs) {
  LOAD_FUNCTION(ANeuralNetworksModel_addOperation);
  EXECUTE_FUNCTION_RETURN(model, type, inputCount, inputs, outputCount,
                          outputs);
}

/**
 * Specifies which operands will be the model's inputs and outputs.
 *
 * An operand cannot be used for both input and output. Doing so will
 * return an error.
 *
 * @param model The model to be modified.
 * @param inputCount The number of entries in the inputs array.
 * @param inputs An array of indexes identifying the input operands.
 * @param outputCount The number of entries in the outputs array.
 * @param outputs An array of indexes identifying the output operands.
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to {@link ANeuralNetworksModel_addOperand}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 */
inline int ANeuralNetworksModel_identifyInputsAndOutputs(
    ANeuralNetworksModel* model, uint32_t inputCount, const uint32_t* inputs,
    uint32_t outputCount, const uint32_t* outputs) {
  LOAD_FUNCTION(ANeuralNetworksModel_identifyInputsAndOutputs);
  EXECUTE_FUNCTION_RETURN(model, inputCount, inputs, outputCount, outputs);
}

/**
 * Specifies whether {@link ANEURALNETWORKS_TENSOR_FLOAT32} is allowed to be
 * calculated with range and/or precision as low as that of the IEEE 754 16-bit
 * floating-point format. By default, {@link ANEURALNETWORKS_TENSOR_FLOAT32}
 * must be calculated using at least the range and precision of the IEEE 754
 * 32-bit floating-point format.
 *
 * @param model The model to be modified.
 * @param allow 'true' indicates {@link ANEURALNETWORKS_TENSOR_FLOAT32} may be
 *              calculated with range and/or precision as low as that of the
 *              IEEE 754 16-bit floating point format. 'false' indicates
 *              {@link ANEURALNETWORKS_TENSOR_FLOAT32} must be calculated using
 *              at least the range and precision of the IEEE 754 32-bit floating
 *              point format.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * Available since API level 28.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 */
inline int ANeuralNetworksModel_relaxComputationFloat32toFloat16(
    ANeuralNetworksModel* model, bool allow) {
  LOAD_FUNCTION(ANeuralNetworksModel_relaxComputationFloat32toFloat16);
  EXECUTE_FUNCTION_RETURN(model, allow);
}

/**
 * Create a {@link ANeuralNetworksCompilation} to compile the given model.
 * This only creates the object. Compilation is only performed once
 * {@link ANeuralNetworksCompilation_start} is invoked.
 *
 * <p>The provided model must outlive the compilation.</p>
 *
 * The model must already have been finished by a call to
 * {@link ANeuralNetworksModel_finish}.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @param model The {@link ANeuralNetworksModel} to be compiled.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the model is invalid.
 */
inline int ANeuralNetworksCompilation_create(
    ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_create);
  EXECUTE_FUNCTION_RETURN(model, compilation);
}

/**
 * Destroy a compilation.
 *
 * <p>If called on a compilation for which
 * {@link ANeuralNetworksCompilation_start} has been called, the
 * function will return immediately but will mark the compilation to be deleted
 * once the compilation completes. The {@link ANeuralNetworksCompilation_wait}
 * will return ERROR_DELETED.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @param compilation The compilation to be destroyed. Passing NULL is
 * acceptable and results in no operation.
 */
inline void ANeuralNetworksCompilation_free(
    ANeuralNetworksCompilation* compilation) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_free);
  EXECUTE_FUNCTION(compilation);
}

/**
 * Sets the execution preference.
 *
 * <p>Provides guidance to the runtime when trade-offs are possible.</p>
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @param compilation The compilation to be modified.
 * @param preference Either {@link PREFER_LOW_POWER},
 *                  {@link PREFER_SINGLE_FAST_ANSWER}, or
 *                  {@link PREFER_SUSTAINED_SPEED}.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksCompilation_setPreference(
    ANeuralNetworksCompilation* compilation, int32_t preference) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_setPreference);
  EXECUTE_FUNCTION_RETURN(compilation, preference);
}

/**
 * Waits until the compilation completes.
 *
 * More than one thread can wait on a compilation. When the compilation
 * completes, all threads will be released.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the compilation completed normally.
 */
inline int ANeuralNetworksCompilation_finish(
    ANeuralNetworksCompilation* compilation) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_finish);
  EXECUTE_FUNCTION_RETURN(compilation);
}
/**
 * Create a {@link ANeuralNetworksExecution} to apply the given compilation.
 * This only creates the object. Computation is only performed once
 * {@link ANeuralNetworksExecution_startCompute} is invoked.
 *
 * <p>The provided compilation must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param compilation The {@link ANeuralNetworksCompilation} to be evaluated.
 * @param execution The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the compilation is invalid.
 */
inline int ANeuralNetworksExecution_create(
    ANeuralNetworksCompilation* compilation,
    ANeuralNetworksExecution** execution) {
  LOAD_FUNCTION(ANeuralNetworksExecution_create);
  EXECUTE_FUNCTION_RETURN(compilation, execution);
}

/**
 * Destroy an execution.
 *
 * <p>If called on an execution for which
 * {@link ANeuralNetworksExecution_startCompute} has been called, the
 * function will return immediately but will mark the execution to be deleted
 * once the computation completes.   The {link ANeuralNetworksExecution_wait}
 * will return ANEURALNETWORKS_ERROR_DELETED.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be destroyed. Passing NULL is acceptable
 * and results in no operation.
 */
inline void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution) {
  LOAD_FUNCTION(ANeuralNetworksExecution_free);
  EXECUTE_FUNCTION(execution);
}

/**
 * Associate a user buffer with an input of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided buffer must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be modified.
 * @param index The index of the input argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The type of the operand. This should be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other properties of the type must be the same as
 *             specified in the model. If the type is the same as specified
 *             when the model was built, NULL can be passed.
 * @param buffer The buffer containing the data.
 * @param length The length in bytes of the buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the input.
 */
inline int ANeuralNetworksExecution_setInput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const void* buffer, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setInput);
  EXECUTE_FUNCTION_RETURN(execution, index, type, buffer, length);
}

/**
 * Associate part of a memory object with an input of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided memory must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be modified.
 * @param index The index of the input argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The type of the operand. This can be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other values must be the same as specified in the
 *             model. If the type is the same as specified when the model
 *             was built, NULL can be passed.
 * @param memory The memory containing the data.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the input.
 */
inline int ANeuralNetworksExecution_setInputFromMemory(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    size_t offset, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setInputFromMemory);
  EXECUTE_FUNCTION_RETURN(execution, index, type, memory, offset, length);
}

/**
 * Associate a user buffer with an output of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided buffer must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be modified.
 * @param index The index of the output argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The type of the operand. This can be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other values must be the same as specified in the
 *             model. If the type is the same as specified when the model
 *             was built, NULL can be passed.
 * @param buffer The buffer where the data is to be written.
 * @param length The length in bytes of the buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the output.
 */
inline int ANeuralNetworksExecution_setOutput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, void* buffer, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setOutput);
  EXECUTE_FUNCTION_RETURN(execution, index, type, buffer, length);
}

/**
 * Associate part of a memory object with an output of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided memory must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be modified.
 * @param index The index of the output argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The type of the operand. This can be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other values must be the same as specified in the
 *             model. If the type is the same as specified when the model
 *             was built, NULL can be passed.
 * @param memory The memory where the data is to be stored.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The length in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the output.
 */
inline int ANeuralNetworksExecution_setOutputFromMemory(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    size_t offset, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setOutputFromMemory);
  EXECUTE_FUNCTION_RETURN(execution, index, type, memory, offset, length);
}

/**
 * Schedule evaluation of the execution.
 *
 * <p>Schedules evaluation of the execution. Once the model has been
 * applied and the outputs are ready to be consumed, the execution will be
 * signaled. Use {@link ANeuralNetworksExecution_wait} to wait for that signal.
 * </p>
 *
 * Multiple executions can be scheduled and evaluated concurrently, and
 * compilations can be performed concurrently with executions. The runtime makes
 * no guarantee on the ordering of the completion of compilations and
 * executions. If it's important to the application, the application should
 * enforce the ordering by using {@link ANeuralNetworksCompilation_wait} and
 * {@link ANeuralNetworksExecution_wait}.
 *
 * ANeuralNetworksExecution_wait must be called to recuperate the resources used
 * by the execution.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be scheduled and executed.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksExecution_startCompute(
    ANeuralNetworksExecution* execution, ANeuralNetworksEvent** event) {
  LOAD_FUNCTION(ANeuralNetworksExecution_startCompute);
  EXECUTE_FUNCTION_RETURN(execution, event);
}

/**
 * Waits until the execution completes.
 *
 * More than one thread can wait on an event. When the execution completes,
 * all threads will be released.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the execution completed normally.
 */
inline int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event) {
  LOAD_FUNCTION(ANeuralNetworksEvent_wait);
  EXECUTE_FUNCTION_RETURN(event);
}

/**
 * Destroys the event.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 */
inline void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event) {
  LOAD_FUNCTION(ANeuralNetworksEvent_free);
  EXECUTE_FUNCTION(event);
}

/**/

#endif  // TENSORFLOW_LITE_NNAPI_NEURALNETWORKSSHIM_H_
