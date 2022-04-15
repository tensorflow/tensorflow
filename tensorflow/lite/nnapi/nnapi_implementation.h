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
#ifndef TENSORFLOW_LITE_NNAPI_NNAPI_IMPLEMENTATION_H_
#define TENSORFLOW_LITE_NNAPI_NNAPI_IMPLEMENTATION_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <memory>

#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"

struct NnApi {
  bool nnapi_exists;
  int32_t android_sdk_version;
  // NNAPI feature level should be used when deciding which NNAPI feature to
  // use, as feature levels after Android API level 31 have no association with
  // API level because the NNAPI specification can be updated between Android
  // API releases.
  int64_t nnapi_runtime_feature_level;

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
  int (*ANeuralNetworksMemory_createFromFd)(size_t size, int protect, int fd,
                                            size_t offset,
                                            ANeuralNetworksMemory** memory);

  /**
   * Delete a memory object.
   *
   * Destroys the object used by the run time to keep track of the memory.
   * This will free the underlying actual memory if no other code has open
   * handles to this memory.
   *
   * @param memory The memory object to be freed.
   */
  void (*ANeuralNetworksMemory_free)(ANeuralNetworksMemory* memory);

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
  int (*ANeuralNetworksModel_create)(ANeuralNetworksModel** model);

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
  void (*ANeuralNetworksModel_free)(ANeuralNetworksModel* model);

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
  int (*ANeuralNetworksModel_finish)(ANeuralNetworksModel* model);

  /**
   * Add an operand to a model.
   *
   * The order in which the operands are added is important. The first one added
   * to a model will have the index value 0, the second 1, etc. These indexes
   * are used as operand identifiers in
   * {@link ANeuralNetworksModel_addOperation},
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
  int (*ANeuralNetworksModel_addOperand)(
      ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type);

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
  int (*ANeuralNetworksModel_setOperandValue)(ANeuralNetworksModel* model,
                                              int32_t index, const void* buffer,
                                              size_t length);

  /**
   * Sets an operand's per channel quantization parameters.
   *
   * Sets parameters required by a tensor of type
   * {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}.
   * This function must be called for every tensor of type
   * {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL} before
   * calling {@link ANeuralNetworksModel_finish}.
   *
   * Available since API level 29.
   *
   * @param model The model to be modified.
   * @param index The index of the model operand we're setting.
   * @param channelQuant The per channel quantization parameters for the
   *                     operand. No memory in this struct needs to outlive the
   *                     call to this function.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksModel_setOperandSymmPerChannelQuantParams)(
      ANeuralNetworksModel* model, int32_t index,
      const ANeuralNetworksSymmPerChannelQuantParams* channelQuant);

  /**
   * Sets an operand to a value stored in a memory object.
   *
   * The content of the memory is not copied. A reference to that memory is
   * stored inside the model. The application is responsible for not changing
   * the content of the memory region until all executions using this model have
   * completed.
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
  int (*ANeuralNetworksModel_setOperandValueFromMemory)(
      ANeuralNetworksModel* model, int32_t index,
      const ANeuralNetworksMemory* memory, size_t offset, size_t length);

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
  int (*ANeuralNetworksModel_addOperation)(ANeuralNetworksModel* model,
                                           ANeuralNetworksOperationType type,
                                           uint32_t inputCount,
                                           const uint32_t* inputs,
                                           uint32_t outputCount,
                                           const uint32_t* outputs);

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
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksModel_identifyInputsAndOutputs)(
      ANeuralNetworksModel* model, uint32_t inputCount, const uint32_t* inputs,
      uint32_t outputCount, const uint32_t* outputs);

  /**
   * Specifies whether {@link ANEURALNETWORKS_TENSOR_FLOAT32} is allowed to be
   * calculated with range and/or precision as low as that of the
   * IEEE 754 16-bit floating-point format. By default,
   * {@link ANEURALNETWORKS_TENSOR_FLOAT32} must be calculated using at least
   * the range and precision of the IEEE 754 32-bit floating-point format.
   *
   * @param model The model to be modified.
   * @param allow 'true' indicates {@link ANEURALNETWORKS_TENSOR_FLOAT32} may be
   *              calculated with range and/or precision as low as that of the
   *              IEEE 754 16-bit floating point format. 'false' indicates
   *              {@link ANEURALNETWORKS_TENSOR_FLOAT32} must be calculated
   *              using at least the range and precision of the IEEE 754 32-bit
   *              floating point format.
   *
   * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
   * been called will return an error.
   *
   * Available since API level 28.
   *
   * See {@link ANeuralNetworksModel} for information on multithreaded usage.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksModel_relaxComputationFloat32toFloat16)(
      ANeuralNetworksModel* model, bool allow);

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
  int (*ANeuralNetworksCompilation_create)(
      ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation);

  /**
   * Destroy a compilation.
   *
   * <p>If called on a compilation for which
   * {@link ANeuralNetworksCompilation_start} has been called, the
   * function will return immediately but will mark the compilation to be
   * deleted once the compilation completes. The
   * {@link ANeuralNetworksCompilation_wait} will return ERROR_DELETED.
   *
   * See {@link ANeuralNetworksCompilation} for information on multithreaded
   * usage.
   *
   * @param compilation The compilation to be destroyed. Passing NULL is
   * acceptable and results in no operation.
   */
  void (*ANeuralNetworksCompilation_free)(
      ANeuralNetworksCompilation* compilation);

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
  int (*ANeuralNetworksCompilation_setPreference)(
      ANeuralNetworksCompilation* compilation, int32_t preference);

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
  int (*ANeuralNetworksCompilation_finish)(
      ANeuralNetworksCompilation* compilation);

  /**
   * Create a {@link ANeuralNetworksExecution} to apply the given compilation.
   * This only creates the object. Computation is only performed once
   * {@link ANeuralNetworksExecution_startCompute} is invoked.
   *
   * <p>The provided compilation must outlive the execution.</p>
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * @param compilation The {@link ANeuralNetworksCompilation} to be evaluated.
   * @param execution The newly created object or NULL if unsuccessful.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
   *         if the compilation is invalid.
   */
  int (*ANeuralNetworksExecution_create)(
      ANeuralNetworksCompilation* compilation,
      ANeuralNetworksExecution** execution);

  /**
   * Destroy an execution.
   *
   * <p>If called on an execution for which
   * {@link ANeuralNetworksExecution_startCompute} has been called, the
   * function will return immediately but will mark the execution to be deleted
   * once the computation completes.   The {link ANeuralNetworksExecution_wait}
   * will return ANEURALNETWORKS_ERROR_DELETED.
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * @param execution The execution to be destroyed. Passing NULL is acceptable
   * and results in no operation.
   */
  void (*ANeuralNetworksExecution_free)(ANeuralNetworksExecution* execution);

  /**
   * Associate a user buffer with an input of the model of the
   * {@link ANeuralNetworksExecution}.
   *
   * <p>The provided buffer must outlive the execution.</p>
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * @param execution The execution to be modified.
   * @param index The index of the input argument we are setting. It is
   *              an index into the lists passed to
   *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is
   *              not the index associated with {@link
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
  int (*ANeuralNetworksExecution_setInput)(
      ANeuralNetworksExecution* execution, int32_t index,
      const ANeuralNetworksOperandType* type, const void* buffer,
      size_t length);

  /**
   * Associate part of a memory object with an input of the model of the
   * {@link ANeuralNetworksExecution}.
   *
   * <p>The provided memory must outlive the execution.</p>
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * @param execution The execution to be modified.
   * @param index The index of the input argument we are setting. It is
   *              an index into the lists passed to
   *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is
   *              not the index associated with {@link
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
  int (*ANeuralNetworksExecution_setInputFromMemory)(
      ANeuralNetworksExecution* execution, int32_t index,
      const ANeuralNetworksOperandType* type,
      const ANeuralNetworksMemory* memory, size_t offset, size_t length);

  /**
   * Associate a user buffer with an output of the model of the
   * {@link ANeuralNetworksExecution}.
   *
   * <p>The provided buffer must outlive the execution.</p>
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * @param execution The execution to be modified.
   * @param index The index of the output argument we are setting. It is
   *              an index into the lists passed to
   *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is
   *              not the index associated with {@link
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
  int (*ANeuralNetworksExecution_setOutput)(
      ANeuralNetworksExecution* execution, int32_t index,
      const ANeuralNetworksOperandType* type, void* buffer, size_t length);

  /**
   * Associate part of a memory object with an output of the model of the
   * {@link ANeuralNetworksExecution}.
   *
   * <p>The provided memory must outlive the execution.</p>
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * @param execution The execution to be modified.
   * @param index The index of the output argument we are setting. It is
   *              an index into the lists passed to
   *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is
   *              not the index associated with {@link
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
  int (*ANeuralNetworksExecution_setOutputFromMemory)(
      ANeuralNetworksExecution* execution, int32_t index,
      const ANeuralNetworksOperandType* type,
      const ANeuralNetworksMemory* memory, size_t offset, size_t length);

  /**
   * Schedule evaluation of the execution.
   *
   * <p>Schedules evaluation of the execution. Once the model has been
   * applied and the outputs are ready to be consumed, the execution will be
   * signaled. Use {@link ANeuralNetworksExecution_wait} to wait for that
   * signal.
   * </p>
   *
   * Multiple executions can be scheduled and evaluated concurrently, and
   * compilations can be performed concurrently with executions. The runtime
   * makes no guarantee on the ordering of the completion of compilations and
   * executions. If it's important to the application, the application should
   * enforce the ordering by using {@link ANeuralNetworksCompilation_wait} and
   * {@link ANeuralNetworksExecution_wait}.
   *
   * ANeuralNetworksExecution_wait must be called to recuperate the resources
   * used by the execution.
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * @param execution The execution to be scheduled and executed.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksExecution_startCompute)(
      ANeuralNetworksExecution* execution, ANeuralNetworksEvent** event);

  /**
   * Waits until the execution completes.
   *
   * More than one thread can wait on an event. When the execution completes,
   * all threads will be released.
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * @return ANEURALNETWORKS_NO_ERROR if the execution completed normally.
   */
  int (*ANeuralNetworksEvent_wait)(ANeuralNetworksEvent* event);

  /**
   * Destroys the event.
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   */
  void (*ANeuralNetworksEvent_free)(ANeuralNetworksEvent* event);

  // ASharedMemory_create was added in Android 8.0, so safe to use with NNAPI
  // which was added in 8.1.
  int (*ASharedMemory_create)(const char* name, size_t size);

  /**
   * Get the number of available devices.
   *
   * @param numDevices Used to return the number of devices.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 29.
   */
  int (*ANeuralNetworks_getDeviceCount)(uint32_t* numDevices);

  /**
   * Get the representation of the specified device.
   *
   * @param devIndex The index of the specified device. Must be less than the
   *                 number of available devices.
   * @param device The representation of the specified device.
   *               The same representation will always be returned for the
   *               specified device.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 29.
   */

  int (*ANeuralNetworks_getDevice)(uint32_t devIndex,
                                   ANeuralNetworksDevice** device);

  /**
   * Get the name of the specified device.
   *
   * @param device The representation of the specified device.
   * @param name The returned name of the specified device. The name will be
   *             in UTF-8 and will be null-terminated. It will be recognizable
   *             as a known device name rather than a cryptic string. For
   *             devices with API level 29 and above, the format of the name is
   *             {VENDOR}-{DEVICE}, e.g. “google-ipu”. For devices with feature
   *             level 28 or lower, the name will always be “unknown-device”.
   *             The name will remain valid for the duration of the application.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 29.
   */
  int (*ANeuralNetworksDevice_getName)(const ANeuralNetworksDevice* device,
                                       const char** name);

  /**
   * Get the version of the driver implementation of the specified device.
   *
   * It’s the responsibility of the driver implementor to insure that this
   * version string uniquely distinguishes this implementation from all previous
   * implementations.
   *
   * This version string must not be confused with the feature level which is
   * solely defined by {@link ANeuralNetworksDevice_getFeatureLevel}. There is
   * no implicit ordering of the versions. For example, it is not possible to
   * filter all drivers older than a certain version.
   *
   * Application developers may use this version string to avoid or prefer
   * specific driver implementations. For example, an application may want to do
   * so because:
   *     - A specific version of the driver does not provide the required
   * performance, perhaps because of a performance regression.
   *     - A specific version of the driver has a bug or returns results that
   * don’t match the minimum precision requirement for the application.
   *
   * @param device  The representation of the specified device.
   * @param version The returned version string of the driver for the specified
   *                device. The string will be in UTF-8 and will be
   *                null-terminated. For devices with feature level 28 or lower,
   *                "UNKNOWN" will be returned. The version string will remain
   *                valid for the duration of the application.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 29.
   */
  int (*ANeuralNetworksDevice_getVersion)(const ANeuralNetworksDevice* device,
                                          const char** version);

  /**
   * Get the supported NNAPI version of the specified device.
   *
   * Each device has a supported feature level, which is the most advanced
   * feature this driver implements. For example, if the driver implements the
   * features introduced in Android P, but does not implement the features
   * introduced after Android P, the value would be 28. Developers could decide
   * whether or not the specified device should be used for a Model that has
   * certain feature requirements.
   *
   * @param device       The representation of the specified device.
   * @param featureLevel The API level of the most advanced feature this driver
   *                     implements.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 29.
   */
  int (*ANeuralNetworksDevice_getFeatureLevel)(
      const ANeuralNetworksDevice* device, int64_t* featureLevel);

  /**
   * Get the type of a given device.
   *
   * The device type can be used to help application developers to distribute
   * Machine Learning workloads and other workloads such as graphical rendering.
   * E.g., for an app which renders AR scenes based on real time object
   * detection results, the developer could choose an ACCELERATOR type device
   * for ML workloads, and reserve GPU for graphical rendering.
   *
   * @param device The representation of the specified device.
   * @param type The returned {@link DeviceTypeCode} of the specified device.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 29.
   */
  int (*ANeuralNetworksDevice_getType)(const ANeuralNetworksDevice* device,
                                       int32_t* type);

  /**
   * Get the supported operations for a specified set of devices. If multiple
   * devices are selected, the supported operation list is a union of supported
   * operations of all selected devices.
   *
   * @param model        The model to be queried.
   * @param devices      The set of devices. Must not contain duplicates.
   * @param numDevices   The number of devices in the set.
   * @param supportedOps The boolean array to be filled. True means supported.
   *                     The size of the boolean array must be at least as large
   *                     as the number of operations in the model. The order of
   *                     elements in the supportedOps array matches the order in
   *                     which the corresponding operations were added to the
   *                     model.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 29.
   */
  int (*ANeuralNetworksModel_getSupportedOperationsForDevices)(
      const ANeuralNetworksModel* model,
      const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
      bool* supportedOps);

  /**
   * Create a {@link ANeuralNetworksCompilation} to compile the given model for
   * a specified set of devices. If more than one device is specified, the
   * compilation will distribute the workload automatically across the devices.
   * The model must be fully supported by the specified set of devices. This
   * means that ANeuralNetworksModel_getSupportedOperationsForDevices() must
   * have returned true for every operation for that model/devices pair.
   *
   * @param model       The {@link ANeuralNetworksModel} to be compiled.
   * @param devices     The set of devices. Must not contain duplicates.
   * @param numDevices  The number of devices in the set.
   * @param compilation The newly created object or NULL if unsuccessful.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
   *         if the model is invalid.
   *
   * Available since API level 29.
   */
  int (*ANeuralNetworksCompilation_createForDevices)(
      ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices,
      uint32_t numDevices, ANeuralNetworksCompilation** compilation);

  /**
   * Sets the compilation caching signature and the cache directory.
   *
   * Provides optional caching information to the runtime for faster repeated
   * compilation.
   *
   * See {@link ANeuralNetworksCompilation} for information on multithreaded
   * usage.
   *
   * @param compilation The compilation to be modified.
   * @param cacheDir The cache directory to store and retrieve caching data. It
   *                 is recommended to use the code_cache provided by the
   *                 Android runtime. If not using the code_cache, the user
   *                 should choose a directory local to the application, and is
   *                 responsible to manage and clean the cache entries.
   * @param token The token provided by the user to specify a model, must be of
   *              length ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN. The user
   *              should ensure that the token is unique to a model within the
   *              application. The NNAPI runtime will not detected token
   *              collisions. If there is a collision, the compilation outcome
   *              may be incorrect without notifying with error.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 29.
   */
  int (*ANeuralNetworksCompilation_setCaching)(
      ANeuralNetworksCompilation* compilation, const char* cacheDir,
      const uint8_t* token);

  /**
   * Set the maximum expected duration for compiling the model.
   *
   * If the device is not able to complete the compilation within the specified
   * duration, the compilation may be aborted. The timeout duration begins at
   * the call to {@link ANeuralNetworksCompilation_finish}.
   *
   * This timeout duration acts as a hint to drivers, and can be used to both
   * free up compute resources within the driver and return control back to the
   * application quicker than is possible without the hint. It enables drivers
   * that are able to estimate how long a compilation will take to abort the
   * compilation before it has even started if the driver believes the
   * compilation cannot be completed within the timeout duration. Similarly, it
   * enables drivers to abort an ongoing compilation if it is taking too long.
   * However, this call does not guarantee that the compilation will complete or
   * abort within the timeout duration.
   *
   * By default (i.e., unless ANeuralNetworksCompilation_setTimeout is called),
   * the timeout duration for compiling the model is considered infinite.
   *
   * The {@link ANeuralNetworksCompilation} must have been created with
   * {@link ANeuralNetworksCompilation_createForDevices} with numDevices = 1,
   * otherwise this function will fail with ANEURALNETWORKS_BAD_DATA. If the
   * device has a feature level reported by
   * {@link ANeuralNetworksDevice_getFeatureLevel} that is lower than 30, then
   * the timeout duration hint will be ignored.
   *
   * See {@link ANeuralNetworksCompilation} for information on multithreaded
   * usage.
   *
   * @param compilation The compilation to be modified.
   * @param duration The maximum amount of time in nanoseconds that is expected
   * to be spent finishing a compilation. If this duration is exceeded, the
   *     compilation may be aborted. If set to 0, the timeout duration is
   *     considered infinite.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 30.
   */
  int (*ANeuralNetworksCompilation_setTimeout)(
      ANeuralNetworksCompilation* compilation, uint64_t duration);

  /**
   * Set the execution priority.
   *
   * Execution priorities are relative to other executions created by the same
   * application (specifically same uid) for the same device. Specifically,
   * priorities of executions from one application will not affect executions
   * from another application. Similarly, priorities of executions on one device
   * will not affect executions on another device.
   *
   * Higher priority executions may use more compute resources than lower
   * priority executions, and may preempt or starve lower priority executions.
   *
   * See {@link ANeuralNetworksCompilation} for information on multithreaded
   * usage.
   *
   * Available since API level 30.
   *
   * @param compilation The compilation to be modified.
   * @param priority The relative priority of the execution compared to other
   *     executions created by the application. Must be one of
   *     ANEURALNETWORKS_PRIORITY_*.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksCompilation_setPriority)(
      ANeuralNetworksCompilation* compilation, int priority);

  /**
   * Schedule synchronous evaluation of the execution.
   *
   * <p>Schedules synchronous evaluation of the execution. Returns once the
   * execution has completed and the outputs are ready to be consumed.
   * </p>
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * See {@link ANeuralNetworksExecution_startCompute} for asynchronous
   * execution. Synchronous execution incurs lower overhead than asynchronous
   * execution.
   *
   * Available since API level 29.
   *
   * @param execution The execution to be scheduled and executed.
   *
   * @return ANEURALNETWORKS_NO_ERROR if the execution completed normally.
   *         ANEURALNETWORKS_UNMAPPABLE if the execution input or output memory
   *         cannot be properly mapped.
   */
  int (*ANeuralNetworksExecution_compute)(ANeuralNetworksExecution* execution);

  /**
   * Set the maximum expected duration of the specified execution.
   *
   * If the device is not able to complete the execution within the specified
   * duration, the execution may be aborted. The timeout duration begins at a
   * call to one of:
   * - {@link ANeuralNetworksExecution_burstCompute}
   * - {@link ANeuralNetworksExecution_compute}
   * - {@link ANeuralNetworksExecution_startCompute}
   * - {@link ANeuralNetworksExecution_startComputeWithDependencies}
   *
   * This timeout duration acts as a hint to drivers, and can be used to both
   * free up compute resources within the driver and return control back to the
   * application quicker than is possible without the hint. It enables drivers
   * that are able to estimate how long an execution will take to abort the
   * execution before it has even started if the driver believes the execution
   * cannot be completed within the timeout duration. Similarly, it enables
   * drivers to abort an ongoing execution if it is taking too long. However,
   * this call does not guarantee that the execution will complete or abort
   * within the timeout duration.
   *
   * By default (i.e., unless ANeuralNetworksExecution_setTimeout is called),
   * the timeout duration for execution is considered infinite.
   *
   * The {@link ANeuralNetworksExecution} must have been created from an
   * {@link ANeuralNetworksCompilation} which in turn was created from
   * {@link ANeuralNetworksCompilation_createForDevices} with numDevices = 1,
   * otherwise this function will fail with ANEURALNETWORKS_BAD_DATA. If the
   * device has a feature level reported by
   * {@link ANeuralNetworksDevice_getFeatureLevel} that is lower than 30, then
   * the timeout duration hint will be ignored.
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * @param execution The execution to be modified.
   * @param duration The maximum amount of time in nanoseconds that is expected
   * to be spent executing a model. If this duration is exceeded, the execution
   *     may be aborted. If set to 0, the timeout duration is considered
   * infinite.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 30.
   */
  int (*ANeuralNetworksExecution_setTimeout)(
      ANeuralNetworksExecution* execution, uint64_t duration);

  /**
   * Set the maximum duration of WHILE loops in the specified execution.
   *
   * This is a fuzzy per-loop timeout intended to prevent infinite loops.
   *
   * If a WHILE loop condition model does not output false within the specified
   * duration, the execution will be aborted.
   *
   * See {@link ANeuralNetworks_getDefaultLoopTimeout} and
   * {@link ANeuralNetworks_getMaximumLoopTimeout} for the default
   * and maximum timeout values.
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * @param execution The execution to be modified.
   * @param duration The maximum amount of time in nanoseconds that can be spent
   *     executing a WHILE loop. If the specified duration value exceeds the
   * value produced by {@link ANeuralNetworks_getMaximumLoopTimeout}, it will be
   *     overridden by that value.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *         ANEURALNETWORKS_BAD_STATE if execution has started.
   *         ANEURALNETWORKS_UNEXPECTED_NULL if execution is NULL.
   *
   * Available since API level 30.
   */
  int (*ANeuralNetworksExecution_setLoopTimeout)(
      ANeuralNetworksExecution* execution, uint64_t duration);

  /**
   * Get the dimensional information of the specified output operand of the
   * model of the
   * {@link ANeuralNetworksExecution}.
   *
   * On asynchronous execution initiated by {@link
   * ANeuralNetworksExecution_startCompute},
   * {@link ANeuralNetworksEvent_wait} must be called prior to this function to
   * recuperate the resources used by the execution.
   *
   * @param execution The execution to be queried.
   * @param index The index of the output argument we are querying. It is
   *              an index into the lists passed to
   *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is
   *              not the index associated with
   *              {@link ANeuralNetworksModel_addOperand}.
   * @param rank The rank of the output operand.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful,
   *         ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE if the target output is
   *         provided an insufficient buffer at execution time,
   *         ANEURALNETWORKS_BAD_DATA if the index is invalid.
   *
   * Available since API level 29.
   */
  int (*ANeuralNetworksExecution_getOutputOperandRank)(
      ANeuralNetworksExecution* execution, int32_t index, uint32_t* rank);

  /**
   * Get the dimensional information of the specified output operand of the
   * model of the
   * {@link ANeuralNetworksExecution}. The target output operand cannot be a
   * scalar.
   *
   * On asynchronous execution initiated by {@link
   * ANeuralNetworksExecution_startCompute},
   * {@link ANeuralNetworksEvent_wait} must be called prior to this function to
   * recuperate the resources used by the execution.
   *
   * @param execution The execution to be queried.
   * @param index The index of the output argument we are querying. It is an
   *              index into the lists passed to
   *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is
   *              not the index associated with
   *              {@link ANeuralNetworksModel_addOperand}.
   * @param dimensions The dimension array to be filled. The size of the array
   *                   must be exactly as large as the rank of the output
   *                   operand to be queried in the model.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful,
   *         ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE if the target output is
   *         provided an insufficient buffer at execution time,
   *         ANEURALNETWORKS_BAD_DATA if the index is invalid or if the target
   *         is a scalar.
   *
   * Available since API level 29.
   */
  int (*ANeuralNetworksExecution_getOutputOperandDimensions)(
      ANeuralNetworksExecution* execution, int32_t index, uint32_t* dimensions);

  /**
   * Create a {@link ANeuralNetworksBurst} to apply the given compilation.
   * This only creates the burst object. Computation is only performed once
   * {@link ANeuralNetworksExecution_burstCompute} is invoked with a valid
   * {@link ANeuralNetworksExecution} and {@link ANeuralNetworksBurst}.
   *
   * <p>The provided compilation must outlive the burst object.</p>
   *
   * Available since API level 29.
   *
   * @param compilation The {@link ANeuralNetworksCompilation} to be evaluated.
   * @param burst The newly created object or NULL if unsuccessful.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
   *         if the compilation is invalid.
   */
  int (*ANeuralNetworksBurst_create)(ANeuralNetworksCompilation* compilation,
                                     ANeuralNetworksBurst** burst);

  /**
   * Destroys the burst object.
   *
   * Available since API level 29.
   *
   * @param burst The burst object to be destroyed. Passing NULL is acceptable
   * and results in no operation.
   */
  void (*ANeuralNetworksBurst_free)(ANeuralNetworksBurst* burst);

  /**
   * Schedule synchronous evaluation of the execution on a burst object.
   *
   * <p>Schedules synchronous evaluation of the execution. Returns once the
   * execution has completed and the outputs are ready to be consumed.</p>
   *
   * <p>There must be at most one {@link ANeuralNetworksExecution} processing at
   * any given time for any given burst object. Any
   * {@link ANeuralNetworksExecution} launched before the previous has finished
   * will result in ANEURALNETWORKS_BAD_STATE.</p>
   *
   * Available since API level 29.
   *
   * @param burst The burst object to execute on.
   * @param execution The execution to be scheduled and executed. The execution
   *                  must be created from the same {@link
   *                  ANeuralNetworksCompilation} as the burst object.
   *
   * @return ANEURALNETWORKS_NO_ERROR if the execution completed normally.
   */
  int (*ANeuralNetworksExecution_burstCompute)(
      ANeuralNetworksExecution* execution, ANeuralNetworksBurst* burst);

  /**
   * Creates a shared memory object from an AHardwareBuffer handle.
   *
   * If the shared memory is backed by an AHardwareBuffer of
   * AHARDWAREBUFFER_FORMAT_BLOB format, it can be used the same way as
   * shared memory created from a file handle. See
   * {@link ANeuralNetworksMemory} for a description on how to use this
   * shared memory.
   *
   * If the shared memory is backed by an AHardwareBuffer of a format other
   * than AHARDWAREBUFFER_FORMAT_BLOB, it can only be used for Model inputs
   * and outputs. When calling
   * {@link ANeuralNetworksExecution_setInputFromMemory} or
   * {@link ANeuralNetworksExecution_setOutputFromMemory} with the shared
   * memory, both offset and length must be set to zero and the entire
   * memory region will be associated with the specified input or output
   * operand. There is no guarantee that an arbitrary AHardwareBuffer_Format
   * and AHardwareBuffer_UsageFlags combination can be used by arbitrary
   * devices. The execution will fail if selected set of devices cannot
   * consume the buffer.
   *
   * Calling {@link ANeuralNetworksModel_setOperandValueFromMemory} with
   * shared memory backed by an AHardwareBuffer of a format other than
   * AHARDWAREBUFFER_FORMAT_BLOB is disallowed.
   *
   * TODO(miaowang): add documentation about intended usage with
   * introspection API.
   *
   * Available since API level 29.
   *
   * @param ahwb The AHardwareBuffer handle.
   * @param memory The memory object to be created.
   *               Set to NULL if unsuccessful.
   *
   * @return ANEURALNETWORKS_NO_ERROR if the request completed normally.
   *
   * @see AHardwareBuffer
   */
  int (*ANeuralNetworksMemory_createFromAHardwareBuffer)(
      const AHardwareBuffer* ahwb, ANeuralNetworksMemory** memory);

  /**
   * Specifies whether duration of the {@link ANeuralNetworksExecution} is to be
   * measured. By default, duration is not measured.
   *
   * The {@link ANeuralNetworksExecution} must have been created with
   * {@link ANeuralNetworksCompilation_createForDevices} with numDevices = 1.
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * Available since API level 29.
   *
   * @param execution The execution to be modified.
   * @param measure 'true' if duration is to be measured, 'false' if not.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksExecution_setMeasureTiming)(
      ANeuralNetworksExecution* execution, bool measure);

  /**
   * Get the time spent in the specified {@link ANeuralNetworksExecution}, in
   * nanoseconds. The execution must have completed.
   *
   * @param execution The execution to be queried.
   * @param durationCode The measurement to be queried, specified by {@link
   * DurationCode}.
   * @param duration The returned duration. If no measurement was requested by
   *                 {@link ANeuralNetworksExecution_setMeasureTiming}, or for
   * some other reason the duration is not available, UINT64_MAX will be
   * returned. A particular device need not support any given measurement.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksExecution_getDuration)(
      const ANeuralNetworksExecution* execution, int32_t durationCode,
      uint64_t* duration);

  /**
   * Queries whether an extension is supported by the driver implementation of
   * the specified device.
   *
   * @param device The representation of the specified device.
   * @param extension The extension name.
   * @param isExtensionSupported The boolean value indicating whether the
   * extension is supported.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 29.
   */
  int (*ANeuralNetworksDevice_getExtensionSupport)(
      const ANeuralNetworksDevice* device, const char* extensionName,
      bool* isExtensionSupported);

  /**
   * Creates an operand type from an extension name and an extension operand
   * code.
   *
   * See {@link ANeuralNetworksModel} for information on multithreaded usage.
   *
   * Available since API level 29.
   *
   * @param model The model to contain the operand.
   * @param extensionName The extension name.
   * @param operandCodeWithinExtension The extension operand code.
   * @param type The operand type.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksModel_getExtensionOperandType)(
      ANeuralNetworksModel* model, const char* extensionName,
      uint16_t operandCodeWithinExtension, int32_t* type);

  /**
   * Creates an operation type from an extension name and an extension operation
   * code.
   *
   * See {@link ANeuralNetworksModel} for information on multithreaded usage.
   *
   * Available since API level 29.
   *
   * @param model The model to contain the operation.
   * @param extensionName The extension name.
   * @param operationCodeWithinExtension The extension operation code.
   * @param type The operation type.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksModel_getExtensionOperationType)(
      ANeuralNetworksModel* model, const char* extensionName,
      uint16_t operationCodeWithinExtension,
      ANeuralNetworksOperationType* type);

  /**
   * Sets extension operand parameters.
   *
   * Available since API level 29.
   *
   * @param model The model to be modified.
   * @param index The index of the model operand we're setting.
   * @param data A pointer to the extension operand data.
   *             The data does not have to outlive the call to this function.
   * @param length The size in bytes of the data value.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksModel_setOperandExtensionData)(
      ANeuralNetworksModel* model, int32_t index, const void* data,
      size_t length);

  /**
   * Create a {@link ANeuralNetworksMemoryDesc} with no properties.
   *
   * This only creates the memory descriptor. Its properties should be set with
   * calls to
   * {@link ANeuralNetworksMemoryDesc_addInputRole},
   * {@link ANeuralNetworksMemoryDesc_addOutputRole}, and
   * {@link ANeuralNetworksMemoryDesc_setDimensions}.
   *
   * {@link ANeuralNetworksMemoryDesc_finish} must be called once all properties
   * have been set.
   *
   * {@link ANeuralNetworksMemoryDesc_free} must be called once the memory
   * descriptor is no longer needed.
   *
   * Available since API level 30.
   *
   * @param desc The {@link ANeuralNetworksMemoryDesc} to be created.
   *             Set to NULL if unsuccessful.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksMemoryDesc_create)(ANeuralNetworksMemoryDesc** desc);

  /**
   * Destroy a memory descriptor.
   *
   * The memory descriptor need not have been finished by a call to
   * {@link ANeuralNetworksMemoryDesc_finish}.
   *
   * See {@link ANeuralNetworksMemoryDesc} for information on multithreaded
   * usage.
   *
   * Available since API level 30.
   *
   * @param desc The memory descriptor to be destroyed. Passing NULL is
   * acceptable and results in no operation.
   */
  void (*ANeuralNetworksMemoryDesc_free)(ANeuralNetworksMemoryDesc* desc);

  /**
   * Specify that a memory object will be playing the role of an input to an
   * execution created from a particular compilation.
   *
   * The compilation and the input index fully specify an input operand. This
   * function may be invoked multiple times on the same memory descriptor with
   * different input operands, and the same input operand may be specified on
   * multiple memory descriptors. However, specifying the same input operand on
   * the same memory descriptor more than once will return an error.
   *
   * The dimensions of the corresponding model operands of all the roles
   * specified by
   * {@link ANeuralNetworksMemoryDesc_addInputRole} and
   * {@link ANeuralNetworksMemoryDesc_addOutputRole} must be compatible with
   * each other. Two dimensions are incompatible if both ranks are fully
   * specified but have different values, or if there is at least one axis that
   * is fully specified in both but has different values.
   *
   * At least one of {@link ANeuralNetworksMemoryDesc_addInputRole} and
   * {@link ANeuralNetworksMemoryDesc_addOutputRole} must be called on a memory
   * descriptor before invoking {@link ANeuralNetworksMemoryDesc_finish}.
   *
   * Attempting to modify a memory descriptor once
   * {@link ANeuralNetworksMemoryDesc_finish} has been called will return an
   * error.
   *
   * See {@link ANeuralNetworksMemoryDesc} for information on multithreaded
   * usage.
   *
   * Available since API level 30.
   *
   * @param desc The memory descriptor to be modified.
   * @param compilation The compilation object. It must already have been
   * finished by calling {@link ANeuralNetworksCompilation_finish}, and must
   * outlive the memory descriptor.
   * @param index The index of the input argument we are referencing from the
   * compilation. It is an index into the inputs list passed to
   *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is
   * not the index associated with {@link ANeuralNetworksModel_addOperand}.
   * @param frequency A floating-point value within the range (0.0, 1.0].
   * Describes how likely the memory is to be used in the specified role. This
   * is provided as a hint to optimize the case when different roles prefer
   * different memory locations or data layouts.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksMemoryDesc_addInputRole)(
      ANeuralNetworksMemoryDesc* desc,
      const ANeuralNetworksCompilation* compilation, uint32_t index,
      float frequency);

  /**
   * Specify that a memory object will be playing the role of an output to an
   * execution created from a particular compilation.
   *
   * The compilation and the output index fully specify an output operand. This
   * function may be invoked multiple times on the same memory descriptor with
   * different output operands, and the same output operand may be specified on
   * multiple memory descriptors. However, specifying the same output operand on
   * the same memory descriptor object more than once will return an error.
   *
   * The dimensions of the corresponding model operands of all the roles
   * specified by
   * {@link ANeuralNetworksMemoryDesc_addInputRole} and
   * {@link ANeuralNetworksMemoryDesc_addOutputRole} must be compatible with
   * each other. Two dimensions are incompatible if both ranks are fully
   * specified but have different values, or if there is at least one axis that
   * is fully specified in both but has different values.
   *
   * At least one of {@link ANeuralNetworksMemoryDesc_addInputRole} and
   * {@link ANeuralNetworksMemoryDesc_addOutputRole} must be called on the
   * memory descriptor before invoking {@link ANeuralNetworksMemoryDesc_finish}.
   *
   * Attempting to modify a memory descriptor once
   * {@link ANeuralNetworksMemoryDesc_finish} has been called will return an
   * error.
   *
   * See {@link ANeuralNetworksMemoryDesc} for information on multithreaded
   * usage.
   *
   * Available since API level 30.
   *
   * @param desc The memory descriptor to be modified.
   * @param compilation The compilation object. It must already have been
   * finished by calling {@link ANeuralNetworksCompilation_finish}, and must
   * outlive the memory descriptor.
   * @param index The index of the output argument we are referencing from the
   *              compilation. It is an index into the outputs list passed to
   *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is
   * not the index associated with {@link ANeuralNetworksModel_addOperand}.
   * @param frequency A floating-point value within the range (0.0, 1.0].
   * Describes how likely the memory is to be used in the specified role. This
   * is provided as a hint to optimize the case when multiple roles prefer
   * different memory locations or data layouts.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksMemoryDesc_addOutputRole)(
      ANeuralNetworksMemoryDesc* desc,
      const ANeuralNetworksCompilation* compilation, uint32_t index,
      float frequency);

  /**
   * Set the dimensional information of the memory descriptor.
   *
   * The specified dimensions must be compatible with the dimensions of the
   * corresponding model operands of all the roles specified by
   * {@link ANeuralNetworksMemoryDesc_addInputRole} and
   * {@link ANeuralNetworksMemoryDesc_addOutputRole}. Two dimensions are
   * incompatible if both ranks are fully specified but have different values,
   * or if there is at least one axis that is fully specified in both but has
   * different values.
   *
   * Attempting to modify a memory descriptor once
   * {@link ANeuralNetworksMemoryDesc_finish} has been called will return an
   * error.
   *
   * See {@link ANeuralNetworksMemoryDesc} for information on multithreaded
   * usage.
   *
   * Available since API level 30.
   *
   * @param desc The memory descriptor to be modified.
   * @param rank The number of dimensions. Must be 0 for scalars.
   * @param dimensions An array of dimensions. An entry with the value 0
   * indicates that the corresponding axis has an unknown size.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksMemoryDesc_setDimensions)(
      ANeuralNetworksMemoryDesc* desc, uint32_t rank,
      const uint32_t* dimensions);

  /**
   * Indicate that we have finished modifying a memory descriptor. Required
   * before calling
   * {@link ANeuralNetworksMemory_createFromDesc}.
   *
   * This function must only be called once for a given memory descriptor.
   *
   * See {@link ANeuralNetworksMemoryDesc} for information on multithreaded
   * usage.
   *
   * Available since API level 30.
   *
   * @param desc The memory descriptor to be finished.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksMemoryDesc_finish)(ANeuralNetworksMemoryDesc* desc);

  /**
   * Creates a memory object from a memory descriptor.
   *
   * The memory object is created with an uninitialized buffer. A memory object
   * with an uninitialized buffer may only be used according to the roles
   * specified by
   * {@link ANeuralNetworksMemoryDesc_addOutputRole}, or as the destination
   * memory in
   * {@link ANeuralNetworksMemory_copy}. The buffer of a memory object is
   * initialized after the memory object is used as an output in a successful
   * execution, or used as the destination memory in a successful {@link
   * ANeuralNetworksMemory_copy}. A memory object with an initialized buffer may
   * be used according to all roles specified in
   * {@link ANeuralNetworksMemoryDesc}, or as the source or destination memory
   * in
   * {@link ANeuralNetworksMemory_copy}. The buffer of a memory object will
   * return to the uninitialized state if the memory object is used as an output
   * in a failed execution, or used as the destination memory in a failed {@link
   * ANeuralNetworksMemory_copy}.
   *
   * The dimensions of the memory descriptor are deduced from the dimensions of
   * the corresponding model operands of all the roles specified by
   * {@link ANeuralNetworksMemoryDesc_addInputRole} and
   * {@link ANeuralNetworksMemoryDesc_addOutputRole}, as well as the dimensions
   * set by the call to {@link ANeuralNetworksMemoryDesc_setDimensions}, if any.
   * The memory descriptor may have unspecified dimensions or rank. In such a
   * case, the same memory object may be used with different shapes of outputs
   * in different executions. When the memory is used as an input, the input
   * shape must be the same as the output shape from the last execution using
   * this memory object as an output, or the last
   * {@link ANeuralNetworkMemory_copy} using this memory object as the
   * destination memory. Creating a memory object with unspecified dimensions or
   * rank may fail for certain sets of roles.
   *
   * Using the memory in roles or shapes that are not compatible with the rules
   * specified above will return an error.
   *
   * When calling {@link ANeuralNetworksExecution_setInputFromMemory} or
   * {@link ANeuralNetworksExecution_setOutputFromMemory} with the memory
   * object, both offset and length must be set to zero and the entire memory
   * region will be associated with the specified input or output operand.
   *
   * Calling {@link ANeuralNetworksModel_setOperandValueFromMemory} with the
   * memory created from this function will return an error.
   *
   * {@link ANeuralNetworksMemory_free} must be called once the memory is no
   * longer needed.
   *
   * Attempting to create memory from an unfinished memory descriptor will
   * return an error.
   *
   * The provided {@link ANeuralNetworksMemoryDesc} need not outlive the
   * {@link ANeuralNetworksMemory} object.
   *
   * Available since API level 30.
   *
   * @param desc The memory descriptor.
   * @param memory The memory object to be created.
   *               Set to NULL if unsuccessful.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful; ANEURALNETWORKS_OP_FAILED
   * if the memory is created with unspecified dimensions or rank and it is not
   * supported for this set of roles.
   */
  int (*ANeuralNetworksMemory_createFromDesc)(
      const ANeuralNetworksMemoryDesc* desc, ANeuralNetworksMemory** memory);

  /**
   * Copies data from one memory object to another.
   *
   * If at most one of the src and dst is created from
   * {@link ANeuralNetworksMemory_createFromDesc}, the src and dst must have the
   * same logical size:
   * - If the memory is created from {@link ANeuralNetworksMemory_createFromFd},
   * or if it is created from {@link
   * ANeuralNetworksMemory_createFromAHardwareBuffer} with format of
   * AHARDWAREBUFFER_FORMAT_BLOB, the logical size equals the size of the
   * memory.
   * - If the memory is created from
   *   {@link ANeuralNetworksMemory_createFromAHardwareBuffer} with a format
   * other than AHARDWAREBUFFER_FORMAT_BLOB, the logical size equals the size
   * when there is no padding and the data is tightly packed. This function may
   * fail if the AHardwareBuffer cannot be accessed.
   * - If the memory is created from {@link
   * ANeuralNetworksMemory_createFromDesc}, the logical size equals the size
   * indicated by the {@link OperandCode} multiplied by the number of elements.
   * This function will fail if the number of elements is unknown.
   *
   * If both src and dst are created from {@link
   * ANeuralNetworksMemory_createFromDesc}, they must have compatible
   * dimensions. Two dimensions are incompatible if both ranks are fully
   * specified but have different values, or if there is at least one axis that
   * is fully specified in both but has different values. The dst may have
   * unspecified dimensions or rank. In such a case, the dimensions of dst will
   * get updated according to the dimensions of the src.
   *
   * In both cases, if the src is created from
   * {@link ANeuralNetworksMemory_createFromDesc}, it must have been used as an
   * output in a successful execution, or used as the destination memory in a
   * successful
   * {@link ANeuralNetworksMemory_copy}.
   *
   * The src and dst may have different data layout, in which case the data
   * copying is performed logically with data layout transformation.
   *
   * Available since API level 30.
   *
   * @param src The source memory object.
   * @param dst The destination memory object.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   */
  int (*ANeuralNetworksMemory_copy)(const ANeuralNetworksMemory* src,
                                    const ANeuralNetworksMemory* dst);

  /**
   * Create a {@link ANeuralNetworksEvent} from a sync_fence file descriptor.
   *
   * The newly created ANeuralNetworksEvent does not take ownership of the
   * provided sync_fence_fd, it will instead dup the provided sync_fence_fd and
   * own the duplicate.
   *
   * @param sync_fence_fd The sync_fence file descriptor.
   * @param event The newly created object or NULL if unsuccessful.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 30.
   */
  int (*ANeuralNetworksEvent_createFromSyncFenceFd)(
      int sync_fence_fd, ANeuralNetworksEvent** event);

  /**
   * Get sync_fence file descriptor from the event.
   *
   * If the ANeuralNetworksEvent is not backed by a sync fence, the
   * sync_fence_fd will be set to -1, and ANEURALNETWORKS_BAD_DATA will be
   * returned.
   *
   * See {@link ANeuralNetworksEvent_createFromSyncFenceFd} and
   * {@link ANeuralNetworksExecution_startComputeWithDependencies} to see how to
   * create an event backed by a sync fence.
   *
   * The user takes ownership of the returned fd, and must close the returned
   * file descriptor when it is no longer needed.
   *
   * @param event An event that is backed by a sync fence.
   * @param sync_fence_fd The sync_fence file descriptor. The file descriptor
   * will be set to -1 if there is an error.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *
   * Available since API level 30.
   */
  int (*ANeuralNetworksEvent_getSyncFenceFd)(const ANeuralNetworksEvent* event,
                                             int* sync_fence_fd);

  /**
   * Schedule asynchronous evaluation of the execution with dependencies.
   *
   * The execution will wait for all the depending events to be signaled before
   * starting the evaluation. Once the execution has completed and the outputs
   * are ready to be consumed, the returned event will be signaled. Depending on
   * which devices are handling the execution, the event could be backed by a
   * sync fence. Use {@link ANeuralNetworksEvent_wait} to wait for that event.
   *
   * ANeuralNetworksEvent_wait must be called to recurperate the resources used
   * by the execution.
   *
   * If parts of the execution are scheduled on devices that do not support
   * fenced execution, the function call may wait for such parts to finish
   * before returning.
   *
   * The function will return an error if any of the events in dependencies is
   * already in a bad state. After the execution is scheduled, if any of the
   * events in dependencies does not complete normally, the execution will fail,
   * and {@link ANeuralNetworksEvent_wait} on the returned event will return an
   * error.
   *
   * The function will return an error if any of the execution outputs has a
   * tensor operand type that is not fully specified.
   *
   * The function can be passed a timeout duration in nanoseconds. This timeout
   * duration acts as a hint to drivers in the same way that the timeout
   * durations in {@link ANeuralNetworksCompilation_setTimeout} and {@link
   * ANeuralNetworksExecution_setTimeout} act as hints to drivers. The duration
   * begins when all waitFor sync fences have been signaled, and can be used
   * together with {@link ANeuralNetworksExecution_setTimeout} which specifies
   * the maximum timeout duration beginning at the call to
   * {@link ANeuralNetworksExecution_startComputeWithDependencies}.
   * If the duration is non-zero, the {@link ANeuralNetworksExecution} must have
   * been created from an {@link ANeuralNetworksCompilation} which in turn was
   * created from
   * {@link ANeuralNetworksCompilation_createForDevices} with numDevices = 1,
   * otherwise this function will fail with ANEURALNETWORKS_BAD_DATA. If either
   * the timeout duration from {@link ANeuralNetworksExecution_setTimeout} or
   * the timeout duration passed to this call is exceeded, the execution may be
   * aborted, in which case {@link ANEURALNETWORKS_MISSED_DEADLINE_*} will be
   * returned through {@link
   * ANeuralNetworksExecution_startComputeWithDependencies} or {@link
   * ANeuralNetworksEvent_wait} on the event object. If the device has a feature
   * level reported by {@link ANeuralNetworksDevice_getFeatureLevel} that is
   * lower than 30, then the timeout duration hints will be ignored.
   *
   * If this execution contains a {@link ANEURALNETWORKS_WHILE} operation, and
   * the condition model does not output false within the loop timeout duration,
   * then execution will be aborted and {@link
   * ANEURALNETWORKS_MISSED_DEADLINE_*} will be returned through {@link
   * ANeuralNetworksEvent_wait} on the event object.
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * See {@link ANeuralNetworksExecution_compute} for synchronous execution.
   * See {@link ANeuralNetworksExecution_burstCompute} for burst synchronous
   * execution. See {@link ANeuralNetworksExecution_startCompute} for regular
   * asynchronous execution.
   *
   * @param execution The execution to be scheduled and executed.
   * @param dependencies A set of depending events. The actual evaluation will
   * not start until all the events are signaled.
   * @param num_dependencies The number of events in the dependencies set.
   * @param duration The maximum amount of time in nanoseconds that is expected
   * to be spent executing the model after all dependencies are signaled. If set
   * to 0, the timeout duration is considered infinite.
   * @param event The event that will be signaled on completion. event is set to
   *              NULL if there's an error.
   *
   * @return ANEURALNETWORKS_NO_ERROR if the evaluation is successfully
   * scheduled.
   *
   * Available since API level 30.
   */
  int (*ANeuralNetworksExecution_startComputeWithDependencies)(
      ANeuralNetworksExecution* execution,
      const ANeuralNetworksEvent* const* dependencies,
      uint32_t num_dependencies, uint64_t duration,
      ANeuralNetworksEvent** event);

  /**
   * Specifies whether the {@link ANeuralNetworksExecution} is able to accept
   * padded input and output buffers and memory objects.
   *
   * By default, the input and output buffers and memory objects of {@link
   * ANeuralNetworksExecution} do not allow padding.
   *
   * Setting the execution to accept padded input and output buffers and memory
   * objects enables the length argument of {@link
   * ANeuralNetworksExecution_setInput},
   * {@link ANeuralNetworksExecution_setInputFromMemory}, {@link
   * ANeuralNetworksExecution_setOutput}, and {@link
   * ANeuralNetworksExecution_setOutputFromMemory} to be greater than the raw
   * size of the operand (i.e. the size of an element multiplied by the number
   * of elements). The extra bytes at the end of the buffer or memory region may
   * be used by the driver to access data in chunks, for efficiency.
   *
   * This method must not be called after {@link
   * ANeuralNetworksExecution_setInput},
   * {@link ANeuralNetworksExecution_setInputFromMemory}, {@link
   * ANeuralNetworksExecution_setOutput}, or {@link
   * ANeuralNetworksExecution_setOutputFromMemory}.
   *
   * See {@link ANeuralNetworksExecution} for information on multithreaded
   * usage.
   *
   * @param execution The execution to be modified.
   * @param enable 'true' if the execution is to be able to accept padded input
   * and output buffers and memory objects, 'false' if not.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *         ANEURALNETWORKS_UNEXPECTED_NULL if execution is NULL.
   *         ANEURALNETWORKS_BAD_STATE if {@link
   * ANeuralNetworksExecution_setInput},
   *         {@link ANeuralNetworksExecution_setInputFromMemory},
   *         {@link ANeuralNetworksExecution_setOutput}, or
   *         {@link ANeuralNetworksExecution_setOutputFromMemory} has been
   * called on the execution.
   *
   * Available since API level 31.
   */
  int (*ANeuralNetworksExecution_enableInputAndOutputPadding)(
      ANeuralNetworksExecution* execution, bool enable);

  /**
   * Specifies whether the {@link ANeuralNetworksExecution} can be reused for
   * multiple computations.
   *
   * By default, the {@link ANeuralNetworksExecution} is not reusable.
   *
   * Setting the execution to be reusable enables multiple computations to be
   * scheduled and evaluated on the same execution sequentially, either by means
   * of
   * {@link ANeuralNetworksExecution_burstCompute}, {@link
   * ANeuralNetworksExecution_compute},
   * {@link ANeuralNetworksExecution_startCompute} or
   * {@link ANeuralNetworksExecution_startComputeWithDependencies}.
   *
   * This function may only be invoked when the execution is in the preparation
   * state.
   *
   * See {@link ANeuralNetworksExecution} for information on execution states
   * and multithreaded usage.
   *
   * @param execution The execution to be modified.
   * @param reusable 'true' if the execution is to be reusable, 'false' if not.
   *
   * @return ANEURALNETWORKS_NO_ERROR if successful.
   *         ANEURALNETWORKS_UNEXPECTED_NULL if execution is NULL.
   *         ANEURALNETWORKS_BAD_STATE if the execution is not in the
   * preparation state.
   *
   * Available since API level 31.
   */
  int (*ANeuralNetworksExecution_setReusable)(
      ANeuralNetworksExecution* execution, bool reusable);

  /**
   * Get the NNAPI runtime feature level.
   *
   * Since API level 31 (NNAPI feature level 5), the NNAPI runtime
   * (libneuralnetworks.so) and its API specification can be updated between
   * Android API releases.
   *
   * On Android devices with API level 31 and newer, for NNAPI runtime feature
   * discovery, the NNAPI runtime feature level must be used instead of the
   * Android device API level.
   *
   * On Android devices with API level 30 and older, the Android API level of
   * the Android device must be used for NNAPI runtime feature discovery. Enum
   * values in
   * {@link FeatureLevelCode} from feature level 1 to 5 have their corresponding
   * Android API levels listed in their documentation, and each such enum value
   * equals the corresponding API level. This allows using the Android API level
   * as the feature level. This mapping between enum value and Android API level
   * does not exist for feature levels after NNAPI feature level 5 and API
   * levels after S (31).
   *
   * Example usage:
   * int device_api_level = android_get_device_api_level();
   * int64_t runtime_feature_level = (device_api_level < __ANDROID_API_S__) ?
   *                                  device_api_level :
   * ANeuralNetworks_getRuntimeFeatureLevel();
   *
   * Runtime feature level is closely related to NNAPI device feature level
   * ({@link ANeuralNetworksDevice_getFeatureLevel}), which indicates an NNAPI
   * device feature level (the most advanced NNAPI specification and features
   * that the driver implements). This function expresses NNAPI runtime feature
   * level, which indicates the most advanced NNAPI specification and features
   * the runtime implements. An NNAPI device feature level is always less than
   * or equal to the runtime feature level.
   *
   * This function returns a {@link FeatureLevelCode} enum value,
   * which is the NNAPI specification version that this NNAPI runtime
   * implements. It is NOT an Android API level.
   *
   * Available since NNAPI feature level 5.
   */
  int64_t (*ANeuralNetworks_getRuntimeFeatureLevel)();

  /**
   * Gets the ID that identifies a single session of client interacting with
   * NNAPI runtime.
   *
   * @param diagnosticCompilationInfo The NNAPI diagnostic compilation info
   * object.
   * @return Session info id.
   */
  int32_t (*SL_ANeuralNetworksDiagnosticCompilationInfo_getSessionId)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * Gets NNAPI version.
   *
   * @param diagnosticCompilationInfo The NNAPI diagnostic compilation info
   * object.
   * @return NNAPI version.
   */
  int64_t (*SL_ANeuralNetworksDiagnosticCompilationInfo_getNnApiVersion)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * Gets the hash of the model architecture (without weights).
   *
   * @param diagnosticCompilationInfo The NNAPI diagnostic compilation info
   * object.
   * @return Model hash.
   */
  const uint8_t* (
      *SL_ANeuralNetworksDiagnosticCompilationInfo_getModelArchHash)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * Gets the device IDs as a comma-concatenated string.
   *
   * @param diagnosticCompilationInfo The NNAPI diagnostic compilation info
   * object.
   * @return Device ID.
   */
  const char* (*SL_ANeuralNetworksDiagnosticCompilationInfo_getDeviceIds)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * Gets the error code.
   *
   * @param diagnosticCompilationInfo The NNAPI diagnostic compilation info
   * object.
   * @return Error code.
   */
  int32_t (*SL_ANeuralNetworksDiagnosticCompilationInfo_getErrorCode)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * Gets the type of tensors used for inputs.
   *
   * @param diagnosticCompilationInfo The NNAPI diagnostic compilation info
   * object.
   * @return Input data class.
   */
  ANeuralNetworksDiagnosticDataClass (
      *SL_ANeuralNetworksDiagnosticCompilationInfo_getInputDataClass)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * Gets the type of tensors used for outputs.
   *
   * @param diagnosticCompilationInfo The NNAPI diagnostic compilation info
   * object.
   * @return Output data class.
   */
  ANeuralNetworksDiagnosticDataClass (
      *SL_ANeuralNetworksDiagnosticCompilationInfo_getOutputDataClass)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * Gets how many nanoseconds elapsed when compiling the model.
   *
   * @param diagnosticCompilationInfo The NNAPI diagnostic compilation info
   * object.
   * @return Time to compile the model in nanoseconds. UINT64_MAX indicates that
   * timing information is not available.
   */
  uint64_t (
      *SL_ANeuralNetworksDiagnosticCompilationInfo_getCompilationTimeNanos)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * Is caching enabled?
   *
   * @param diagnosticCompilationInfo The NNAPI diagnostic compilation info
   * object.
   * @return Whether caching is enabled.
   */
  bool (*SL_ANeuralNetworksDiagnosticCompilationInfo_isCachingEnabled)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * Is control flow used?
   *
   * @param diagnosticCompilationInfo The NNAPI diagnostic compilation info
   * object.
   * @return Whether control flow was used.
   */
  bool (*SL_ANeuralNetworksDiagnosticCompilationInfo_isControlFlowUsed)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * Are dynamic tensors used?
   *
   * @param diagnosticCompilationInfo The NNAPI diagnostic compilation info
   * object.
   * @return Whether dynamic tensors were used.
   */
  bool (*SL_ANeuralNetworksDiagnosticCompilationInfo_areDynamicTensorsUsed)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * Gets the ID that identifies a single session of client interacting with
   * NNAPI runtime.
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Session info id.
   */
  int32_t (*SL_ANeuralNetworksDiagnosticExecutionInfo_getSessionId)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Gets NNAPI version.
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return NNAPI version.
   */
  int64_t (*SL_ANeuralNetworksDiagnosticExecutionInfo_getNnApiVersion)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Gets the hash of the model architecture (without weights).
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Model hash.
   */
  const uint8_t* (*SL_ANeuralNetworksDiagnosticExecutionInfo_getModelArchHash)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Gets the device IDs as a comma-concatenated string.
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Device ID.
   */
  const char* (*SL_ANeuralNetworksDiagnosticExecutionInfo_getDeviceIds)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Gets the execution mode.
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Execution mode.
   */
  ANeuralNetworksDiagnosticExecutionMode (
      *SL_ANeuralNetworksDiagnosticExecutionInfo_getExecutionMode)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Gets the input data class.
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Input data class.
   */
  ANeuralNetworksDiagnosticDataClass (
      *SL_ANeuralNetworksDiagnosticExecutionInfo_getInputDataClass)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Gets the output data class.
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Output data class.
   */
  ANeuralNetworksDiagnosticDataClass (
      *SL_ANeuralNetworksDiagnosticExecutionInfo_getOutputDataClass)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Gets the error code.
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Error code.
   */
  uint32_t (*SL_ANeuralNetworksDiagnosticExecutionInfo_getErrorCode)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Gets the time taken to execute from runtime, including runtime/ipc
   * overhead.
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Time taken to execute as measured by the runtime in nanoseconds.
   * UINT64_MAX indicates that timing information is not available.
   */
  uint64_t (
      *SL_ANeuralNetworksDiagnosticExecutionInfo_getRuntimeExecutionTimeNanos)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Gets the time taken to execute in the driver, excluding runtime/ipc
   * overhead.
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Time taken to execute on the driver in nanoseconds. UINT64_MAX
   * indicates that timing information is not available.
   */
  uint64_t (
      *SL_ANeuralNetworksDiagnosticExecutionInfo_getDriverExecutionTimeNanos)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Gets the time taken to execute on the hardware, excluding driver overhead.
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Time taken to execute on the hardware in nanoseconds. UINT64_MAX
   * indicates that timing information is not available.
   */
  uint64_t (
      *SL_ANeuralNetworksDiagnosticExecutionInfo_getHardwareExecutionTimeNanos)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Is caching enabled?
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Whether caching is enabled.
   */
  bool (*SL_ANeuralNetworksDiagnosticExecutionInfo_isCachingEnabled)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Is control flow used?
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Whether control flow was used.
   */
  bool (*SL_ANeuralNetworksDiagnosticExecutionInfo_isControlFlowUsed)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Are dynamic tensors used?
   *
   * @param diagnosticExecutionInfo The NNAPI diagnostic compilation info
   * object.
   * @return Whether dynamic tensors were used.
   */
  bool (*SL_ANeuralNetworksDiagnosticExecutionInfo_areDynamicTensorsUsed)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * Sets the callbacks to be called when compilations or executions finish.
   *
   * Example usage:
   *
   * // Callback to be invoked whenever a compilation has completed.
   * void compilationCallback(void* context,
   * ANeuralNetworksDiagnosticCompilationInfo* info) {
   *     // The context object can be used to store state without the use of a
   * global variable. ExampleLoggerObject* logger =
   * static_cast<ExampleLoggerObject*>(context);
   *
   *     // Calls to getters to get the details...
   *     const int32_t sessionId =
   * ANeuralNetworksDiagnosticCompilationInfo_getSessionId(info);
   *
   *     ...
   *
   *     logger->write(...);
   * }
   *
   * void executionCallback(void* context,
   * ANeuralNetworksDiagnosticExecutionInfo* info) {
   *      ...
   * }
   *
   * ExampleLoggerObject exampleLoggerObject;
   * ANeuralNetworksDiagnostic_registerCallbacks(&compilationCallback,
   * &executionCallback, static_cast<void*>(&exampleLoggerObject));
   *
   * @param compilationCallback The compilation callback to set.
   * @param executionCallback The execution callback to set.
   * @param callbackContext The context to be passed to the callbacks when they
   * are invoked. The context object may be used by multiple threads
   * simulatenously, so it must be thread-safe.
   */
  void (*SL_ANeuralNetworksDiagnostic_registerCallbacks)(
      ANeuralNetworksDiagnosticCompilationFinishedCallback compilationCallback,
      ANeuralNetworksDiagnosticExecutionFinishedCallback executionCallback,
      void* callbackContext);
};

/**
 * Load the NNAPI implementation from the shared libraries.
 * The NnApi structure is filled with all the pointers. If one function doesn't
 * exist, a null pointer is stored.
 */
const NnApi* NnApiImplementation();

// Forward declaration for CreateNnApiFromSupportLibrary below.
struct NnApiSLDriverImplFL5;

/**
 * Allocate a new NnApi structure instance and fill it with function pointers
 * from NnApiSLDriverImplFL5 instance. Functions that are not present in the
 * support library are assigned null pointers.
 *
 * The NN API Support Library Driver must support at least NNAPI Feature Level 5
 * (introduced in SDK level 31), but this might point to a compatible struct
 * that also supports a higher NNAPI Feature Level. These cases can be
 * distinguished by examining the base.implFeatureLevel field, which should be
 * set to the supported feature level (which must be >=
 * ANEURALNETWORKS_FEATURE_LEVEL_5).
 */
std::unique_ptr<const NnApi> CreateNnApiFromSupportLibrary(
    const NnApiSLDriverImplFL5* nnapi_support_library_driver);

#endif  // TENSORFLOW_LITE_NNAPI_NNAPI_IMPLEMENTATION_H_
