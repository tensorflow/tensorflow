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

package org.tensorflow.lite;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * A wrapper wraps native interpreter and controls model execution.
 *
 * <p><b>WARNING:</b> Resources consumed by the {@code NativeInterpreterWrapper} object must be
 * explicitly freed by invoking the {@link #close()} method when the {@code
 * NativeInterpreterWrapper} object is no longer needed.
 */
final class NativeInterpreterWrapper implements AutoCloseable {

  NativeInterpreterWrapper(String modelPath) {
    this(modelPath, /* numThreads= */ -1);
  }

  NativeInterpreterWrapper(String modelPath, int numThreads) {
    errorHandle = createErrorReporter(ERROR_BUFFER_SIZE);
    modelHandle = createModel(modelPath, errorHandle);
    interpreterHandle = createInterpreter(modelHandle, errorHandle, numThreads);
    isMemoryAllocated = true;
  }

  /**
   * Initializes a {@code NativeInterpreterWrapper} with a {@code MappedByteBuffer}. The
   * MappedByteBuffer should not be modified after the construction of a {@code
   * NativeInterpreterWrapper}.
   */
  NativeInterpreterWrapper(MappedByteBuffer mappedByteBuffer) {
    this(mappedByteBuffer, /* numThreads= */ -1);
  }

  /**
   * Initializes a {@code NativeInterpreterWrapper} with a {@code MappedByteBuffer} and specifies
   * the number of inference threads. The MappedByteBuffer should not be modified after the
   * construction of a {@code NativeInterpreterWrapper}.
   */
  NativeInterpreterWrapper(MappedByteBuffer mappedByteBuffer, int numThreads) {
    modelByteBuffer = mappedByteBuffer;
    errorHandle = createErrorReporter(ERROR_BUFFER_SIZE);
    modelHandle = createModelWithBuffer(modelByteBuffer, errorHandle);
    interpreterHandle = createInterpreter(modelHandle, errorHandle, numThreads);
    isMemoryAllocated = true;
  }

  /** Releases resources associated with this {@code NativeInterpreterWrapper}. */
  @Override
  public void close() {
    delete(errorHandle, modelHandle, interpreterHandle);
    errorHandle = 0;
    modelHandle = 0;
    interpreterHandle = 0;
    modelByteBuffer = null;
    inputsIndexes = null;
    outputsIndexes = null;
    isMemoryAllocated = false;
  }

  /** Sets inputs, runs model inference and returns outputs. */
  Tensor[] run(Object[] inputs) {
    if (inputs == null || inputs.length == 0) {
      throw new IllegalArgumentException("Invalid inputs. Inputs should not be null or empty.");
    }
    int[] dataTypes = new int[inputs.length];
    Object[] sizes = new Object[inputs.length];
    int[] numsOfBytes = new int[inputs.length];
    for (int i = 0; i < inputs.length; ++i) {
      DataType dataType = dataTypeOf(inputs[i]);
      dataTypes[i] = dataType.getNumber();
      if (dataType == DataType.BYTEBUFFER) {
        ByteBuffer buffer = (ByteBuffer) inputs[i];
        if (buffer.order() != ByteOrder.nativeOrder()) {
          throw new IllegalArgumentException(
              "Invalid ByteBuffer. It shoud use ByteOrder.nativeOrder().");
        }
        numsOfBytes[i] = buffer.limit();
        sizes[i] = getInputDims(interpreterHandle, i, numsOfBytes[i]);
      } else if (isNonEmptyArray(inputs[i])) {
        int[] dims = shapeOf(inputs[i]);
        sizes[i] = dims;
        numsOfBytes[i] = dataType.elemByteSize() * numElements(dims);
      } else {
        throw new IllegalArgumentException(
            String.format(
                "%d-th element of the %d inputs is not an array or a ByteBuffer.",
                i, inputs.length));
      }
    }
    inferenceDurationNanoseconds = -1;
    long[] outputsHandles =
        run(
            interpreterHandle,
            errorHandle,
            sizes,
            dataTypes,
            numsOfBytes,
            inputs,
            this,
            isMemoryAllocated);
    if (outputsHandles == null || outputsHandles.length == 0) {
      throw new IllegalStateException("Interpreter has no outputs.");
    }
    isMemoryAllocated = true;
    Tensor[] outputs = new Tensor[outputsHandles.length];
    for (int i = 0; i < outputsHandles.length; ++i) {
      outputs[i] = Tensor.fromHandle(outputsHandles[i]);
    }
    return outputs;
  }

  private static native long[] run(
      long interpreterHandle,
      long errorHandle,
      Object[] sizes,
      int[] dtypes,
      int[] numsOfBytes,
      Object[] values,
      NativeInterpreterWrapper wrapper,
      boolean memoryAllocated);

  /** Resizes dimensions of a specific input. */
  void resizeInput(int idx, int[] dims) {
    if (resizeInput(interpreterHandle, errorHandle, idx, dims)) {
      isMemoryAllocated = false;
    }
  }

  private static native boolean resizeInput(
      long interpreterHandle, long errorHandle, int inputIdx, int[] dims);

  void setUseNNAPI(boolean useNNAPI) {
    useNNAPI(interpreterHandle, useNNAPI);
  }

  /** Gets index of an input given its name. */
  int getInputIndex(String name) {
    if (inputsIndexes == null) {
      String[] names = getInputNames(interpreterHandle);
      inputsIndexes = new HashMap<>();
      if (names != null) {
        for (int i = 0; i < names.length; ++i) {
          inputsIndexes.put(names[i], i);
        }
      }
    }
    if (inputsIndexes.containsKey(name)) {
      return inputsIndexes.get(name);
    } else {
      throw new IllegalArgumentException(
          String.format(
              "%s is not a valid name for any input. The indexes of the inputs are %s",
              name, inputsIndexes.toString()));
    }
  }

  /** Gets index of an output given its name. */
  int getOutputIndex(String name) {
    if (outputsIndexes == null) {
      String[] names = getOutputNames(interpreterHandle);
      outputsIndexes = new HashMap<>();
      if (names != null) {
        for (int i = 0; i < names.length; ++i) {
          outputsIndexes.put(names[i], i);
        }
      }
    }
    if (outputsIndexes.containsKey(name)) {
      return outputsIndexes.get(name);
    } else {
      throw new IllegalArgumentException(
          String.format(
              "%s is not a valid name for any output. The indexes of the outputs are %s",
              name, outputsIndexes.toString()));
    }
  }

  static int numElements(int[] shape) {
    if (shape == null) {
      return 0;
    }
    int n = 1;
    for (int i = 0; i < shape.length; i++) {
      n *= shape[i];
    }
    return n;
  }

  static boolean isNonEmptyArray(Object o) {
    return (o != null && o.getClass().isArray() && Array.getLength(o) != 0);
  }

  /** Returns the type of the data. */
  static DataType dataTypeOf(Object o) {
    if (o != null) {
      Class<?> c = o.getClass();
      while (c.isArray()) {
        c = c.getComponentType();
      }
      if (float.class.equals(c)) {
        return DataType.FLOAT32;
      } else if (int.class.equals(c)) {
        return DataType.INT32;
      } else if (byte.class.equals(c)) {
        return DataType.UINT8;
      } else if (long.class.equals(c)) {
        return DataType.INT64;
      } else if (ByteBuffer.class.isInstance(o)) {
        return DataType.BYTEBUFFER;
      }
    }
    throw new IllegalArgumentException("cannot resolve DataType of " + o.getClass().getName());
  }

  /** Returns the shape of an object as an int array. */
  static int[] shapeOf(Object o) {
    int size = numDimensions(o);
    int[] dimensions = new int[size];
    fillShape(o, 0, dimensions);
    return dimensions;
  }

  static int numDimensions(Object o) {
    if (o == null || !o.getClass().isArray()) {
      return 0;
    }
    if (Array.getLength(o) == 0) {
      throw new IllegalArgumentException("array lengths cannot be 0.");
    }
    return 1 + numDimensions(Array.get(o, 0));
  }

  static void fillShape(Object o, int dim, int[] shape) {
    if (shape == null || dim == shape.length) {
      return;
    }
    final int len = Array.getLength(o);
    if (shape[dim] == 0) {
      shape[dim] = len;
    } else if (shape[dim] != len) {
      throw new IllegalArgumentException(
          String.format("mismatched lengths (%d and %d) in dimension %d", shape[dim], len, dim));
    }
    for (int i = 0; i < len; ++i) {
      fillShape(Array.get(o, i), dim + 1, shape);
    }
  }

  /**
   * Gets the last inference duration in nanoseconds. It returns null if there is no previous
   * inference run or the last inference run failed.
   */
  Long getLastNativeInferenceDurationNanoseconds() {
    return (inferenceDurationNanoseconds < 0) ? null : inferenceDurationNanoseconds;
  }

  /**
   * Gets the dimensions of an input. It throws IllegalArgumentException if input index is invalid.
   */
  int[] getInputDims(int index) {
    return getInputDims(interpreterHandle, index, -1);
  }

  /**
   * Gets the dimensions of an input. If numBytes >= 0, it will check whether num of bytes match the
   * input.
   */
  private static native int[] getInputDims(long interpreterHandle, int inputIdx, int numBytes);

  /** Gets the type of an output. It throws IllegalArgumentException if output index is invalid. */
  String getOutputDataType(int index) {
    int type = getOutputDataType(interpreterHandle, index);
    return DataType.fromNumber(type).toStringName();
  }

  private static native int getOutputDataType(long interpreterHandle, int outputIdx);

  private static final int ERROR_BUFFER_SIZE = 512;

  private long errorHandle;

  private long interpreterHandle;

  private long modelHandle;

  private int inputSize;

  private long inferenceDurationNanoseconds = -1;

  private MappedByteBuffer modelByteBuffer;

  private Map<String, Integer> inputsIndexes;

  private Map<String, Integer> outputsIndexes;

  private boolean isMemoryAllocated = false;

  private static native String[] getInputNames(long interpreterHandle);

  private static native String[] getOutputNames(long interpreterHandle);

  private static native void useNNAPI(long interpreterHandle, boolean state);

  private static native long createErrorReporter(int size);

  private static native long createModel(String modelPathOrBuffer, long errorHandle);

  private static native long createModelWithBuffer(MappedByteBuffer modelBuffer, long errorHandle);

  private static native long createInterpreter(long modelHandle, long errorHandle, int numThreads);

  private static native void delete(long errorHandle, long modelHandle, long interpreterHandle);

  static {
    TensorFlowLite.init();
  }
}
