/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow;

import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * Implementation of an {@link Operation} executed eagerly.
 *
 * <p>EagerOperation instances are valid only as long as the {@link EagerSession} they are a part of
 * is valid. Thus, if {@link EagerSession#close()} has been invoked, then methods on the
 * EagerOperation instance may fail with an {@code IllegalStateException}.
 *
 * <p>EagerOperation instances are thread-safe.
 */
class EagerOperation extends AbstractOperation {

  EagerOperation(
      EagerSession session,
      long opNativeHandle,
      long[] outputNativeHandles,
      String type,
      String name) {
    this.session = session;
    this.type = type;
    this.name = name;
    this.nativeRef = new NativeReference(session, this, opNativeHandle, outputNativeHandles);
    this.outputTensors = new AtomicReferenceArray<Tensor<?>>(outputNativeHandles.length);
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public String type() {
    return type;
  }

  @Override
  public int numOutputs() {
    return nativeRef.outputHandles.length;
  }

  @Override
  public int outputListLength(final String name) {
    return outputListLength(nativeRef.opHandle, name);
  }

  @Override
  public int inputListLength(final String name) {
    return inputListLength(nativeRef.opHandle, name);
  }

  @Override
  public long getUnsafeNativeHandle(int outputIndex) {
    return nativeRef.outputHandles[outputIndex];
  }

  @Override
  public long[] shape(int outputIndex) {
    // If the tensor of this output has already been resolved, return its shape.
    // Otherwise, retrieve the tensor shape from the native library.
    Tensor<?> tensor = outputTensors.get(outputIndex);
    if (tensor != null) {
      return tensor.shape();
    }
    long outputNativeHandle = getUnsafeNativeHandle(outputIndex);
    long[] shape = new long[numDims(outputNativeHandle)];
    for (int i = 0; i < shape.length; ++i) {
      shape[i] = dim(outputNativeHandle, i);
    }
    return shape;
  }

  @Override
  public DataType dtype(int outputIndex) {
    // If the tensor of this output has already been resolved, return its datatype.
    // Otherwise, retrieve the tensor datatype from the native library.
    Tensor<?> tensor = outputTensors.get(outputIndex);
    if (tensor != null) {
      return tensor.dataType();
    }
    long outputNativeHandle = getUnsafeNativeHandle(outputIndex);
    return DataType.fromC(dataType(outputNativeHandle));
  }

  @Override
  public Tensor<?> tensor(int outputIndex) {
    Tensor<?> tensor = outputTensors.get(outputIndex);
    if (tensor == null) {
      tensor = resolveTensor(outputIndex);
    }
    return tensor;
  }

  private final EagerSession session;
  private final NativeReference nativeRef;
  private final String type;
  private final String name;
  private final AtomicReferenceArray<Tensor<?>> outputTensors;

  private Tensor<?> resolveTensor(int outputIndex) {
    // Take an optimistic approach, where we attempt to resolve the output tensor without locking.
    // If another thread has resolved it meanwhile, release our copy and reuse the existing one
    // instead.
    long tensorNativeHandle = resolveTensorHandle(getUnsafeNativeHandle(outputIndex));
    Tensor<?> tensor = Tensor.fromHandle(tensorNativeHandle, session);
    if (!outputTensors.compareAndSet(outputIndex, null, tensor)) {
      tensor.close();
      tensor = outputTensors.get(outputIndex);
    }
    return tensor;
  }

  private static class NativeReference extends EagerSession.NativeReference {

    NativeReference(
        EagerSession session, EagerOperation operation, long opHandle, long[] outputHandles) {
      super(session, operation);
      this.opHandle = opHandle;
      this.outputHandles = outputHandles;
    }

    @Override
    void delete() {
      if (opHandle != 0L) {
        for (int i = 0; i < outputHandles.length; ++i) {
          if (outputHandles[i] != 0L) {
            EagerOperation.deleteTensorHandle(outputHandles[i]);
            outputHandles[i] = 0L;
          }
        }
        EagerOperation.delete(opHandle);
        opHandle = 0L;
      }
    }

    private long opHandle;
    private final long[] outputHandles;
  }
  
  private static native void delete(long handle);

  private static native void deleteTensorHandle(long handle);

  private static native long resolveTensorHandle(long handle);

  private static native int outputListLength(long handle, String name);

  private static native int inputListLength(long handle, String name);

  private static native int dataType(long handle);

  private static native int numDims(long handle);

  private static native long dim(long handle, int index);
}
