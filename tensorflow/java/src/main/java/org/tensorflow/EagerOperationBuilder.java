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

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

/**
 * An {@link OperationBuilder} for building {@link Operation Operations} that are executed eagerly.
 */
final class EagerOperationBuilder implements OperationBuilder {

  EagerOperationBuilder(EagerSession session, String type, String name) {
    this.session = session;
    this.type = type;
    this.name = name;
    this.nativeRef = new NativeReference(session, this, allocate(session.nativeHandle(), type));
  }

  @Override
  public EagerOperation build() {
    long[] tensorHandles = execute(nativeRef.opHandle);
    EagerOperation operation =
        new EagerOperation(session, nativeRef.opHandle, tensorHandles, type, name);
    // Release our reference to the native op handle now that we transferred its
    // ownership to the EagerOperation
    nativeRef.clear();
    return operation;
  }

  @Override
  public EagerOperationBuilder addInput(Output<?> input) {
    addInput(nativeRef.opHandle, input.getUnsafeNativeHandle());
    return this;
  }

  @Override
  public EagerOperationBuilder addInputList(Output<?>[] inputs) {
    long[] inputHandles = new long[inputs.length];
    for (int i = 0; i < inputs.length; ++i) {
      inputHandles[i] = inputs[i].getUnsafeNativeHandle();
    }
    addInputList(nativeRef.opHandle, inputHandles);
    return this;
  }

  @Override
  public OperationBuilder addControlInput(Operation control) {
    throw new UnsupportedOperationException(
        "Control inputs are not supported in an eager execution environment");
  }

  @Override
  public EagerOperationBuilder setDevice(String device) {
    setDevice(nativeRef.opHandle, device);
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, String value) {
    return setAttr(name, value.getBytes(StandardCharsets.UTF_8));
  }

  @Override
  public EagerOperationBuilder setAttr(String name, String[] values) {
    Charset utf8 = StandardCharsets.UTF_8;
    Object[] objects = new Object[values.length];
    for (int i = 0; i < values.length; ++i) {
      objects[i] = values[i].getBytes(utf8);
    }
    setAttrStringList(nativeRef.opHandle, name, values);
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, byte[] values) {
    setAttrString(nativeRef.opHandle, name, values);
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, long value) {
    setAttrInt(nativeRef.opHandle, name, value);
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, long[] values) {
    setAttrIntList(nativeRef.opHandle, name, values);
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, float value) {
    setAttrFloat(nativeRef.opHandle, name, value);
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, float[] values) {
    setAttrFloatList(nativeRef.opHandle, name, values);
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, boolean value) {
    setAttrBool(nativeRef.opHandle, name, value);
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, boolean[] values) {
    setAttrBoolList(nativeRef.opHandle, name, values);
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, DataType value) {
    setAttrType(nativeRef.opHandle, name, value.c());
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, DataType[] values) {
    int[] c = new int[values.length];
    for (int i = 0; i < values.length; ++i) {
      c[i] = values[i].c();
    }
    setAttrTypeList(nativeRef.opHandle, name, c);
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, Tensor<?> value) {
    setAttrTensor(nativeRef.opHandle, name, value.getNativeHandle());
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, Tensor<?>[] values) {
    // TODO (karllessard) could be supported by adding this attribute type in the eager C API
    throw new UnsupportedOperationException(
        "Tensor list attributes are not supported in eager mode");
  }

  @Override
  public EagerOperationBuilder setAttr(String name, Shape value) {
    setAttrShape(nativeRef.opHandle, name, value.asArray(), value.numDimensions());
    return this;
  }

  @Override
  public EagerOperationBuilder setAttr(String name, Shape[] values) {
    int[] numDimensions = new int[values.length];
    int totalNumDimensions = 0;
    for (int idx = 0; idx < values.length; ++idx) {
      int n = values[idx].numDimensions();
      numDimensions[idx] = n;
      if (n > 0) {
        totalNumDimensions += n;
      }
    }
    // Flatten the shapes into a single array to avoid too much overhead in the
    // native part
    long[] shapes = new long[totalNumDimensions];
    int shapeIdx = 0;
    for (Shape shape : values) {
      if (shape.numDimensions() > 0) {
        for (long dim : shape.asArray()) {
          shapes[shapeIdx++] = dim;
        }
      }
    }
    setAttrShapeList(nativeRef.opHandle, name, shapes, numDimensions);
    return this;
  }

  private static class NativeReference extends EagerSession.NativeReference {

    NativeReference(EagerSession session, EagerOperationBuilder operation, long opHandle) {
      super(session, operation);
      this.opHandle = opHandle;
    }

    @Override
    public void clear() {
      super.clear();
      opHandle = 0L;
    }

    @Override
    synchronized void delete() {
      if (opHandle != 0L) {
        EagerOperationBuilder.delete(opHandle);
        opHandle = 0L;
      }
    }

    private long opHandle;
  }

  private final EagerSession session;
  private final String type;
  private final String name;
  private final NativeReference nativeRef;

  private static native long allocate(long ctxHandle, String type);

  private static native void delete(long opHandle);

  private static native long[] execute(long opHandle);

  private static native void addInput(long opHandle, long tensorHandle);

  private static native void addInputList(long opHandle, long[] tensorHandles);

  private static native void setDevice(long opHandle, String device);

  private static native void setAttrString(long opHandle, String name, byte[] value);

  private static native void setAttrStringList(long opHandle, String name, Object[] value);

  private static native void setAttrInt(long opHandle, String name, long value);

  private static native void setAttrIntList(long opHandle, String name, long[] values);

  private static native void setAttrFloat(long opHandle, String name, float value);

  private static native void setAttrFloatList(long opHandle, String name, float[] values);

  private static native void setAttrBool(long opHandle, String name, boolean value);

  private static native void setAttrBoolList(long opHandle, String name, boolean[] values);

  private static native void setAttrType(long opHandle, String name, int type);

  private static native void setAttrTypeList(long opHandle, String name, int[] types);

  private static native void setAttrTensor(long opHandle, String name, long tensorHandle);

  private static native void setAttrShape(long opHandle, String name, long[] shape, int numDims);

  private static native void setAttrShapeList(
      long opHandle, String name, long[] shapes, int[] numDims);
}
