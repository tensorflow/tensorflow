/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

/**
 * A builder for {@link Operation}s in a {@link Graph}.
 *
 * <p>Instances of an OperationBuilder are <b>not</b> thread-safe.
 *
 * <p>A builder for adding {@link Operation}s to a {@link Graph}. For example, the following uses
 * the builder to create an operation that produces the constant "3" as its output:
 *
 * <pre>{@code
 * // g is a Graph instance.
 * try (Tensor c1 = Tensor.create(3.0f)) {
 *   g.opBuilder("Constant", "MyConst")
 *       .setAttr("dtype", c1.dataType())
 *       .setAttr("value", c1)
 *       .build();
 * }
 * }</pre>
 */
public final class OperationBuilder {

  OperationBuilder(Graph graph, String type, String name) {
    this.graph = graph;
    try (Graph.Reference r = graph.ref()) {
      this.unsafeNativeHandle = allocate(r.nativeHandle(), type, name);
    }
  }

  /**
   * Add the {@link Operation} being built to the {@link Graph}.
   *
   * <p>The OperationBuilder is not usable after build() returns.
   */
  public Operation build() {
    try (Graph.Reference r = graph.ref()) {
      Operation op = new Operation(graph, finish(unsafeNativeHandle));
      unsafeNativeHandle = 0;
      return op;
    }
  }

  public OperationBuilder addInput(Output input) {
    try (Graph.Reference r = graph.ref()) {
      addInput(unsafeNativeHandle, input.op().getUnsafeNativeHandle(), input.index());
    }
    return this;
  }

  public OperationBuilder addInputList(Output[] inputs) {
    try (Graph.Reference r = graph.ref()) {
      long[] opHandles = new long[inputs.length];
      int[] indices = new int[inputs.length];
      for (int i = 0; i < inputs.length; ++i) {
        opHandles[i] = inputs[i].op().getUnsafeNativeHandle();
        indices[i] = inputs[i].index();
      }
      addInputList(unsafeNativeHandle, opHandles, indices);
    }
    return this;
  }

  public OperationBuilder setDevice(String device) {
    try (Graph.Reference r = graph.ref()) {
      setDevice(unsafeNativeHandle, device);
    }
    return this;
  }

  public OperationBuilder setAttr(String name, String value) {
    setAttr(name, value.getBytes(Charset.forName("UTF-8")));
    return this;
  }

  public OperationBuilder setAttr(String name, byte[] value) {
    try (Graph.Reference r = graph.ref()) {
      setAttrString(unsafeNativeHandle, name, value);
    }
    return this;
  }

  public OperationBuilder setAttr(String name, long value) {
    try (Graph.Reference r = graph.ref()) {
      setAttrInt(unsafeNativeHandle, name, value);
    }
    return this;
  }

  public OperationBuilder setAttr(String name, long[] value) {
    try (Graph.Reference r = graph.ref()) {
      setAttrIntList(unsafeNativeHandle, name, value);
    }
    return this;
  }

  public OperationBuilder setAttr(String name, float value) {
    try (Graph.Reference r = graph.ref()) {
      setAttrFloat(unsafeNativeHandle, name, value);
    }
    return this;
  }

  public OperationBuilder setAttr(String name, float[] value) {
    try (Graph.Reference r = graph.ref()) {
      setAttrFloatList(unsafeNativeHandle, name, value);
    }
    return this;
  }

  public OperationBuilder setAttr(String name, boolean value) {
    try (Graph.Reference r = graph.ref()) {
      setAttrBool(unsafeNativeHandle, name, value);
    }
    return this;
  }

  public OperationBuilder setAttr(String name, boolean[] value) {
    try (Graph.Reference r = graph.ref()) {
      setAttrBoolList(unsafeNativeHandle, name, value);
    }
    return this;
  }

  public OperationBuilder setAttr(String name, DataType value) {
    try (Graph.Reference r = graph.ref()) {
      setAttrType(unsafeNativeHandle, name, value.c());
    }
    return this;
  }

  public OperationBuilder setAttr(String name, DataType[] value) {
    int[] ctypes = new int[value.length];
    for (int i = 0; i < value.length; ++i) {
      ctypes[i] = value[i].c();
    }
    try (Graph.Reference r = graph.ref()) {
      setAttrTypeList(unsafeNativeHandle, name, ctypes);
    }
    return this;
  }

  public OperationBuilder setAttr(String name, Tensor value) {
    try (Graph.Reference r = graph.ref()) {
      setAttrTensor(unsafeNativeHandle, name, value.getNativeHandle());
    }
    return this;
  }

  public OperationBuilder setAttr(String name, Tensor[] value) {
    long[] handles = new long[value.length];
    int idx = 0;
    for (Tensor t : value) {
      handles[idx++] = t.getNativeHandle();
    }
    try (Graph.Reference r = graph.ref()) {
      setAttrTensorList(unsafeNativeHandle, name, handles);
    }
    return this;
  }

  private long unsafeNativeHandle;
  private Graph graph;

  private static native long allocate(long graphHandle, String type, String name);

  private static native long finish(long handle);

  private static native void addInput(long handle, long opHandle, int index);

  private static native void addInputList(long handle, long[] opHandles, int[] indices);

  private static native void setDevice(long handle, String device);

  // The names of all the setAttr* family functions below correspond to the C library types, not the
  // Java library types. Roughly, setAttrFoo calls the TensorFlow C library function: TF_SetAttrFoo.
  //
  // TODO(ashankar):
  // - setAttrStringList: Which would take in an array of byte[] (java Strings will need to be UTF-8
  //   encoded?)
  // - setAttrShape and setAttrShapeList: Which would take in a long[] or long[][]?

  private static native void setAttrString(long handle, String name, byte[] value);

  private static native void setAttrInt(long handle, String name, long value);

  private static native void setAttrIntList(long handle, String name, long[] value);

  private static native void setAttrFloat(long handle, String name, float value);

  private static native void setAttrFloatList(long handle, String name, float[] value);

  private static native void setAttrBool(long handle, String name, boolean value);

  private static native void setAttrBoolList(long handle, String name, boolean[] value);

  private static native void setAttrType(long handle, String name, int type);

  private static native void setAttrTypeList(long handle, String name, int[] type);

  private static native void setAttrTensor(long handle, String name, long tensorHandle);

  private static native void setAttrTensorList(long handle, String name, long[] tensorHandle);
}
