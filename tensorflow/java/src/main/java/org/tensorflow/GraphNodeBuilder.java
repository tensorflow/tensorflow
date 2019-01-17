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

/**
 * An {@link OperationBuilder} for adding {@link GraphNode}s to a {@link Graph}.
 */
public final class GraphNodeBuilder implements OperationBuilder {

  GraphNodeBuilder(Graph graph, String type, String name) {
    this.graph = graph;
    Graph.Reference r = graph.ref();
    try {
      this.unsafeNativeHandle = allocate(r.nativeHandle(), type, name);
    } finally {
      r.close();
    }
  }

  /**
   * Add the {@link GraphNode} being built to the {@link Graph}.
   *
   * <p>The OperationBuilder is not usable after build() returns.
   */
  @Override
  public GraphNode build() {
    Graph.Reference r = graph.ref();
    try {
      GraphNode op = new GraphNode(graph, finish(unsafeNativeHandle));
      unsafeNativeHandle = 0;
      return op;
    } finally {
      r.close();
    }
  }

  @Override
  public GraphNodeBuilder addControlInput(Operation control) {
    if (!(control instanceof GraphNode)) {
      throw new IllegalArgumentException("Only GraphNode operations can be used as control input");
    }
    Graph.Reference r = graph.ref();
    try {
      addControlInput(unsafeNativeHandle, ((GraphNode)control).getUnsafeNativeHandle());
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder addInput(Output<?> input) {
    Graph.Reference r = graph.ref();
    try {
      addInput(unsafeNativeHandle, input.getUnsafeNativeHandle(), input.index());
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder addInputList(Output<?>[] inputs) {
    Graph.Reference r = graph.ref();
    try {
      long[] opHandles = new long[inputs.length];
      int[] indices = new int[inputs.length];
      for (int i = 0; i < inputs.length; ++i) {
        opHandles[i] = inputs[i].getUnsafeNativeHandle();
        indices[i] = inputs[i].index();
      }
      addInputList(unsafeNativeHandle, opHandles, indices);
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setDevice(String device) {
    Graph.Reference r = graph.ref();
    try {
      setDevice(unsafeNativeHandle, device);
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, String value) {
    setAttr(name, value.getBytes(Charset.forName("UTF-8")));
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, byte[] value) {
    Graph.Reference r = graph.ref();
    try {
      setAttrString(unsafeNativeHandle, name, value);
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, long value) {
    Graph.Reference r = graph.ref();
    try {
      setAttrInt(unsafeNativeHandle, name, value);
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, long[] value) {
    Graph.Reference r = graph.ref();
    try {
      setAttrIntList(unsafeNativeHandle, name, value);
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, float value) {
    Graph.Reference r = graph.ref();
    try {
      setAttrFloat(unsafeNativeHandle, name, value);
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, float[] value) {
    Graph.Reference r = graph.ref();
    try {
      setAttrFloatList(unsafeNativeHandle, name, value);
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, boolean value) {
    Graph.Reference r = graph.ref();
    try {
      setAttrBool(unsafeNativeHandle, name, value);
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, boolean[] value) {
    Graph.Reference r = graph.ref();
    try {
      setAttrBoolList(unsafeNativeHandle, name, value);
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, DataType value) {
    Graph.Reference r = graph.ref();
    try {
      setAttrType(unsafeNativeHandle, name, value.c());
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, DataType[] value) {
    int[] ctypes = new int[value.length];
    for (int i = 0; i < value.length; ++i) {
      ctypes[i] = value[i].c();
    }
    Graph.Reference r = graph.ref();
    try {
      setAttrTypeList(unsafeNativeHandle, name, ctypes);
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, Tensor<?> value) {
    Graph.Reference r = graph.ref();
    try {
      setAttrTensor(unsafeNativeHandle, name, value.getNativeHandle());
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, Tensor<?>[] value) {
    long[] handles = new long[value.length];
    int idx = 0;
    for (Tensor<?> t : value) {
      handles[idx++] = t.getNativeHandle();
    }
    Graph.Reference r = graph.ref();
    try {
      setAttrTensorList(unsafeNativeHandle, name, handles);
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, Shape value) {
    Graph.Reference r = graph.ref();
    try {
      setAttrShape(unsafeNativeHandle, name, value.asArray(), value.numDimensions());
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, Shape[] value) {
    int[] numDimensions = new int[value.length];
    int totalNumDimensions = 0;
    for (int idx = 0; idx < value.length; ++idx) {
      int n = value[idx].numDimensions();
      numDimensions[idx] = n;
      if (n > 0) {
        totalNumDimensions += n;
      }
    }
    // Flatten the shapes into a single array to avoid too much overhead in the
    // native part
    long[] shapes = new long[totalNumDimensions];
    int shapeIdx = 0;
    for (Shape shape : value) {
      if (shape.numDimensions() > 0) {
        for (long dim : shape.asArray()) {
          shapes[shapeIdx++] = dim;
        }
      }
    }
    Graph.Reference r = graph.ref();
    try {
      setAttrShapeList(unsafeNativeHandle, name, shapes, numDimensions);
    } finally {
      r.close();
    }
    return this;
  }

  @Override
  public GraphNodeBuilder setAttr(String name, String[] value) {
    Charset utf8 = Charset.forName("UTF-8");
    Object[] objects = new Object[value.length];
    for (int i = 0; i < value.length; ++i) {
      objects[i] = value[i].getBytes(utf8);
    }
    Graph.Reference r = graph.ref();
    try {
      setAttrStringList(unsafeNativeHandle, name, objects);
    } finally {
      r.close();
    }
    return this;
  }

  private long unsafeNativeHandle;
  private Graph graph;

  private static native long allocate(long graphHandle, String type, String name);

  private static native long finish(long handle);

  private static native void addInput(long handle, long opHandle, int index);

  private static native void addInputList(long handle, long[] opHandles, int[] indices);

  private static native void addControlInput(long handle, long opHandle);

  private static native void setDevice(long handle, String device);

  // The names of all the setAttr* family functions below correspond to the C library types, not the
  // Java library types. Roughly, setAttrFoo calls the TensorFlow C library function: TF_SetAttrFoo.

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

  private static native void setAttrShape(long handle, String name, long[] shape, int numDims);

  private static native void setAttrShapeList(
      long handle, String name, long[] shapes, int[] numDims);

  private static native void setAttrStringList(long handle, String name, Object[] value);
}
