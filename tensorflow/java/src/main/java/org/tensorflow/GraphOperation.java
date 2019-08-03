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

/**
 * Implementation for an {@link Operation} added as a node to a {@link Graph}.
 *
 * <p>GraphOperation instances are valid only as long as the {@link Graph} they are a part of is
 * valid. Thus, if {@link Graph#close()} has been invoked, then methods on the GraphOperation
 * instance may fail with an {@code IllegalStateException}.
 *
 * <p>GraphOperation instances are immutable and thread-safe.
 */
public final class GraphOperation extends AbstractOperation {

  // Create an GraphOperation instance referring to an operation in g, with the given handle to the
  // C
  // TF_Operation object.  The handle is valid only as long as g has not been closed, hence it is
  // called unsafeHandle.  Graph.ref() is used to safely use the unsafeHandle.
  GraphOperation(Graph g, long unsafeNativeHandle) {
    this.graph = g;
    this.unsafeNativeHandle = unsafeNativeHandle;
  }

  @Override
  public String name() {
    Graph.Reference r = graph.ref();
    try {
      return name(getUnsafeNativeHandle());
    } finally {
      r.close();
    }
  }

  @Override
  public String type() {
    Graph.Reference r = graph.ref();
    try {
      return type(getUnsafeNativeHandle());
    } finally {
      r.close();
    }
  }

  @Override
  public int numOutputs() {
    Graph.Reference r = graph.ref();
    try {
      return numOutputs(getUnsafeNativeHandle());
    } finally {
      r.close();
    }
  }

  @Override
  public int outputListLength(final String name) {
    Graph.Reference r = graph.ref();
    try {
      return outputListLength(getUnsafeNativeHandle(), name);
    } finally {
      r.close();
    }
  }

  @Override
  public int hashCode() {
    return Long.valueOf(getUnsafeNativeHandle()).hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }
    if (!(o instanceof GraphOperation)) {
      return false;
    }
    GraphOperation that = (GraphOperation) o;
    if (graph != that.graph) {
      return false;
    }

    // The graph object is known to be identical here, so this one
    // reference is sufficient to validate the use of native pointers
    // in both objects.
    Graph.Reference r = graph.ref();
    try {
      return getUnsafeNativeHandle() == that.getUnsafeNativeHandle();
    } finally {
      r.close();
    }
  }

  @Override
  public int inputListLength(final String name) {
    Graph.Reference r = graph.ref();
    try {
      return inputListLength(getUnsafeNativeHandle(), name);
    } finally {
      r.close();
    }
  }

  @Override
  long getUnsafeNativeHandle(int outputIdx) {
    return getUnsafeNativeHandle();
  }

  @Override
  long[] shape(int outputIdx) {
    Graph.Reference r = graph.ref();
    try {
      return shape(r.nativeHandle(), getUnsafeNativeHandle(), outputIdx);
    } finally {
      r.close();
    }
  }

  @Override
  DataType dtype(int outputIdx) {
    Graph.Reference r = graph.ref();
    try {
      return DataType.fromC(dtype(r.nativeHandle(), getUnsafeNativeHandle(), outputIdx));
    } finally {
      r.close();
    }
  }

  @Override
  Tensor<?> tensor(int outputIdx) {
    throw new IllegalStateException("Graph tensors must be fetched by running a session");
  }

  long getUnsafeNativeHandle() {
    return unsafeNativeHandle;
  }

  private final Graph graph;

  private final long unsafeNativeHandle;

  private static native String name(long handle);

  private static native String type(long handle);

  private static native int numOutputs(long handle);

  private static native int outputListLength(long handle, String name);

  private static native int inputListLength(long handle, String name);

  private static native long[] shape(long graphHandle, long opHandle, int output);

  private static native int dtype(long graphHandle, long opHandle, int output);
}
