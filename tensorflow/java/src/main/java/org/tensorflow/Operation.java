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

/**
 * A Graph node that performs computation on Tensors.
 *
 * <p>An Operation is a node in a {@link Graph} that takes zero or more {@link Tensor}s (produced by
 * other Operations in the Graph) as input, and produces zero or more {@link Tensor}s as output.
 *
 * <p>Operation instances are valid only as long as the Graph they are a part of is valid. Thus, if
 * {@link Graph#close()} has been invoked, then methods on the Operation instance may fail with an
 * {@code IllegalStateException}.
 *
 * <p>Operation instances are immutable and thread-safe.
 */
public final class Operation {

  // Create an Operation instance referring to an operation in g, with the given handle to the C
  // TF_Operation object.  The handle is valid only as long as g has not been closed, hence it is
  // called unsafeHandle.  Graph.ref() is used to safely use the unsafeHandle.
  Operation(Graph g, long unsafeNativeHandle) {
    this.graph = g;
    this.unsafeNativeHandle = unsafeNativeHandle;
  }

  /** Returns the full name of the Operation. */
  public String name() {
    Graph.Reference r = graph.ref();
    try {
      return name(unsafeNativeHandle);
    } finally {
      r.close();
    }
  }

  /**
   * Returns the type of the operation, i.e., the name of the computation performed by the
   * operation.
   */
  public String type() {
    Graph.Reference r = graph.ref();
    try {
      return type(unsafeNativeHandle);
    } finally {
      r.close();
    }
  }

  /** Returns the number of tensors produced by this operation. */
  public int numOutputs() {
    Graph.Reference r = graph.ref();
    try {
      return numOutputs(unsafeNativeHandle);
    } finally {
      r.close();
    }
  }

  /** Returns a symbolic handle to one of the tensors produced by this operation. */
  public Output output(int idx) {
    return new Output(this, idx);
  }

  long getUnsafeNativeHandle() {
    return unsafeNativeHandle;
  }

  // Package private, meant primarily for the public Output.shape() method.
  long[] shape(int output) {
    Graph.Reference r = graph.ref();
    try {
      return shape(r.nativeHandle(), unsafeNativeHandle, output);
    } finally {
      r.close();
    }
  }

  private final long unsafeNativeHandle;
  private final Graph graph;

  private static native String name(long handle);

  private static native String type(long handle);

  private static native int numOutputs(long handle);

  private static native long[] shape(long graphHandle, long opHandle, int output);
}
