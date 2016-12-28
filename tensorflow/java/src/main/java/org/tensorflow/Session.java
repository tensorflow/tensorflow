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

import java.util.ArrayList;
import java.util.List;

/**
 * Driver for {@link Graph} execution.
 *
 * <p>A {@code Session} instance encapsulates the environment in which {@link Operation}s in a
 * {@link Graph} are executed to compute {@link Tensor}s. For example:
 *
 * <pre>{@code
 * // Let's say graph is an instance of the Graph class
 * // for the computation y = 3 * x
 *
 * try (Session s = new Session(graph)) {
 *   try (Tensor x = Tensor.create(2.0f);
 *       Tensor y = s.runner().feed("x", x).fetch("y").run().get(0)) {
 *       System.out.println(y.floatValue());  // Will print 6.0f
 *   }
 *   try (Tensor x = Tensor.create(1.1f);
 *       Tensor y = s.runner().feed("x", x).fetch("y").run().get(0)) {
 *       System.out.println(y.floatValue());  // Will print 3.3f
 *   }
 * }
 * }</pre>
 *
 * <p><b>WARNING:</b>A {@code Session} ownes resources that <b>must</b> be explicitly freed by
 * invoking {@link #close()}.
 *
 * <p>Instances of a Session are thread-safe.
 */
public final class Session implements AutoCloseable {

  /** Construct a new session with the associated {@link Graph}. */
  public Session(Graph g) {
    graph = g;
    try (Graph.Reference r = g.ref()) {
      nativeHandle = allocate(r.nativeHandle());
      graphRef = g.ref();
    }
  }

  /**
   * Release resources associated with the Session.
   *
   * <p>Blocks until there are no active executions ({@link Session.Runner#run()} calls). A Session
   * is not usable after close returns.
   */
  @Override
  public void close() {
    graphRef.close();
    synchronized (nativeHandleLock) {
      if (nativeHandle == 0) {
        return;
      }
      while (numActiveRuns > 0) {
        try {
          nativeHandleLock.wait();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          // Possible leak of the Session and Graph in this case?
          return;
        }
      }
      delete(nativeHandle);
      nativeHandle = 0;
    }
  }

  /**
   * Run {@link Operation}s and evaluate {@link Tensor}s.
   *
   * <p>A Runner runs the necessary graph fragments to execute every {@link Operation} required to
   * evaluate the {@link Tensor}s to fetch. The {@link #feed(String,int,Tensor)} call allows callers
   * to override the value of {@link Tensor}s in the graph by substituing the provided {@link
   * Tensor}s for the outputs of the operations provided to {@link #feed(String,int,Tensor)}.
   */
  public final class Runner {
    /**
     * Avoid evaluating {@code operation} and substitute {@code t} for the value it produces.
     *
     * <p>This method is a shorthand for {@code feed(operation, 0, t)}.
     */
    public Runner feed(String operation, Tensor t) {
      return feed(operation, 0, t);
    }

    /**
     * Avoid evaluating the {@code index}-th output of {@code operation} by substituting {@code t}
     * for the value it produces.
     *
     * <p>Operations in a {@link Graph} can have multiple outputs, {@code index} identifies which
     * one {@code t} is being provided for.
     */
    public Runner feed(String operation, int index, Tensor t) {
      Operation op = operationByName(operation);
      if (op != null) {
        inputs.add(op.output(index));
        inputTensors.add(t);
      }
      return this;
    }

    /**
     * Make {@link #run()} return the output of {@code operation}.
     *
     * <p>This method is a shorthand for {@code fetch(operation, 0)}
     */
    public Runner fetch(String operation) {
      return fetch(operation, 0);
    }

    /**
     * Make {@link #run()} return the {@code index}-th output of {@code operation}.
     *
     * <p>Operations in a {@link Graph} can have multiple outputs, {@code index} identifies which
     * one to return.
     */
    public Runner fetch(String operation, int index) {
      Operation op = operationByName(operation);
      if (op != null) {
        outputs.add(op.output(index));
      }
      return this;
    }

    /**
     * Make {@link #run()} execute {@code operation}, but not return the evaluated {@link Tensor}.
     */
    public Runner addTarget(String operation) {
      Operation op = operationByName(operation);
      if (op != null) {
        targets.add(op);
      }
      return this;
    }

    /**
     * Execute the graph fragments necessary to compute all requested fetches.
     *
     * <p><b>WARNING:</b> The caller assumes ownership of all returned {@link Tensor}s, i.e., the
     * caller must call {@link Tensor#close()} on all elements of the returned list to free up
     * resources.
     *
     * <p>TODO(ashankar): Reconsider the return type here. Two things in particular: (a) Make it
     * easier for the caller to cleanup (perhaps returning something like AutoCloseableList in
     * SessionTest.java), and (b) Evaluate whether the return value should be a list, or maybe a
     * {@code Map<Output, Tensor>}?
     */
    public List<Tensor> run() {
      long[] inputTensorHandles = new long[inputTensors.size()];
      long[] inputOpHandles = new long[inputs.size()];
      int[] inputOpIndices = new int[inputs.size()];
      long[] outputOpHandles = new long[outputs.size()];
      int[] outputOpIndices = new int[outputs.size()];
      long[] targetOpHandles = new long[targets.size()];
      long[] outputTensorHandles = new long[outputs.size()];

      // It's okay to use Operation.getUnsafeNativeHandle() here since the safety depends on the
      // validity of the Graph and graphRef ensures that.
      int idx = 0;
      for (Tensor t : inputTensors) {
        inputTensorHandles[idx++] = t.getNativeHandle();
      }
      idx = 0;
      for (Output o : inputs) {
        inputOpHandles[idx] = o.op().getUnsafeNativeHandle();
        inputOpIndices[idx] = o.index();
        idx++;
      }
      idx = 0;
      for (Output o : outputs) {
        outputOpHandles[idx] = o.op().getUnsafeNativeHandle();
        outputOpIndices[idx] = o.index();
      }
      idx = 0;
      for (Operation op : targets) {
        targetOpHandles[idx++] = op.getUnsafeNativeHandle();
      }
      try (Reference runref = new Reference()) {
        Session.run(
            nativeHandle,
            null, /* runOptions */
            inputTensorHandles,
            inputOpHandles,
            inputOpIndices,
            outputOpHandles,
            outputOpIndices,
            targetOpHandles,
            false, /* wantRunMetadata */
            outputTensorHandles);
      }
      List<Tensor> ret = new ArrayList<Tensor>();
      for (long h : outputTensorHandles) {
        try {
          ret.add(Tensor.fromHandle(h));
        } catch (Exception e) {
          for (Tensor t : ret) {
            t.close();
          }
          ret.clear();
          throw e;
        }
      }
      return ret;
    }

    private class Reference implements AutoCloseable {
      public Reference() {
        synchronized (nativeHandleLock) {
          if (nativeHandle == 0) {
            throw new IllegalStateException("run() cannot be called on the Session after close()");
          }
          ++numActiveRuns;
        }
      }

      @Override
      public void close() {
        synchronized (nativeHandleLock) {
          if (nativeHandle == 0) {
            return;
          }
          if (--numActiveRuns == 0) {
            nativeHandleLock.notifyAll();
          }
        }
      }
    }

    private Operation operationByName(String opName) {
      Operation op = graph.operation(opName);
      if (op == null) {
        throw new IllegalArgumentException("No Operation named [" + opName + "] in the Graph");
      }
      return op;
    }

    private ArrayList<Output> inputs = new ArrayList<Output>();
    private ArrayList<Tensor> inputTensors = new ArrayList<Tensor>();
    private ArrayList<Output> outputs = new ArrayList<Output>();
    private ArrayList<Operation> targets = new ArrayList<Operation>();
  }

  /** Create a Runner to execute graph operations and evaluate Tensors. */
  public Runner runner() {
    return new Runner();
  }

  private final Graph graph;
  private final Graph.Reference graphRef;

  private final Object nativeHandleLock = new Object();
  private long nativeHandle;
  private int numActiveRuns;

  private static native long allocate(long graphHandle);

  private static native void delete(long handle);

  /**
   * Execute a session.
   *
   * <p>The author apologizes for the ugliness of the long argument list of this method. However,
   * take solace in the fact that this is a private method meant to cross the JNI boundary.
   *
   * @param handle to the C API TF_Session object (Session.nativeHandle)
   * @param runOptions serialized representation of a RunOptions protocol buffer, or null
   * @param inputOpHandles (see inputOpIndices)
   * @param inputOpIndices (see inputTensorHandles)
   * @param inputTensorHandles together with inputOpHandles and inputOpIndices specifies the values
   *     that are being "fed" (do not need to be computed) during graph execution.
   *     inputTensorHandles[i] (which correponds to a Tensor.nativeHandle) is considered to be the
   *     inputOpIndices[i]-th output of the Operation inputOpHandles[i]. Thus, it is required that
   *     inputOpHandles.length == inputOpIndices.length == inputTensorHandles.length.
   * @param outputOpHandles (see outputOpIndices)
   * @param outputOpIndices together with outputOpHandles identifies the set of values that should
   *     be computed. The outputOpIndices[i]-th output of the Operation outputOpHandles[i], It is
   *     required that outputOpHandles.length == outputOpIndices.length.
   * @param targetOpHandles is the set of Operations in the graph that are to be executed but whose
   *     output will not be returned
   * @param wantRunMetadata indicates whether metadata about this execution should be returned.
   * @param outputTensorHandles will be filled in with handles to the outputs requested. It is
   *     required that outputTensorHandles.length == outputOpHandles.length.
   * @return if wantRunMetadata is true, serialized representation of the RunMetadata protocol
   *     buffer, false otherwise.
   */
  private static native byte[] run(
      long handle,
      byte[] runOptions,
      long[] inputTensorHandles,
      long[] inputOpHandles,
      int[] inputOpIndices,
      long[] outputOpHandles,
      int[] outputOpIndices,
      long[] targetOpHandles,
      boolean wantRunMetadata,
      long[] outputTensorHandles);
}
