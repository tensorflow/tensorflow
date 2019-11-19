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
 * {@link Graph} are executed to compute {@link Tensor Tensors}. For example:
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
 * <p><b>WARNING:</b>A {@code Session} owns resources that <b>must</b> be explicitly freed by
 * invoking {@link #close()}.
 *
 * <p>Instances of a Session are thread-safe.
 */
public final class Session implements AutoCloseable {

  /** Construct a new session with the associated {@link Graph}. */
  public Session(Graph g) {
    this(g, null);
  }

  /**
   * Construct a new session with the associated {@link Graph} and configuration options.
   *
   * @param g The {@link Graph} the created Session will operate on.
   * @param config Configuration parameters for the session specified as a serialized <a
   *     href="https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto">ConfigProto</a>
   *     protocol buffer.
   * @throws IllegalArgumentException if the config is not a valid serialization of the ConfigProto
   *     protocol buffer.
   */
  public Session(Graph g, byte[] config) {
    graph = g;
    Graph.Reference r = g.ref();
    try {
      nativeHandle =
          (config == null) ? allocate(r.nativeHandle()) : allocate2(r.nativeHandle(), null, config);
      graphRef = g.ref();
    } finally {
      r.close();
    }
  }

  /** Wrap an existing session with the associated {@link Graph}. */
  Session(Graph g, long nativeHandle) {
    graph = g;
    this.nativeHandle = nativeHandle;
    graphRef = g.ref();
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
   * Run {@link Operation}s and evaluate {@link Tensor Tensors}.
   *
   * <p>A Runner runs the necessary graph fragments to execute every {@link Operation} required to
   * evaluate the {@link Tensor Tensors} to fetch. The {@link #feed(String,int,Tensor)} call allows
   * callers to override the value of {@link Tensor Tensors} in the graph by substituting the
   * provided {@link Tensor Tensors} for the outputs of the operations provided to {@link
   * #feed(String,int,Tensor)}.
   */
  public final class Runner {
    /**
     * Avoid evaluating {@code operation} and substitute {@code t} for the value it produces.
     *
     * @param operation Is either the string name of the operation, in which case this method is a
     *     shorthand for {@code feed(operation, 0)}, or it is a string of the form
     *     <tt>operation_name:output_index</tt> , in which case this method acts like {@code
     *     feed(operation_name, output_index)}. These colon-separated names are commonly used in the
     *     {@code SignatureDef} protocol buffer messages that are included in {@link
     *     SavedModelBundle#metaGraphDef()}.
     */
    public Runner feed(String operation, Tensor<?> t) {
      return feed(parseOutput(operation), t);
    }

    /**
     * Avoid evaluating the {@code index}-th output of {@code operation} by substituting {@code t}
     * for the value it produces.
     *
     * <p>Operations in a {@link Graph} can have multiple outputs, {@code index} identifies which
     * one {@code t} is being provided for.
     */
    public Runner feed(String operation, int index, Tensor<?> t) {
      Operation op = operationByName(operation);
      if (op != null) {
        inputs.add(op.output(index));
        inputTensors.add(t);
      }
      return this;
    }

    /**
     * Use {@code t} instead of the Tensor referred to by executing the operation referred to by
     * {@code operand}.
     */
    public Runner feed(Operand<?> operand, Tensor<?> t) {
      inputs.add(operand.asOutput());
      inputTensors.add(t);
      return this;
    }

    /**
     * Make {@link #run()} return the output of {@code operation}.
     *
     * @param operation Is either the string name of the operation, in which case this method is a
     *     shorthand for {@code fetch(operation, 0)}, or it is a string of the form
     *     <tt>operation_name:output_index</tt> , in which case this method acts like {@code
     *     fetch(operation_name, output_index)}. These colon-separated names are commonly used in
     *     the {@code SignatureDef} protocol buffer messages that are included in {@link
     *     SavedModelBundle#metaGraphDef()}.
     */
    public Runner fetch(String operation) {
      return fetch(parseOutput(operation));
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
     * Makes {@link #run()} return the Tensor referred to by {@code output}. 
     */
    public Runner fetch(Output<?> output) {
      outputs.add(output);
      return this;
    }
    
    /**
     * Makes {@link #run()} return the Tensor referred to by the output of {@code operand}. 
     */
    public Runner fetch(Operand<?> operand) {
      return fetch(operand.asOutput());
    }

    /**
     * Make {@link #run()} execute {@code operation}, but not return any evaluated {@link Tensor
     * Tensors}.
     */
    public Runner addTarget(String operation) {
      GraphOperation op = operationByName(operation);
      if (op != null) {
        targets.add(op);
      }
      return this;
    }

    /**
     * Make {@link #run()} execute {@code operation}, but not return any evaluated {@link Tensor
     * Tensors}.
     *
     * @throws IllegalArgumentException if the operation is not a {@link GraphOperation}
     */
    public Runner addTarget(Operation operation) {
      if (!(operation instanceof GraphOperation)) {
        throw new IllegalArgumentException(
            "Operation of type "
                + operation.getClass().getName()
                + " is not supported in graph sessions");
      }
      targets.add((GraphOperation) operation);
      return this;
    }

    /**
     * Make {@link #run} execute {@code operand}, but not return any evaluated {@link Tensor
     * Tensors}.
     */
    public Runner addTarget(Operand<?> operand) {
      return addTarget(operand.asOutput().op());
    }

    /**
     * (Experimental method): set options (typically for debugging) for this run.
     *
     * <p>The options are presented as a serialized <a
     * href="https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto">RunOptions
     * protocol buffer</a>.
     *
     * <p>The org.tensorflow package is free of any protocol buffer dependencies in order to remain
     * friendly to resource constrained systems (where something like <a
     * href="https://github.com/google/protobuf/tree/master/javanano#nano-version">nanoproto</a> may
     * be more appropriate). A cost of that is this lack of type-safety in this API function. This
     * choice is under review and this function may be replaced by more type-safe equivalents at any
     * time.
     */
    public Runner setOptions(byte[] options) {
      this.runOptions = options;
      return this;
    }

    /**
     * Execute the graph fragments necessary to compute all requested fetches.
     *
     * <p><b>WARNING:</b> The caller assumes ownership of all returned {@link Tensor Tensors}, i.e.,
     * the caller must call {@link Tensor#close} on all elements of the returned list to free up
     * resources.
     *
     * <p>TODO(ashankar): Reconsider the return type here. Two things in particular: (a) Make it
     * easier for the caller to cleanup (perhaps returning something like AutoCloseableList in
     * SessionTest.java), and (b) Evaluate whether the return value should be a list, or maybe a
     * {@code Map<Output, Tensor>}?
     *
     * <p>TODO(andrewmyers): It would also be good if whatever is returned here made it easier to
     * extract output tensors in a type-safe way.
     */
    public List<Tensor<?>> run() {
      return runHelper(false).outputs;
    }

    /**
     * Execute graph fragments to compute requested fetches and return metadata about the run.
     *
     * <p>This is exactly like {@link #run()}, but in addition to the requested Tensors, also
     * returns metadata about the graph execution in the form of a serialized <a
     * href="https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto">RunMetadata
     * protocol buffer</a>.
     */
    public Run runAndFetchMetadata() {
      return runHelper(true);
    }

    private Run runHelper(boolean wantMetadata) {
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
      for (Tensor<?> t : inputTensors) {
        inputTensorHandles[idx++] = t.getNativeHandle();
      }
      idx = 0;
      for (Output<?> o : inputs) {
        inputOpHandles[idx] = o.getUnsafeNativeHandle();
        inputOpIndices[idx] = o.index();
        idx++;
      }
      idx = 0;
      for (Output<?> o : outputs) {
        outputOpHandles[idx] = o.getUnsafeNativeHandle();
        outputOpIndices[idx] = o.index();
        idx++;
      }
      idx = 0;
      for (GraphOperation op : targets) {
        targetOpHandles[idx++] = op.getUnsafeNativeHandle();
      }
      Reference runRef = new Reference();
      byte[] metadata = null;
      try {
        metadata =
            Session.run(
                nativeHandle,
                runOptions,
                inputTensorHandles,
                inputOpHandles,
                inputOpIndices,
                outputOpHandles,
                outputOpIndices,
                targetOpHandles,
                wantMetadata,
                outputTensorHandles);
      } finally {
        runRef.close();
      }
      List<Tensor<?>> outputs = new ArrayList<Tensor<?>>();
      for (long h : outputTensorHandles) {
        try {
          outputs.add(Tensor.fromHandle(h));
        } catch (Exception e) {
          for (Tensor<?> t : outputs) {
            t.close();
          }
          outputs.clear();
          throw e;
        }
      }
      Run ret = new Run();
      ret.outputs = outputs;
      ret.metadata = metadata;
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

    private GraphOperation operationByName(String opName) {
      GraphOperation op = graph.operation(opName);
      if (op == null) {
        throw new IllegalArgumentException("No Operation named [" + opName + "] in the Graph");
      }
      return op;
    }

    @SuppressWarnings("rawtypes")
    private Output<?> parseOutput(String opName) {
      int colon = opName.lastIndexOf(':');
      if (colon == -1 || colon == opName.length() - 1) {
        return new Output(operationByName(opName), 0);
      }
      try {
        String op = opName.substring(0, colon);
        int index = Integer.parseInt(opName.substring(colon + 1));
        return new Output(operationByName(op), index);
      } catch (NumberFormatException e) {
        return new Output(operationByName(opName), 0);
      }
    }

    private ArrayList<Output<?>> inputs = new ArrayList<Output<?>>();
    private ArrayList<Tensor<?>> inputTensors = new ArrayList<Tensor<?>>();
    private ArrayList<Output<?>> outputs = new ArrayList<Output<?>>();
    private ArrayList<GraphOperation> targets = new ArrayList<GraphOperation>();
    private byte[] runOptions = null;
  }

  /** Create a Runner to execute graph operations and evaluate Tensors. */
  public Runner runner() {
    return new Runner();
  }

  /**
   * Output tensors and metadata obtained when executing a session.
   *
   * <p>See {@link Runner#runAndFetchMetadata()}
   */
  public static final class Run {
    /** Tensors from requested fetches. */
    public List<Tensor<?>> outputs;

    /**
     * (Experimental): Metadata about the run.
     *
     * <p>A serialized <a
     * href="https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto">RunMetadata
     * protocol buffer</a>. The org.tensorflow package is free of any protocol buffer dependencies
     * in order to remain friendly to resource constrained systems (where something like <a
     * href="https://github.com/google/protobuf/tree/master/javanano#nano-version">nanoproto</a> may
     * be more appropriate). A cost of that is this opaque blob. This choice is under review and
     * this field may be replaced by more type-safe equivalents at any time.
     */
    public byte[] metadata;
  }

  private final Graph graph;
  private final Graph.Reference graphRef;

  private final Object nativeHandleLock = new Object();
  private long nativeHandle;
  private int numActiveRuns;

  // TODO(ashankar): Remove after TensorFlow 1.2 has been released with allocate2().
  private static native long allocate(long graphHandle);

  private static native long allocate2(long graphHandle, String target, byte[] config);

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
   *     inputTensorHandles[i] (which corresponds to a Tensor.nativeHandle) is considered to be the
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
