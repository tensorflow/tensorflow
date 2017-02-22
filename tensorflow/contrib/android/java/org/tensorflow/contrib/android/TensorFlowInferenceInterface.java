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

package org.tensorflow.contrib.android;

import android.content.res.AssetManager;
import android.os.Trace;
import android.util.Log;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

/**
 * Wrapper over the TensorFlow API ({@link Graph}, {@link Session}) providing a smaller API surface
 * for inference.
 *
 * <p>See tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowImageClassifier.java for an
 * example usage.
 */
public class TensorFlowInferenceInterface {
  private static final String TAG = "TensorFlowInferenceInterface";
  private static final String ASSET_FILE_PREFIX = "file:///android_asset/";

  public TensorFlowInferenceInterface() {
    try {
      // Hack to see if the native libraries have been loaded.
      new RunStats();
      Log.i(TAG, "Native methods already loaded.");
    } catch (UnsatisfiedLinkError e1) {
      Log.i(TAG, "Loading tensorflow_inference.");
      try {
        System.loadLibrary("tensorflow_inference");
      } catch (UnsatisfiedLinkError e2) {
        throw new RuntimeException(
            "Native TF methods not found; check that the correct native"
                + " libraries are present and loaded.");
      }
    }
  }

  /**
   * Load a TensorFlow model from the AssetManager or from disk if it is not an asset file.
   *
   * @param assetManager The AssetManager to use to load the model file.
   * @param model The filepath to the GraphDef proto representing the model.
   * @return 0 on success.
   */
  public int initializeTensorFlow(AssetManager assetManager, String model) {
    final boolean hasAssetPrefix = model.startsWith(ASSET_FILE_PREFIX);
    InputStream is = null;
    try {
      String aname = hasAssetPrefix ? model.split(ASSET_FILE_PREFIX)[1] : model;
      is = assetManager.open(aname);
    } catch (IOException e) {
      if (hasAssetPrefix) {
        Log.e(TAG, "Failed to initialize: " + e.toString());
        return 1;
      }
      // Perhaps the model file is not an asset but is on disk.
      try {
        is = new FileInputStream(model);
      } catch (IOException e2) {
        Log.e(TAG, "Failed to open " + model + ": " + e2.toString());
        return 1;
      }
    }
    try {
      load(is);
      is.close();
      return 0;
    } catch (IOException e) {
      Log.e(TAG, "Failed to initialize: " + e.toString());
      return 1;
    }
  }

  /**
   * Runs inference between the previously registered input nodes (via fillNode*) and the requested
   * output nodes. Output nodes can then be queried with the readNode* methods.
   *
   * @param outputNames A list of output nodes which should be filled by the inference pass.
   * @return 0 on success.
   */
  public int runInference(String[] outputNames) {
    // Release any Tensors from the previous runInference calls.
    closeFetches();

    // Add fetches.
    for (String o : outputNames) {
      fetchNames.add(o);
      TensorId tid = TensorId.parse(o);
      runner.fetch(tid.name, tid.outputIndex);
    }

    // Run the session.
    try {
      if (enableStats) {
        Session.Run r = runner.setOptions(RunStats.runOptions()).runAndFetchMetadata();
        fetchTensors = r.outputs;
        runStats.add(r.metadata);
      } else {
        fetchTensors = runner.run();
      }
    } catch (RuntimeException e) {
      // Ideally the exception would have been let through, but since this interface predates the
      // TensorFlow Java API, must return -1.
      Log.e(TAG, "Failed to run TensorFlow session: " + e.toString());
      return -1;
    } finally {
      // Always release the feeds (to save resources) and reset the runner, this runInference is
      // over.
      closeFeeds();
      runner = sess.runner();
    }

    return 0;
  }

  /** Returns a reference to the Graph describing the computation run during inference. */
  public Graph graph() {
    return g;
  }

  /**
   * Whether to collect stats during inference. This should only be enabled when needed, as it will
   * add overhead.
   */
  public void enableStatLogging(boolean enabled) {
    enableStats = enabled;
    if (enableStats && runStats == null) {
      runStats = new RunStats();
    }
  }

  /** Returns the last stat summary string if logging is enabled. */
  public String getStatString() {
    return (runStats == null) ? "" : runStats.summary();
  }

  /**
   * Cleans up the state associated with this Object. initializeTensorFlow() can then be called
   * again to initialize a new session.
   */
  public void close() {
    closeFeeds();
    closeFetches();
    sess.close();
    g.close();
    if (runStats != null) {
      runStats.close();
    }
    runStats = null;
    enableStats = false;
  }

  // Methods for taking a native Tensor and filling it with values from Java arrays.

  /**
   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void fillNodeFloat(String inputName, int[] dims, float[] src) {
    addFeed(inputName, Tensor.create(mkDims(dims), FloatBuffer.wrap(src)));
  }

  /**
   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void fillNodeInt(String inputName, int[] dims, int[] src) {
    addFeed(inputName, Tensor.create(mkDims(dims), IntBuffer.wrap(src)));
  }

  /**
   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void fillNodeDouble(String inputName, int[] dims, double[] src) {
    addFeed(inputName, Tensor.create(mkDims(dims), DoubleBuffer.wrap(src)));
  }

  /**
   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void fillNodeByte(String inputName, int[] dims, byte[] src) {
    addFeed(inputName, Tensor.create(DataType.UINT8, mkDims(dims), ByteBuffer.wrap(src)));
  }

  // Methods for taking a native Tensor and filling it with src from Java native IO buffers.

  /**
   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
   * elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void fillNodeFromFloatBuffer(String inputName, IntBuffer dims, FloatBuffer src) {
    addFeed(inputName, Tensor.create(mkDims(dims), src));
  }

  /**
   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
   * elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void fillNodeFromIntBuffer(String inputName, IntBuffer dims, IntBuffer src) {
    addFeed(inputName, Tensor.create(mkDims(dims), src));
  }

  /**
   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
   * elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void fillNodeFromDoubleBuffer(String inputName, IntBuffer dims, DoubleBuffer src) {
    addFeed(inputName, Tensor.create(mkDims(dims), src));
  }

  /**
   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
   * elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void fillNodeFromByteBuffer(String inputName, IntBuffer dims, ByteBuffer src) {
    addFeed(inputName, Tensor.create(DataType.UINT8, mkDims(dims), src));
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
   * dst} must have length greater than or equal to that of the source Tensor. This operation will
   * not affect dst's content past the source Tensor's size.
   *
   * @return 0 on success, -1 on failure.
   */
  public int readNodeFloat(String outputName, float[] dst) {
    return readNodeIntoFloatBuffer(outputName, FloatBuffer.wrap(dst));
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
   * dst} must have length greater than or equal to that of the source Tensor. This operation will
   * not affect dst's content past the source Tensor's size.
   *
   * @return 0 on success, -1 on failure.
   */
  public int readNodeInt(String outputName, int[] dst) {
    return readNodeIntoIntBuffer(outputName, IntBuffer.wrap(dst));
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
   * dst} must have length greater than or equal to that of the source Tensor. This operation will
   * not affect dst's content past the source Tensor's size.
   *
   * @return 0 on success, -1 on failure.
   */
  public int readNodeDouble(String outputName, double[] dst) {
    return readNodeIntoDoubleBuffer(outputName, DoubleBuffer.wrap(dst));
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
   * dst} must have length greater than or equal to that of the source Tensor. This operation will
   * not affect dst's content past the source Tensor's size.
   *
   * @return 0 on success, -1 on failure.
   */
  public int readNodeByte(String outputName, byte[] dst) {
    return readNodeIntoByteBuffer(outputName, ByteBuffer.wrap(dst));
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
   * or equal to that of the source Tensor. This operation will not affect dst's content past the
   * source Tensor's size.
   *
   * @return 0 on success, -1 on failure.
   */
  public int readNodeIntoFloatBuffer(String outputName, FloatBuffer dst) {
    Tensor t = getTensor(outputName);
    if (t == null) {
      return -1;
    }
    t.writeTo(dst);
    return 0;
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
   * or equal to that of the source Tensor. This operation will not affect dst's content past the
   * source Tensor's size.
   *
   * @return 0 on success, -1 on failure.
   */
  public int readNodeIntoIntBuffer(String outputName, IntBuffer dst) {
    Tensor t = getTensor(outputName);
    if (t == null) {
      return -1;
    }
    t.writeTo(dst);
    return 0;
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
   * or equal to that of the source Tensor. This operation will not affect dst's content past the
   * source Tensor's size.
   *
   * @return 0 on success, -1 on failure.
   */
  public int readNodeIntoDoubleBuffer(String outputName, DoubleBuffer dst) {
    Tensor t = getTensor(outputName);
    if (t == null) {
      return -1;
    }
    t.writeTo(dst);
    return 0;
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
   * or equal to that of the source Tensor. This operation will not affect dst's content past the
   * source Tensor's size.
   *
   * @return 0 on success, -1 on failure.
   */
  public int readNodeIntoByteBuffer(String outputName, ByteBuffer dst) {
    Tensor t = getTensor(outputName);
    if (t == null) {
      return -1;
    }
    t.writeTo(dst);
    return 0;
  }

  private void load(InputStream is) throws IOException {
    this.g = new Graph();
    this.sess = new Session(g);
    this.runner = sess.runner();
    final long startMs = System.currentTimeMillis();

    Trace.beginSection("initializeTensorFlow");

    Trace.beginSection("readGraphDef");
    // TODO(ashankar): Can we somehow mmap the contents instead of copying them?
    byte[] graphDef = new byte[is.available()];
    final int numBytesRead = is.read(graphDef);
    if (numBytesRead != graphDef.length) {
      throw new IOException(
          "read error: read only "
              + numBytesRead
              + " of the graph, expected to read "
              + graphDef.length);
    }
    Trace.endSection();

    Trace.beginSection("importGraphDef");
    try {
      g.importGraphDef(graphDef);
    } catch (IllegalArgumentException e) {
      throw new IOException("Not a valid TensorFlow Graph serialization: " + e.getMessage());
    }
    Trace.endSection();

    Trace.endSection(); // initializeTensorFlow.

    final long endMs = System.currentTimeMillis();
    Log.i(
        TAG,
        "Model load took " + (startMs - endMs) + "ms, TensorFlow version: " + TensorFlow.version());
  }

  // The TensorFlowInferenceInterface API used int[] for dims, but the underlying TensorFlow runtime
  // allows for 64-bit dimension sizes, so it needs to be converted to a long[]
  private long[] mkDims(int[] dims) {
    long[] ret = new long[dims.length];
    for (int i = 0; i < dims.length; ++i) {
      ret[i] = (long) dims[i];
    }
    return ret;
  }

  // Similar to mkDims(int[]), with the shape provided in an IntBuffer.
  private long[] mkDims(IntBuffer dims) {
    if (dims.hasArray()) {
      return mkDims(dims.array());
    }
    int[] copy = new int[dims.remaining()];
    dims.duplicate().get(copy);
    return mkDims(copy);
  }

  private void addFeed(String inputName, Tensor t) {
    // The string format accepted by TensorFlowInferenceInterface is node_name[:output_index].
    TensorId tid = TensorId.parse(inputName);
    runner.feed(tid.name, tid.outputIndex, t);
    feedTensors.add(t);
  }

  private static class TensorId {
    String name;
    int outputIndex;

    // Parse output names into a TensorId.
    //
    // E.g., "foo" --> ("foo", 0), while "foo:1" --> ("foo", 1)
    public static TensorId parse(String name) {
      TensorId tid = new TensorId();
      int colonIndex = name.lastIndexOf(':');
      if (colonIndex < 0) {
        tid.outputIndex = 0;
        tid.name = name;
        return tid;
      }
      try {
        tid.outputIndex = Integer.parseInt(name.substring(colonIndex + 1));
        tid.name = name.substring(0, colonIndex);
      } catch (NumberFormatException e) {
        tid.outputIndex = 0;
        tid.name = name;
      }
      return tid;
    }
  }

  private Tensor getTensor(String outputName) {
    int i = 0;
    for (String n : fetchNames) {
      if (n.equals(outputName)) {
        return fetchTensors.get(i);
      }
      i++;
    }
    return null;
  }

  private void closeFeeds() {
    for (Tensor t : feedTensors) {
      t.close();
    }
    feedTensors.clear();
  }

  private void closeFetches() {
    for (Tensor t : fetchTensors) {
      t.close();
    }
    fetchTensors.clear();
    fetchNames.clear();
  }

  // State immutable between initializeTensorFlow calls.
  private Graph g;
  private Session sess;

  // State reset on every call to runInference.
  private Session.Runner runner;
  private List<Tensor> feedTensors = new ArrayList<Tensor>();
  private List<String> fetchNames = new ArrayList<String>();
  private List<Tensor> fetchTensors = new ArrayList<Tensor>();

  // Mutable state.
  private boolean enableStats;
  private RunStats runStats;
}
