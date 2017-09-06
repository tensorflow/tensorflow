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

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Build.VERSION;
import android.os.Trace;
import android.text.TextUtils;
import android.util.Log;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

/**
 * Wrapper over the TensorFlow API ({@link Graph}, {@link Session}) providing a smaller API surface
 * for inference.
 * <p>
 * <p>See tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowImageClassifier.java for an
 * example usage.
 */
public class TensorFlowInferenceInterface {
    //TAG should be <=23 chars
    private static final String TAG = "TensorFlowInferenceInt";
    //@todo remove
    static File mModelFile = null;

    // @todo refactor
    public static TensorFlowInferenceInterface createTensorFlowInferenceInterface(Context ctx, String model) {
        mModelFile = ctx.getFileStreamPath(model);
        return new TensorFlowInferenceInterface(null, model);
    }

    /* Load a TensorFlow model from the AssetManager or from disk if it is not an asset file.
    *
    * @param assetManager The AssetManager to use to load the model file.
    * @param model The filepath to the GraphDef proto representing the model.`
    */

    public TensorFlowInferenceInterface(AssetManager assetManager, String model) {
        //@todo update usages-- include the copy function here instead of init?
        //@todo or just remove the am
        //@todo cache object
        //@todo add permsission for external storage

        Log.i(TAG, "Checking if TensorFlow native methods are already loaded");
        try {
            // Hack to see if the native libraries have been loaded.
            new RunStats();
            Log.i(TAG, "TensorFlow native methods already loaded");
        } catch (UnsatisfiedLinkError e1) {
            Log.i(
                    TAG, "TensorFlow native methods not found, attempting to load via tensorflow_inference");
            try {
                System.loadLibrary("tensorflow_inference");
                Log.i(TAG, "Successfully loaded TensorFlow native methods (RunStats error may be ignored)");
            } catch (UnsatisfiedLinkError e2) {
                throw new RuntimeException(
                        "Native TF methods not found; check that the correct native"
                                + " libraries are present in the APK.");
            }
        }

        this.modelName = model;
        this.g = new Graph();
        this.sess = new Session(g);
        this.runner = sess.runner();

        MappedByteBuffer buffer = null;
        FileChannel fileChannel = null;
        RandomAccessFile raf = null;
        try {
            raf = new RandomAccessFile(mModelFile, "r");
            fileChannel = raf.getChannel();
            buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
        } catch (IOException e) {
            throw new RuntimeException("Failed to load model from: " + mModelFile, e);

        }
        try {
            //buffer.array() is not written to work with mem map
            //load () can cause IO ops to kick off
            if (!buffer.isLoaded()) {
                buffer.load();
            }
            //no guarantee but can be faster than normal IO reads
            byte[] graphDef = new byte[buffer.limit()];
            //for (int i = 0; i < buffer.limit(); i++) {
            //     graphDef[i] = buffer.get();
            //}
            //bulk method
            buffer.get(graphDef);
            loadGraph(graphDef, g);

            Log.i(TAG, "Successfully loaded model from map '" + model + "'");
        } catch (IOException e) {
            throw new RuntimeException("Failed to load model from '" + model + "'", e);
        } finally {
            try {

                raf.close();
            } catch (IOException e) {
                throw new RuntimeException("Failed to close FileChannel '" + fileChannel + "'", e);
            }
            try {
                //@todo individual exceptions
                fileChannel.close();
            } catch (IOException e) {
                throw new RuntimeException("Failed to close RAF '" + fileChannel + "'", e);
            }
            if (buffer != null) {
                buffer.clear();
                buffer = null;
            }
            /*@todo maybe leave this out--only 4 cached objs, assumes this must only be called once per type
            if (mModelFile != null) {
                mModelFile.delete();
            }*/


        }

    }

    /**
     * Runs inference between the previously registered input nodes (via feed*) and the requested
     * output nodes. Output nodes can then be queried with the fetch* methods.
     *
     * @param outputNames A zlist of output nodes which should be filled by the inference pass.
     */
    public void run(String[] outputNames) {
        run(outputNames, false);
    }

    /**
     * Runs inference between the previously registered input nodes (via feed*) and the requested
     * output nodes. Output nodes can then be queried with the fetch* methods.
     *
     * @param outputNames A list of output nodes which should be filled by the inference pass.
     */
    public void run(String[] outputNames, boolean enableStats) {
        // Release any Tensors from the previous run calls.
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

                if (runStats == null) {
                    runStats = new RunStats();
                }
                runStats.add(r.metadata);
            } else {
                fetchTensors = runner.run();
            }
        } catch (RuntimeException e) {
            // Ideally the exception would have been let through, but since this interface predates the
            // TensorFlow Java API, must return -1.
            Log.e(
                    TAG,
                    "Failed to run TensorFlow inference with inputs:["
                            + TextUtils.join(", ", feedNames)
                            + "], outputs:["
                            + TextUtils.join(", ", fetchNames)
                            + "]");
            throw e;
        } finally {
            // Always release the feeds (to save resources) and reset the runner, this run is
            // over.
            closeFeeds();
            runner = sess.runner();
        }
    }

    /**
     * Returns a reference to the Graph describing the computation run during inference.
     */
    public Graph graph() {
        return g;
    }

    public Operation graphOperation(String operationName) {
        final Operation operation = g.operation(operationName);
        if (operation == null) {
            throw new RuntimeException(
                    "Node '" + operationName + "' does not exist in model '" + modelName + "'");
        }
        return operation;
    }

    /**
     * Returns the last stat summary string if logging is enabled.
     */
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
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }

    // Methods for taking a native Tensor and filling it with values from Java arrays.

    /**
     * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
     * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
     * as many elements as that of the destination Tensor. If {@link src} has more elements than the
     * destination has capacity, the copy is truncated.
     */
    public void feed(String inputName, float[] src, long... dims) {
        addFeed(inputName, Tensor.create(dims, FloatBuffer.wrap(src)));
    }

    /**
     * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
     * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
     * as many elements as that of the destination Tensor. If {@link src} has more elements than the
     * destination has capacity, the copy is truncated.
     */
    public void feed(String inputName, int[] src, long... dims) {
        addFeed(inputName, Tensor.create(dims, IntBuffer.wrap(src)));
    }

    /**
     * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
     * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
     * as many elements as that of the destination Tensor. If {@link src} has more elements than the
     * destination has capacity, the copy is truncated.
     */
    public void feed(String inputName, long[] src, long... dims) {
        addFeed(inputName, Tensor.create(dims, LongBuffer.wrap(src)));
    }

    /**
     * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
     * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
     * as many elements as that of the destination Tensor. If {@link src} has more elements than the
     * destination has capacity, the copy is truncated.
     */
    public void feed(String inputName, double[] src, long... dims) {
        addFeed(inputName, Tensor.create(dims, DoubleBuffer.wrap(src)));
    }

    /**
     * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
     * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
     * as many elements as that of the destination Tensor. If {@link src} has more elements than the
     * destination has capacity, the copy is truncated.
     */
    public void feed(String inputName, byte[] src, long... dims) {
        addFeed(inputName, Tensor.create(DataType.UINT8, dims, ByteBuffer.wrap(src)));
    }

    // Methods for taking a native Tensor and filling it with src from Java native IO buffers.

    /**
     * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
     * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
     * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
     * elements as that of the destination Tensor. If {@link src} has more elements than the
     * destination has capacity, the copy is truncated.
     */
    public void feed(String inputName, FloatBuffer src, long... dims) {
        addFeed(inputName, Tensor.create(dims, src));
    }

    /**
     * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
     * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
     * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
     * elements as that of the destination Tensor. If {@link src} has more elements than the
     * destination has capacity, the copy is truncated.
     */
    public void feed(String inputName, IntBuffer src, long... dims) {
        addFeed(inputName, Tensor.create(dims, src));
    }

    /**
     * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
     * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
     * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
     * elements as that of the destination Tensor. If {@link src} has more elements than the
     * destination has capacity, the copy is truncated.
     */
    public void feed(String inputName, LongBuffer src, long... dims) {
        addFeed(inputName, Tensor.create(dims, src));
    }

    /**
     * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
     * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
     * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
     * elements as that of the destination Tensor. If {@link src} has more elements than the
     * destination has capacity, the copy is truncated.
     */
    public void feed(String inputName, DoubleBuffer src, long... dims) {
        addFeed(inputName, Tensor.create(dims, src));
    }

    /**
     * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
     * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
     * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
     * elements as that of the destination Tensor. If {@link src} has more elements than the
     * destination has capacity, the copy is truncated.
     */
    public void feed(String inputName, ByteBuffer src, long... dims) {
        addFeed(inputName, Tensor.create(DataType.UINT8, dims, src));
    }

    /**
     * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
     * dst} must have length greater than or equal to that of the source Tensor. This operation will
     * not affect dst's content past the source Tensor's size.
     */
    public void fetch(String outputName, float[] dst) {
        fetch(outputName, FloatBuffer.wrap(dst));
    }

    /**
     * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
     * dst} must have length greater than or equal to that of the source Tensor. This operation will
     * not affect dst's content past the source Tensor's size.
     */
    public void fetch(String outputName, int[] dst) {
        fetch(outputName, IntBuffer.wrap(dst));
    }

    /**
     * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
     * dst} must have length greater than or equal to that of the source Tensor. This operation will
     * not affect dst's content past the source Tensor's size.
     */
    public void fetch(String outputName, long[] dst) {
        fetch(outputName, LongBuffer.wrap(dst));
    }

    /**
     * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
     * dst} must have length greater than or equal to that of the source Tensor. This operation will
     * not affect dst's content past the source Tensor's size.
     */
    public void fetch(String outputName, double[] dst) {
        fetch(outputName, DoubleBuffer.wrap(dst));
    }

    /**
     * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
     * dst} must have length greater than or equal to that of the source Tensor. This operation will
     * not affect dst's content past the source Tensor's size.
     */
    public void fetch(String outputName, byte[] dst) {
        fetch(outputName, ByteBuffer.wrap(dst));
    }

    /**
     * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
     * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
     * or equal to that of the source Tensor. This operation will not affect dst's content past the
     * source Tensor's size.
     */
    public void fetch(String outputName, FloatBuffer dst) {
        getTensor(outputName).writeTo(dst);
    }

    /**
     * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
     * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
     * or equal to that of the source Tensor. This operation will not affect dst's content past the
     * source Tensor's size.
     */
    public void fetch(String outputName, IntBuffer dst) {
        getTensor(outputName).writeTo(dst);
    }

    /**
     * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
     * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
     * or equal to that of the source Tensor. This operation will not affect dst's content past the
     * source Tensor's size.
     */
    public void fetch(String outputName, LongBuffer dst) {
        getTensor(outputName).writeTo(dst);
    }

    /**
     * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
     * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
     * or equal to that of the source Tensor. This operation will not affect dst's content past the
     * source Tensor's size.
     */
    public void fetch(String outputName, DoubleBuffer dst) {
        getTensor(outputName).writeTo(dst);
    }

    /**
     * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
     * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
     * or equal to that of the source Tensor. This operation will not affect dst's content past the
     * source Tensor's size.
     */
    public void fetch(String outputName, ByteBuffer dst) {
        getTensor(outputName).writeTo(dst);
    }

    private void loadGraph(byte[] graphDef, Graph g) throws IOException {
        final long startMs = System.currentTimeMillis();

        if (VERSION.SDK_INT >= 18) {
            Trace.beginSection("initializeTensorFlow");
            Trace.beginSection("readGraphDef");
        }

        // TODO(ashankar): Can we somehow mmap the contents instead of copying them?
        //yes

        if (VERSION.SDK_INT >= 18) {
            Trace.endSection(); // readGraphDef.
            Trace.beginSection("importGraphDef");
        }

        try {
            g.importGraphDef(graphDef);
        } catch (IllegalArgumentException e) {
            throw new IOException("Not a valid TensorFlow Graph serialization: " + e.getMessage());
        }

        if (VERSION.SDK_INT >= 18) {
            Trace.endSection(); // importGraphDef.
            Trace.endSection(); // initializeTensorFlow.
        }

        final long endMs = System.currentTimeMillis();
        Log.i(
                TAG,
                "Model load took " + (endMs - startMs) + "ms, TensorFlow version: " + TensorFlow.version());
    }

    private void addFeed(String inputName, Tensor t) {
        // The string format accepted by TensorFlowInferenceInterface is node_name[:output_index].
        TensorId tid = TensorId.parse(inputName);
        runner.feed(tid.name, tid.outputIndex, t);
        feedNames.add(inputName);
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
            ++i;
        }
        throw new RuntimeException(
                "Node '" + outputName + "' was not provided to run(), so it cannot be read");
    }

    private void closeFeeds() {
        for (Tensor t : feedTensors) {
            t.close();
        }
        feedTensors.clear();
        feedNames.clear();
    }

    private void closeFetches() {
        for (Tensor t : fetchTensors) {
            t.close();
        }
        fetchTensors.clear();
        fetchNames.clear();
    }

    // State immutable between initializeTensorFlow calls.
    private final String modelName;
    private final Graph g;
    private final Session sess;

    // State reset on every call to run.
    private Session.Runner runner;
    private List<String> feedNames = new ArrayList<String>();
    private List<Tensor> feedTensors = new ArrayList<Tensor>();
    private List<String> fetchNames = new ArrayList<String>();
    private List<Tensor> fetchTensors = new ArrayList<Tensor>();

    // Mutable state.
    private RunStats runStats;
}
