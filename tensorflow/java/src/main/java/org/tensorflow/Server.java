/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
 * An in-process TensorFlow server, for use in distributed training.
 *
 * <p>A {@code Server} instance encapsulates a set of devices and a {@link org.tensorflow.Session}
 * target that can participate in distributed training. A server belongs to a cluster (specified by
 * a {@code ClusterSpec}), and corresponds to a particular task in a named job. The server can
 * communicate with any other server in the same cluster. The server will not serve any requests
 * until {@link #start()} is invoked. The server will stop serving requests once {@link #stop()} or
 * {@link #close()} is invoked. Be aware that {@link #close()} method stops the server if it is
 * running.
 *
 * <p><b>WARNING:</b> A {@code Server} owns resources that <b>must</b> be explicitly freed by
 * invoking {@link #close()}.
 *
 * <p>Instances of a {@code Server} are thread-safe.
 *
 * <p>Using example:
 *
 * <pre>{@code
 * import org.tensorflow.Server;
 * import org.tensorflow.distruntime.ClusterDef;
 * import org.tensorflow.distruntime.JobDef;
 * import org.tensorflow.distruntime.ServerDef;
 *
 * ClusterDef clusterDef = ClusterDef.newBuilder()
 *   .addJob(JobDef.newBuilder()
 *   .setName("worker")
 *   .putTasks(0, "localhost:4321")
 *   .build()
 * ).build();
 *
 * ServerDef serverDef = ServerDef.newBuilder()
 *   .setCluster(clusterDef)
 *   .setJobName("worker")
 *   .setTaskIndex(0)
 *   .setProtocol("grpc")
 * .build();
 *
 * try (Server srv = new Server(serverDef.toByteArray())) {
 *   srv.start();
 *   srv.join();
 * }
 * }</pre>
 */
public final class Server implements AutoCloseable {
  /**
   * Constructs a new instance of server.
   *
   * @param serverDef Server definition specified as a serialized <a
   *     href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/tensorflow_server.proto">ServerDef</a>
   *     protocol buffer.
   */
  public Server(byte[] serverDef) {
    nativeHandle = allocate(serverDef);
  }

  /** Starts an in-process TensorFlow server. */
  public synchronized void start() {
    start(nativeHandle);
  }

  /** Stops an in-process TensorFlow server. */
  public synchronized void stop() {
    stop(nativeHandle);
  }

  /** Blocks until the server has been successfully stopped. */
  public void join() {
    long handle = 0;
    synchronized (this) {
      handle = nativeHandle;
      if (handle != 0) {
        numJoining++;
      }
    }
    try {
      join(handle);
    } finally {
      synchronized (this) {
        if (handle != 0) {
          numJoining--;
        }
        notifyAll();
      }
    }
  }

  /** Destroy an in-process TensorFlow server, frees memory. */
  @Override
  public synchronized void close() throws InterruptedException {
    stop();
    while (numJoining > 0) {
      wait();
    }
    delete(nativeHandle);
    nativeHandle = 0;
  }

  private static native long allocate(byte[] serverDef);

  private static native void start(long nativeHandle);

  private static native void stop(long nativeHandle);

  private static native void join(long nativeHandle);

  private static native void delete(long nativeHandle);

  private long nativeHandle;

  private int numJoining;

  static {
    TensorFlow.init();
  }
}
