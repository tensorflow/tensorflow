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

import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
/**
 * An in-process TensorFlow server, for use in distributed training.
 *
 * A {@code Server} instance encapsulates a set of devices and a
 * {@link org.tensorflow.Session} target that can participate in distributed
 * training. A server belongs to a cluster (specified by a
 * {@code ClusterSpec}), and corresponds to a particular task in a named job.
 * The server can communicate with any other server in the same cluster.
 *
 * <p><b>WARNING:</b> A {@code Server} owns resources that <b>must</b> be
 * explicitly freed by invoking {@link #close()}.
 *
 * <p>Instances of a {@code Server} are thread-safe.
 *
 * <p>Using example:
 * <pre>
 * {@code
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
 * }
 * </pre>
 */
public final class Server implements AutoCloseable {

  /** 
   * Constructs a new instance of server. 
   *
   * @param serverDef Server definition specified as a serialized
   *        <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/tensorflow_server.proto">ServerDef</a>
   *        protocol buffer.
   */
  public Server(byte[] serverDef) {
    nativeHandle = allocate(serverDef);
  }

  /** Starts this server. */
  public void start() {
    lock.readLock().lock();
    try {
      start(nativeHandle);
    }
    finally {
      lock.readLock().unlock();
    }
  }

  /** Stops this server. */
  public void stop() {
    lock.readLock().lock();
    try {
      stop(nativeHandle);
    }
    finally {
      lock.readLock().unlock();
    }
  }

  /** Blocks until the server has shut down (currently blocks forever). */
  public void join() {
    lock.readLock().lock();
    try {
      join(nativeHandle);
    }
    finally {
      lock.readLock().unlock();
    }
  }

  /** Stops server and frees resources. Server is expected to be stopped before. */
  @Override
  public void close() {
    lock.writeLock().lock();
    try {
      delete(nativeHandle);
      nativeHandle = 0;
    }
    finally {
      lock.writeLock().unlock();
    }
  }

  private static native long allocate(byte[] serverDef);

  private static native void start(long nativeHandle);

  private static native void stop(long nativeHandle);

  private static native void join(long nativeHandle);

  private static native void delete(long nativeHandle);

  private final ReadWriteLock lock = new ReentrantReadWriteLock();

  private long nativeHandle;

  static {
    TensorFlow.init();
  }
}