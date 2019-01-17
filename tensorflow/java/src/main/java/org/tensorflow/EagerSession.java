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

import java.lang.ref.PhantomReference;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * An environment for executing TensorFlow operations eagerly.
 * <p>
 * Eager execution is an imperative programming environment that evaluates operations immediately, 
 * without building graphs. Operations return concrete values instead of constructing a computational 
 * graph to run later, as with {@link Graph}s and {@link Session}s.
 * <p> 
 * This makes it easy to develop with TensorFlow and debug models, as it behaves more like a standard
 * programming library.
 * <p>
 * Instances of a {@code EagerSession} are thread-safe.
 * <p>
 * <b>WARNING:</b> Resources consumed by an {@code EagerSession} object must be explicitly freed by invoking
 * the {@link #close()} method when it is no longer needed. This could be achieve using the `try-with-resources`
 * technique as the example below:
 * <pre>{@code
 * try (EagerSession s = EagerSession.create()) {
 *    // execute operations eagerly
 * }
 * }</pre>
 * In addition, {@code EagerSession} objects clean up unused resources during the session, working in pair 
 * with the JVM garbage collector. See {@link ResourceCleanupStrategy} for more details.
 */
public final class EagerSession implements ExecutionEnvironment, AutoCloseable {

  /**
   * Controls how to act when we try to run an operation on a given device but
   * some input tensors are not on that device.
   */
  public static enum DevicePlacementPolicy {

    /** 
     * Running operations with input tensors on the wrong device will fail. 
     */
    EXPLICIT(0),

    /** 
     * Copy the tensor to the right device but log a warning. 
     */
    WARN(1),

    /** 
     * Silently copy the tensor, which has a performance cost since the 
     * operation will be blocked till the copy completes. This is the default 
     * placement policy.
     */
    SILENT(2),

    /** 
     * Placement policy which silently copies int32 tensors but not other 
     * dtypes.
     */
    SILENT_FOR_INT32(3);
    
    private DevicePlacementPolicy(int code) {
      this.code = code;
    }
    
    private final int code;
  }
  
  /**
   * Controls how TensorFlow resources are cleaned up when they are no longer needed.
   * <p>
   * All resources allocated during an {@code EagerSession} are deleted when the session is closed. 
   * To prevent out-of-memory errors, it is also strongly suggest to cleanup those resources during 
   * the session. For example, executing n operations in a loop of m iterations will allocate a 
   * minimum of n*m resources while in most cases, only resources of the last iteration are still 
   * being used.
   * <p>
   * {@code EagerSession} instances can be notified in different ways when TensorFlow objects are no 
   * longer being referred, so they can proceed to the cleanup of any resources they owned.
   */
  public static enum ResourceCleanupStrategy {
    
    /**
     * Monitor and delete unused resources from a new thread running in background.
     * <p>
     * This is the most reliable approach to cleanup TensorFlow resources, at the cost of starting 
     * and running an additional thread dedicated to this task. Each {@code EagerSession} instance 
     * has its own thread, which is stopped only when the session is closed.
     * <p>
     * This strategy is used by default.
     */
    IN_BACKGROUND,
    
    /**
     * Monitor and delete unused resources from existing threads, before or after they complete 
     * another task.
     * <p>
     * Unused resources are released when a call to the TensorFlow library reaches a safe point for 
     * cleanup. This is done synchronously and might block for a short period of time the thread who 
     * triggered that call.
     * <p> 
     * This strategy should be used only if, for some reasons, no additional thread should be allocated 
     * for cleanup. Otherwise, {@link #IN_BACKGROUND} should be preferred.
     */
    ON_SAFE_POINTS,
    
    /**
     * Only delete resources when the session is closed.
     * <p>
     * All resources allocated during the session will remained in memory until the session is 
     * explicitly closed (or via the traditional `try-with-resource` technique). No extra task for 
     * resource cleanup will be attempted.
     * <p>
     * This strategy can lead up to out-of-memory errors and its usage is not recommended, unless the 
     * scope of the session is limited to execute only a small amount of operations.
     */
    ON_SESSION_CLOSE,
  }
  
  public static class Options {
    
    /**
     * Controls how operations dispatched are actually executed. 
     * <p>
     * When set to true, each operation are executed asynchronously (in which case some operations
     * might return "non-ready" outputs). When set to false, all operations are executed synchronously.
     * <p>
     * Synchronous execution is used by default.
     * 
     * @param value true for asynchronous execution, false for synchronous.
     */
    public Options async(boolean value) {
      async = value;
      return this;
    }
    
    /**
     * Controls how to act when we try to run an operation on a given device but
     * some input tensors are not on that device.
     * <p>
     * {@link DevicePlacementPolicy#SILENT} is used by default.
     * 
     * @param value policy to apply
     * @see {@link DevicePlacementPolicy}
     */
    public Options devicePlacementPolicy(DevicePlacementPolicy value) {
      devicePlacementPolicy = value;
      return this;
    }
    
    /**
     * Controls how TensorFlow resources are cleaned up when no longer needed.
     * <p>
     * {@link ResourceCleanupStrategy#IN_BACKGROUND} is used by default.
     * 
     * @param value strategy to use
     * @see {@link ResourceCleanupStrategy}
     */
    public Options resourceCleanupStrategy(ResourceCleanupStrategy value) {
      resourceCleanupStrategy = value;
      return this;
    }
    
    /**
     * Configures the session based on the data found in the provided buffer, which is serialized TensorFlow 
     * config proto.
     * <p>
     * Warning: the support of this feature is subject to changes since TensorFlow protos might not be supported
     * on public endpoints in the future. 
     * 
     * @param value a serialized config proto
     * @see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
     */
    public Options config(byte[] value) {
      config = value;
      return this;
    }
    
    /** 
     * Builds an eager session with the selected options.
     */
    public EagerSession build() {
      return new EagerSession(this);
    }
    
    private boolean async;
    private DevicePlacementPolicy devicePlacementPolicy;
    private ResourceCleanupStrategy resourceCleanupStrategy;
    private byte[] config;
    
    private Options() {
      async = false;
      devicePlacementPolicy = DevicePlacementPolicy.SILENT;
      resourceCleanupStrategy = ResourceCleanupStrategy.IN_BACKGROUND;
      config = null;
    }
  }
  
  /**
   * Returns an object that configures and builds a {@code EagerSession} with custom options.
   */
  public static EagerSession.Options options() {
    return new Options();
  }
  
  /**
   * Returns an {@code EagerSession} configured with default options.
   */
  public static EagerSession create() {
    return options().build();
  }

  private EagerSession(Options options) {
    this.nativeHandle = allocate(options.async, options.devicePlacementPolicy.code, options.config);
    this.resourceCleanupStrategy = options.resourceCleanupStrategy;

    if (resourceCleanupStrategy == ResourceCleanupStrategy.IN_BACKGROUND) {
      nativeResources.startCleanupThread();
    }
  }

  @Override
  public synchronized void close() {
    if (nativeHandle != 0L) {
      if (resourceCleanupStrategy == ResourceCleanupStrategy.IN_BACKGROUND) {
        nativeResources.stopCleanupThread();
      }
      nativeResources.deleteAll();
      delete(nativeHandle);
      nativeHandle = 0L;
    }
  }

  @Override
  public OperationBuilder opBuilder(String type, String name) {
    if (resourceCleanupStrategy == ResourceCleanupStrategy.ON_SAFE_POINTS) {
      nativeResources.tryCleanup();
    }
    checkSession();
    // TODO (karllessard) create a new EagerOperationBuilder
    throw new UnsupportedOperationException("Eager execution mode is not yet supported in this version of TensorFlow");
  }
  
  /**
   * A reference to one or more allocated native resources.
   * <p>
   * Any Java objects owning native resources must declare a reference to those resources in a 
   * subclass that extends from {@code NativeReference}. When {@link NativeReference#delete()} is invoked, 
   * the resources must be freed. For example:
   * <pre>{@code
   * private static class NativeReference extends EagerSession.NativeReference {
   * 
   *    NativeReference(EagerSession session, MyClass referent, long handle) {
   *        super(session, referent);
   *        this.handle = handle;
   *    }
   *    
   *    @Override
   *    void delete() {
   *        MyClass.nativeDelete(handle);
   *    }
   *    
   *    private final long handle;
   * }
   * }</pre>
   * 
   * A Java object "owns" a native resource if this resource should not survive beyond the lifetime of
   * this object.
   * <p> 
   * <b>IMPORTANT</b>: All nested subclasses of {@code NativeReference} must be declared as static, otherwise 
   * their instances will hold an implicit reference to their enclosing object, preventing the garbage
   * collector to release them when they are no longer needed.
   */
  static abstract class NativeReference extends PhantomReference<Object> {

    /**
     * Attach a new phantom reference of {@code referent} to {@code session}.
     */
    public NativeReference(EagerSession session, Object referent) {
      super(referent, session.nativeResources.garbageQueue);
      session.checkSession();
      nativeResources = session.nativeResources;
      nativeResources.attach(this);
    }
    
    /**
     * Detach this reference from its current session.
     * <p>
     * Clearing a NativeReference does not invoke {@link #delete()}, thus won't release the native resources it 
     * refers to. It can be used when passing the ownership of those resources to another object.
     * <p>
     * If native resources needs to be deleted as well, call {@link #delete()} explicitly.
     */
    @Override
    public void clear() {
      nativeResources.detach(this);
      super.clear();
    }
    
    /**
     * Releases all native resources owned by the referred object, now deleted.  
     */
    abstract void delete();
    
    private final NativeResourceCollector nativeResources;
  }
  
  /**
   * Collects native references attached to this session and releases their resources if they are
   * no longer needed.
   */
  private static class NativeResourceCollector {
    
    void attach(NativeReference nativeRef) {
      synchronized(nativeRefs) {
        nativeRefs.put(nativeRef, null);
      }
    }
    
    void detach(NativeReference nativeRef) {
      synchronized(nativeRefs) {
        nativeRefs.remove(nativeRef);
      }
    }
    
    void delete(NativeReference nativeRef) {
      synchronized(nativeRefs) {
        if (!nativeRefs.keySet().remove(nativeRef)) {
          return;  // safety check
        }
      }
      nativeRef.delete();
    }
    
    void deleteAll() {
      synchronized(nativeRefs) {
        for (NativeReference nativeRef : nativeRefs.keySet()) {
          nativeRef.delete();
        }
        nativeRefs.clear();
      }
    }

    void tryCleanup() {
      Reference<?> nativeRef;
      synchronized(nativeRefs) {
        while ((nativeRef = garbageQueue.poll()) != null) {
          delete((NativeReference)nativeRef);
        }
      }
    }
    
    synchronized void startCleanupThread() {
      if (cleanupInBackground) {
        return;  // ignore if cleanup thread is already running
      }
      try {
        cleanupInBackground = true;
        cleanupService.execute(new Runnable() {
          @Override
          public void run() {
            try {
              while(cleanupInBackground) {
                NativeReference nativeRef = (NativeReference)garbageQueue.remove();
                delete(nativeRef);
              }
            } catch (InterruptedException e) {
              // exit
            }
          }
        });
      } catch (Exception e) {
        cleanupInBackground = false;
        throw e;
      }
    }
    
    void stopCleanupThread() {
      cleanupInBackground = false;
      cleanupService.shutdownNow();  // returns without waiting for the thread to stop
    }

    private final ExecutorService cleanupService = Executors.newSingleThreadExecutor();
    private final Map<NativeReference, Void> nativeRefs = new IdentityHashMap<>();
    private final ReferenceQueue<Object> garbageQueue = new ReferenceQueue<>();
    private volatile boolean cleanupInBackground = false;
  }
  
  private final NativeResourceCollector nativeResources = new NativeResourceCollector();
  private final ResourceCleanupStrategy resourceCleanupStrategy;
  private long nativeHandle;
  
  private void checkSession() {
    if (nativeHandle == 0L) {
      throw new IllegalStateException("Eager session has been closed");
    }
  }

  private static native long allocate(boolean async, int devicePlacementPolicy, byte[] config);

  private static native void delete(long handle);
  
  private static native long allocateOperation(long contextHandle, String name);

  static {
    TensorFlow.init();
  }
}
