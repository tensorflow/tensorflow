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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.EagerSession.ResourceCleanupStrategy;

@RunWith(JUnit4.class)
public class EagerSessionTest {

  @Test
  public void closeSessionTwiceDoesNotFail() {
    try (EagerSession s = EagerSession.create()) {
      s.close();
    }
  }

  @Test
  public void cleanupResourceOnSessionClose() {
    TestReference ref;
    try (EagerSession s =
        EagerSession.options()
            .resourceCleanupStrategy(ResourceCleanupStrategy.ON_SESSION_CLOSE)
            .build()) {
      ref = new TestReference(s, new Object());
      assertFalse(ref.isDeleted());

      // check that reaching safe point did not release resources
      buildOp(s);
      assertFalse(ref.isDeleted());
    }
    assertTrue(ref.isDeleted());
  }

  @Test
  public void cleanupResourceOnSafePoints() {
    TestGarbageCollectorQueue gcQueue = new TestGarbageCollectorQueue();
    try (EagerSession s =
        EagerSession.options()
            .resourceCleanupStrategy(ResourceCleanupStrategy.ON_SAFE_POINTS)
            .buildForGcTest(gcQueue)) {

      TestReference ref = new TestReference(s, new Object());
      assertFalse(ref.isDeleted());

      // garbage collecting the reference won't release until we reached safe point
      gcQueue.collect(ref);
      assertFalse(ref.isDeleted());
      buildOp(s); // safe point
      assertTrue(ref.isDeleted());
      assertTrue(gcQueue.isEmpty());
    }
  }

  @Test
  public void cleanupResourceInBackground() {
    TestGarbageCollectorQueue gcQueue = new TestGarbageCollectorQueue();
    try (EagerSession s =
        EagerSession.options()
            .resourceCleanupStrategy(ResourceCleanupStrategy.IN_BACKGROUND)
            .buildForGcTest(gcQueue)) {

      TestReference ref = new TestReference(s, new Object());
      assertFalse(ref.isDeleted());

      gcQueue.collect(ref);
      sleep(50); // allow some time to the background thread for cleaning up resources
      assertTrue(ref.isDeleted());
      assertTrue(gcQueue.isEmpty());
    }
  }

  @Test
  public void clearedResourcesAreNotCleanedUp() {
    TestReference ref;
    try (EagerSession s = EagerSession.create()) {
      ref = new TestReference(s, new Object());
      ref.clear();
    }
    assertFalse(ref.isDeleted());
  }

  @Test
  public void buildingOpWithClosedSessionFails() {
    EagerSession s = EagerSession.create();
    s.close();
    try {
      buildOp(s);
      fail();
    } catch (IllegalStateException e) {
      // ok
    }
  }

  @Test
  public void addingReferenceToClosedSessionFails() {
    EagerSession s = EagerSession.create();
    s.close();
    try {
      new TestReference(s, new Object());
      fail();
    } catch (IllegalStateException e) {
      // ok
    }
  }

  @Test
  public void defaultSession() throws Exception {
    EagerSession.Options options =
        EagerSession.options().resourceCleanupStrategy(ResourceCleanupStrategy.ON_SESSION_CLOSE);
    EagerSession.initDefault(options);
    EagerSession session = EagerSession.getDefault();
    assertNotNull(session);
    assertEquals(ResourceCleanupStrategy.ON_SESSION_CLOSE, session.resourceCleanupStrategy());
    try {
      EagerSession.initDefault(options);
      fail();
    } catch (IllegalStateException e) {
      // expected
    }
    try {
      session.close();
      fail();
    } catch (IllegalStateException e) {
      // expected
    }
  }

  private static class TestReference extends EagerSession.NativeReference {

    TestReference(EagerSession session, Object referent) {
      super(session, referent);
    }

    @Override
    void delete() {
      if (!deleted.compareAndSet(false, true)) {
        fail("Reference was deleted more than once");
      }
    }

    boolean isDeleted() {
      return deleted.get();
    }

    private final AtomicBoolean deleted = new AtomicBoolean();
  }

  private static class TestGarbageCollectorQueue extends ReferenceQueue<Object> {

    @Override
    public Reference<? extends Object> poll() {
      return garbage.poll();
    }

    @Override
    public Reference<? extends Object> remove() throws InterruptedException {
      return garbage.take();
    }

    @Override
    public Reference<? extends Object> remove(long timeout)
        throws IllegalArgumentException, InterruptedException {
      return garbage.poll(timeout, TimeUnit.MILLISECONDS);
    }

    void collect(TestReference ref) {
      garbage.add(ref);
    }

    boolean isEmpty() {
      return garbage.isEmpty();
    }

    private final BlockingQueue<TestReference> garbage = new LinkedBlockingQueue<>();
  }

  private static void buildOp(EagerSession s) {
    // Creating an operation is a safe point for resource cleanup
    try {
      s.opBuilder("Const", "Const");
    } catch (UnsupportedOperationException e) {
      // TODO (karlllessard) remove this exception catch when EagerOperationBuilder is implemented
    }
  }

  private static void sleep(int millis) {
    try {
      Thread.sleep(millis);
    } catch (InterruptedException e) {
    }
  }
}
