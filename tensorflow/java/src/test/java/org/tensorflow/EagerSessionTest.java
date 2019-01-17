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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

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
    AtomicBoolean deleted = new AtomicBoolean();

    try (EagerSession s = EagerSession.options()
        .resourceCleanupStrategy(ResourceCleanupStrategy.ON_SESSION_CLOSE)
        .build()) {

      new TestReference(s, new Object(), deleted);

      assertFalse(deleted.get());
      runGC();
      assertFalse(deleted.get());

      buildOp(s);
      assertFalse(deleted.get());  // reaching safe point did not release resources
    }
    assertTrue(deleted.get());
  }

  @Test
  public void cleanupResourceOnSafePoints() {
    AtomicBoolean deleted = new AtomicBoolean();

    try (EagerSession s = EagerSession.options()
        .resourceCleanupStrategy(ResourceCleanupStrategy.ON_SAFE_POINTS)
        .build()) {

      new TestReference(s, new Object(), deleted);

      assertFalse(deleted.get());
      runGC();
      assertFalse(deleted.get());

      buildOp(s);  
      assertTrue(deleted.get());  // reaching safe point released resources
    }
  }

  @Test
  public void cleanupResourceInBackground() {
    AtomicBoolean deleted = new AtomicBoolean();

    try (EagerSession s = EagerSession.options()
        .resourceCleanupStrategy(ResourceCleanupStrategy.IN_BACKGROUND)
        .build()) {

      new TestReference(s, new Object(), deleted);

      assertFalse(deleted.get());
      runGC();
      sleep(50);  // allow some time to the background thread for cleaning up resources
      assertTrue(deleted.get());
    }
  }

  @Test
  public void clearedResourcesAreNotCleanedUp() {
    AtomicBoolean deleted = new AtomicBoolean();

    try (EagerSession s = EagerSession.create()) {
      TestReference ref = new TestReference(s, new Object(), deleted);
      ref.clear();
    }
    assertFalse(deleted.get());
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
      new TestReference(s, new Object(), new AtomicBoolean());
      fail();
    } catch (IllegalStateException e) {
      // ok
    }
  }
 
  private static class TestReference extends EagerSession.NativeReference {
    
    TestReference(EagerSession session, Object referent, AtomicBoolean deleted) {
      super(session, referent);
      this.deleted = deleted;
    }
    
    @Override
    void delete() {
      if (!deleted.compareAndSet(false, true)) {
        fail("Reference was deleted more than once");
      }
    }
    
    private final AtomicBoolean deleted;
  }
  
  private static void buildOp(EagerSession s) {
    // Creating an operation is a safe point for resource cleanup
    try {
      s.opBuilder("Const", "Const");
    } catch (UnsupportedOperationException e) {
      // TODO (karlllessard) remove this exception catch when EagerOperationBuilder is implemented
    }
  }
  
  private static void runGC() {
    // Warning: There is no way to force the garbage collector to run, so here we simply to our best
    // to get it triggered but it might be sufficient on some platforms. Adjust accordingly if some 
    // cleanup tests start to fail.
    System.gc();
    System.runFinalization();
  }
  
  private static void sleep(int millis) {
    try {
      Thread.sleep(millis);
    } catch (InterruptedException e) {
    }
  }
}
