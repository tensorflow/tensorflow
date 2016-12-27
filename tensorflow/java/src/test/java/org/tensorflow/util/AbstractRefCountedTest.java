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

package org.tensorflow.util;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static org.junit.Assert.*;

/** Unit tests for {@link org.tensorflow.util.AbstractRefCounted}. */
@RunWith(JUnit4.class)
public class AbstractRefCountedTest {

  static class TestRefCounted extends AbstractRefCounted {
    public boolean deallocated = false;

    @Override
    protected void deallocate() { assertFalse(deallocated); deallocated = true; }
  }

  @Test
  public void initialState() {
    TestRefCounted t = new TestRefCounted();
    assertEquals(1, t.refCount());
  }

  @Test
  public void ref() {
    TestRefCounted t = new TestRefCounted();
    assertSame(t, t.ref());
    assertEquals(2, t.refCount());
    assertFalse(t.unref());
    assertEquals(1, t.refCount());
    assertTrue(t.unref());
    assertEquals(0, t.refCount());
  }

  @Test
  public void deallocated() {
    TestRefCounted t = new TestRefCounted();
    t.ref();
    t.unref();
    assertFalse(t.deallocated);
    t.unref();
    assertTrue(t.deallocated);
  }

  @Test
  public void failUnref() {
    TestRefCounted t = new TestRefCounted();
    t.unref();
    try {
      assertEquals(0, t.refCount());
      t.unref();
      fail("AbstractRefCounted.unref should fail when the refcount is already zero.");
    }
    catch(IllegalStateException e) {
      // The expected exception.
    }
    assertEquals(0, t.refCount());
  }

  @Test
  public void failRef() {
    TestRefCounted t = new TestRefCounted();
    t.unref();
    try {
      assertEquals(0, t.refCount());
      t.ref();
      fail("AbstractRefCounted.ref should fail when the refcount is already zero.");
    }
    catch(IllegalStateException e) {
      // The expected exception.
    }
    assertEquals(0, t.refCount());
  }
}
