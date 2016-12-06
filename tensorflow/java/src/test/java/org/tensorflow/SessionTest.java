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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.Session}. */
@RunWith(JUnit4.class)
public class SessionTest {

  @Test
  public void run() {
    try (Graph g = new Graph();
        Session s = new Session(g);
        Tensor input = Tensor.create(new long[] {1, 2, 3, 4, 5})) {
      try {
        g.importGraphDef(
            Files.readAllBytes(Paths.get("tensorflow/java/test_graph_def.data")));
        AutoCloseableList<Tensor> outputs =
            new AutoCloseableList<Tensor>(
                s.runner().feed("input", 0, input).fetch("output", 0).run());
        assertEquals(1, outputs.size());
        assertEquals(15, outputs.get(0).longValue());
      } catch (IOException e) {
        fail("Graph import failed: " + e);
      }
    }
  }

  @Test
  public void failOnUseAfterClose() {
    try (Graph g = new Graph()) {
      Session s = new Session(g);
      s.close();
      try {
        s.runner().run();
        fail("methods on a close()d session should fail");
      } catch (IllegalStateException e) {
        // expected exception
      }
    }
  }

  private static final class AutoCloseableList<E extends AutoCloseable> extends ArrayList<E>
      implements AutoCloseable {
    AutoCloseableList(Collection<? extends E> c) {
      super(c);
    }

    @Override
    public void close() throws Exception {
      Exception toThrow = null;
      for (AutoCloseable c : this) {
        try {
          c.close();
        } catch (Exception e) {
          toThrow = e;
        }
      }
      if (toThrow != null) {
        throw toThrow;
      }
    }
  }
}
