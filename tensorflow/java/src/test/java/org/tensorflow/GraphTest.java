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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

import java.nio.file.Files;
import java.nio.file.Paths;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.Graph}. */
@RunWith(JUnit4.class)
public class GraphTest {

  @Test
  public void graphDefRoundTrip() {
    try (Graph imported = new Graph()) {
      final byte[] inGraphDef =
          Files.readAllBytes(Paths.get("tensorflow/java/test_graph_def.data"));
      imported.importGraphDef(inGraphDef);
      validateImportedGraph(imported, "");

      final byte[] outGraphDef = imported.toGraphDef();
      try (Graph exported = new Graph()) {
        exported.importGraphDef(outGraphDef, "HeyHeyHey");
        validateImportedGraph(exported, "HeyHeyHey/");
      }
      // Knowing how test_graph_def.data was generated, it should have these nodes:
    } catch (Exception e) {
      fail("Unexpected exception: " + e);
    }
  }

  // Helper function whose implementation is based on knowledge of how test_graph_def.data was
  // produced.
  private void validateImportedGraph(Graph g, String prefix) {
    Operation op = g.operation(prefix + "MyConstant");
    assertNotNull(op);
    assertEquals(prefix + "MyConstant", op.name());
    assertEquals("Const", op.type());
    assertEquals(1, op.numOutputs());
    assertEquals(op, op.output(0).op());

    op = g.operation(prefix + "while/Less");
    assertNotNull(op);
    assertEquals(prefix + "while/Less", op.name());
    assertEquals("Less", op.type());
    assertEquals(1, op.numOutputs());
    assertEquals(op, op.output(0).op());
  }

  @Test
  public void failImportOnInvalidGraphDefs() {
    try (Graph g = new Graph()) {
      try {
        g.importGraphDef(null);
      } catch (IllegalArgumentException e) {
        // expected exception.
      }

      try {
        g.importGraphDef(new byte[] {1});
      } catch (IllegalArgumentException e) {
        // expected exception.
      }
    }
  }

  @Test
  public void failOnUseAfterClose() {
  }
}
