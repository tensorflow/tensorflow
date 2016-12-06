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

import java.nio.file.Files;
import java.nio.file.Paths;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.Graph}. */
@RunWith(JUnit4.class)
public class GraphTest {

  @Test
  public void graphDefImportAndExport() {
    try (Graph g = new Graph()) {
      final byte[] inGraphDef = Files.readAllBytes(Paths.get("tensorflow/java/test_graph_def.data"));
      g.importGraphDef(inGraphDef);
      final byte[] outGraphDef = g.toGraphDef();
      // The graphs may not be identical as the proto format allows the same message
      // to be encoded in multiple ways. Once the Graph API is expressive enough
      // to construct graphs and query for nodes/operations, use that.
      // Till then a very crude test:
      assertEquals(inGraphDef.length, outGraphDef.length);
    } catch (Exception e) {
      fail("Unexpected exception: " + e);
    }
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
}
