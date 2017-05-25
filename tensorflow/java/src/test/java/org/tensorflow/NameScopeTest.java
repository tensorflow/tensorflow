/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.NameScope}. */
@RunWith(JUnit4.class)
public class NameScopeTest {

  @Test
  public void basicNames() {
    NameScope root = NameScope.create();
    assertEquals("add", root.makeOpName("add"));
    assertEquals("add_1", root.makeOpName("add"));
    assertEquals("add_2", root.makeOpName("add"));
    assertEquals("mul", root.makeOpName("mul"));
  }

  @Test
  public void hierarchicalNames() {
    NameScope root = NameScope.create();
    NameScope child = root.withSubScope("child");
    assertEquals("child/add", child.makeOpName("add"));
    assertEquals("child/add_1", child.makeOpName("add"));
    assertEquals("child/mul", child.makeOpName("mul"));

    NameScope child_1 = root.withSubScope("child");
    assertEquals("child_1/add", child_1.makeOpName("add"));
    assertEquals("child_1/add_1", child_1.makeOpName("add"));
    assertEquals("child_1/mul", child_1.makeOpName("mul"));

    NameScope c_c = root.withSubScope("c").withSubScope("c");
    assertEquals("c/c/add", c_c.makeOpName("add"));

    NameScope c_1 = root.withSubScope("c");
    NameScope c_1_c = c_1.withSubScope("c");
    assertEquals("c_1/c/add", c_1_c.makeOpName("add"));

    NameScope c_1_c_1 = c_1.withSubScope("c");
    assertEquals("c_1/c_1/add", c_1_c_1.makeOpName("add"));
  }

  @Test
  public void scopeAndOpNames() {
    NameScope root = NameScope.create();
    NameScope child = root.withSubScope("child");

    assertEquals("child/add", child.makeOpName("add"));
    assertEquals("child_1", root.makeOpName("child"));
    assertEquals("child_2/p", root.withSubScope("child").makeOpName("p"));
  }

  @Test
  public void names() {
    NameScope root = NameScope.create();

    final String[] invalid_names = {
      "_", // Names are constrained to start with [A-Za-z0-9.]
      null, "", "a$", // Invalid characters
      "a/b", // slashes not allowed
    };

    for (String name : invalid_names) {
      try {
        root.withOpName(name);
        fail("failed to catch invalid op name.");
      } catch (IllegalArgumentException ex) {
        // expected
      }
      // Root scopes follow the same rules as opnames
      try {
        root.withSubScope(name);
        fail("failed to catch invalid scope name: " + name);
      } catch (IllegalArgumentException ex) {
        // expected
      }
    }

    // Non-root scopes have a less restrictive constraint.
    assertEquals("a/_/hello", root.withSubScope("a").withSubScope("_").makeOpName("hello"));
  }

  // A dummy composite op - it should create a tree of op names.
  private static void topCompositeOp(NameScope scope, List<String> opnames) {
    NameScope compScope = scope.withSubScope("top");
    opnames.add(compScope.makeOpName("mul"));
    opnames.add(compScope.makeOpName("bias_add"));
    intermediateOp(compScope, opnames);
  }

  private static void intermediateOp(NameScope scope, List<String> opnames) {
    NameScope compScope = scope.withSubScope("intermediate");
    opnames.add(compScope.makeOpName("c1"));
    opnames.add(compScope.makeOpName("mul"));
    leafOp(compScope.withOpName("c2"), opnames);
  }

  private static void leafOp(NameScope scope, List<String> opnames) {
    opnames.add(scope.makeOpName("const"));
  }

  @Test
  public void compositeOp() {
    NameScope root = NameScope.create();
    List<String> names = new ArrayList<String>();
    topCompositeOp(root, names);
    assertEquals(
        Arrays.asList(
            "top/mul",
            "top/bias_add",
            "top/intermediate/c1",
            "top/intermediate/mul",
            "top/intermediate/c2"),
        names);

    assertEquals("top_1", root.makeOpName("top"));

    names.clear();
    topCompositeOp(root, names);
    assertEquals(
        Arrays.asList(
            "top_2/mul",
            "top_2/bias_add",
            "top_2/intermediate/c1",
            "top_2/intermediate/mul",
            "top_2/intermediate/c2"),
        names);

    names.clear();
    topCompositeOp(root.withOpName("c"), names);
    assertEquals(
        Arrays.asList(
            "c/mul", "c/bias_add", "c/intermediate/c1", "c/intermediate/mul", "c/intermediate/c2"),
        names);
  }
}
