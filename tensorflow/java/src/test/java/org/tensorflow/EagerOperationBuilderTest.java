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

import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link EagerOperationBuilder} class. */
@RunWith(JUnit4.class)
public class EagerOperationBuilderTest {

  @Test
  public void failToCreateIfSessionIsClosed() {
    EagerSession session = EagerSession.create();
    session.close();
    try {
      new EagerOperationBuilder(session, "Add", "add");
      fail();
    } catch (IllegalStateException e) {
      // expected
    }
  }

  @Test
  public void failToBuildOpIfSessionIsClosed() {
    EagerOperationBuilder opBuilder;
    try (EagerSession session = EagerSession.create()) {
      opBuilder = new EagerOperationBuilder(session, "Empty", "empty");
    }
    try {
      opBuilder.setAttr("dtype", DataType.FLOAT);
      fail();
    } catch (IllegalStateException e) {
      // expected
    }
  }

  @Test
  public void addInputs() {
    try (EagerSession session = EagerSession.create()) {
      Operation asrt =
          opBuilder(session, "Assert", "assert")
              .addInput(TestUtil.constant(session, "Cond", true))
              .addInputList(new Output<?>[] {TestUtil.constant(session, "Error", -1)})
              .build();
      try {
        opBuilder(session, "Const", "var").addControlInput(asrt);
        fail();
      } catch (UnsupportedOperationException e) {
        // expected
      }
    }
  }

  @Test
  public void setDevice() {
    try (EagerSession session = EagerSession.create()) {
      opBuilder(session, "Add", "SetDevice")
          .setDevice("/job:localhost/replica:0/task:0/device:CPU:0")
          .addInput(TestUtil.constant(session, "Const1", 2))
          .addInput(TestUtil.constant(session, "Const2", 4))
          .build();
    }
  }

  @Test
  public void setAttrs() {
    // The effect of setting an attribute may not easily be visible from the other parts of this
    // package's API. Thus, for now, the test simply executes the various setAttr variants to see
    // that there are no exceptions.
    //
    // This is a bit of an awkward test since it has to find operations with attributes of specific
    // types that aren't inferred from the input arguments.
    try (EagerSession session = EagerSession.create()) {
      // dtype, tensor attributes.
      try (Tensor<Integer> t = Tensors.create(1)) {
        opBuilder(session, "Const", "DataTypeAndTensor")
            .setAttr("dtype", DataType.INT32)
            .setAttr("value", t)
            .build();
      }
      // type, int (TF "int" attributes are 64-bit signed, so a Java long).
      opBuilder(session, "RandomUniform", "DataTypeAndInt")
          .addInput(TestUtil.constant(session, "RandomUniformShape", new int[] {1}))
          .setAttr("seed", 10)
          .setAttr("dtype", DataType.FLOAT)
          .build();
      // list(int), string
      opBuilder(session, "MaxPool", "IntListAndString")
          .addInput(TestUtil.constant(session, "MaxPoolInput", new float[2][2][2][2]))
          .setAttr("ksize", new long[] {1, 1, 1, 1})
          .setAttr("strides", new long[] {1, 1, 1, 1})
          .setAttr("padding", "SAME")
          .build();
      // list(float), device
      opBuilder(session, "FractionalMaxPool", "FloatList")
          .addInput(TestUtil.constant(session, "FractionalMaxPoolInput", new float[2][2][2][2]))
          .setAttr("pooling_ratio", new float[] {1.0f, 1.44f, 1.73f, 1.0f})
          .build();
      // shape
      opBuilder(session, "EnsureShape", "ShapeAttr")
          .addInput(TestUtil.constant(session, "Const", new int[2][2]))
          .setAttr("shape", Shape.make(2, 2))
          .build();
      // list(shape)
      opBuilder(session, "FIFOQueue", "queue")
          .setAttr("component_types", new DataType[] {DataType.INT32, DataType.INT32})
          .setAttr("shapes", new Shape[] {Shape.make(2, 2), Shape.make(2, 2, 2)})
          .build();
      // bool
      opBuilder(session, "All", "Bool")
          .addInput(TestUtil.constant(session, "Const", new boolean[] {true, true, false}))
          .addInput(TestUtil.constant(session, "Axis", 0))
          .setAttr("keep_dims", false)
          .build();
      // float
      opBuilder(session, "ApproximateEqual", "Float")
          .addInput(TestUtil.constant(session, "Const1", 10.00001f))
          .addInput(TestUtil.constant(session, "Const2", 10.00000f))
          .setAttr("tolerance", 0.1f)
          .build();
      // Missing tests: list(string), list(byte), list(bool), list(type)
    }
  }

  private static EagerOperationBuilder opBuilder(EagerSession session, String type, String name) {
    return new EagerOperationBuilder(session, type, name);
  }
}
