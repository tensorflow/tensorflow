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
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link EagerOperation} class. */
@RunWith(JUnit4.class)
public class EagerOperationTest {

  @Test
  public void failToCreateIfSessionIsClosed() {
    EagerSession session = EagerSession.create();
    session.close();
    try {
      new EagerOperation(session, 1L, new long[] {1L}, "Add", "add");
      fail();
    } catch (IllegalStateException e) {
      // expected
    }
  }

  @Test
  public void outputDataTypeAndShape() {
    try (EagerSession session = EagerSession.create();
        Tensor<Integer> t = Tensors.create(new int[2][3])) {
      EagerOperation op =
          opBuilder(session, "Const", "OutputAttrs")
              .setAttr("dtype", DataType.INT32)
              .setAttr("value", t)
              .build();
      assertEquals(DataType.INT32, op.dtype(0));
      assertEquals(2, op.shape(0)[0]);
      assertEquals(3, op.shape(0)[1]);
    }
  }

  @Test
  public void outputTensor() {
    try (EagerSession session = EagerSession.create()) {
      EagerOperation add =
          opBuilder(session, "Add", "CompareResult")
              .addInput(TestUtil.constant(session, "Const1", 2))
              .addInput(TestUtil.constant(session, "Const2", 4))
              .build();
      assertEquals(6, add.tensor(0).intValue());

      // Validate that we retrieve the right shape and datatype from the tensor
      // that has been resolved
      assertEquals(0, add.shape(0).length);
      assertEquals(DataType.INT32, add.dtype(0));
    }
  }

  @Test
  public void inputAndOutputListLengths() {
    try (EagerSession session = EagerSession.create()) {
      Output<Float> c1 = TestUtil.constant(session, "Const1", new float[] {1f, 2f});
      Output<Float> c2 = TestUtil.constant(session, "Const2", new float[] {3f, 4f});

      EagerOperation acc =
          opBuilder(session, "AddN", "InputListLength")
              .addInputList(new Output<?>[] {c1, c2})
              .build();
      assertEquals(2, acc.inputListLength("inputs"));
      assertEquals(1, acc.outputListLength("sum"));

      EagerOperation split =
          opBuilder(session, "Split", "OutputListLength")
              .addInput(TestUtil.constant(session, "Axis", 0))
              .addInput(c1)
              .setAttr("num_split", 2)
              .build();
      assertEquals(1, split.inputListLength("split_dim"));
      assertEquals(2, split.outputListLength("output"));

      try {
        split.inputListLength("no_such_input");
        fail();
      } catch (IllegalArgumentException e) {
        // expected
      }

      try {
        split.outputListLength("no_such_output");
        fail();
      } catch (IllegalArgumentException e) {
        // expected
      }
    }
  }

  @Test
  public void numOutputs() {
    try (EagerSession session = EagerSession.create()) {
      EagerOperation op =
          opBuilder(session, "UniqueWithCountsV2", "unq")
              .addInput(TestUtil.constant(session, "Const1", new int[] {1, 2, 1}))
              .addInput(TestUtil.constant(session, "Axis", new int[] {0}))
              .setAttr("out_idx", DataType.INT32)
              .build();
      assertEquals(3, op.numOutputs());
    }
  }

  @Test
  public void opNotAccessibleIfSessionIsClosed() {
    EagerSession session = EagerSession.create();
    EagerOperation add =
        opBuilder(session, "Add", "SessionClosed")
            .addInput(TestUtil.constant(session, "Const1", 2))
            .addInput(TestUtil.constant(session, "Const2", 4))
            .build();
    assertEquals(1, add.outputListLength("z"));
    session.close();
    try {
      add.outputListLength("z");
      fail();
    } catch (IllegalStateException e) {
      // expected
    }
  }

  @Test
  public void outputIndexOutOfBounds() {
    try (EagerSession session = EagerSession.create()) {
      EagerOperation add =
          opBuilder(session, "Add", "OutOfRange")
              .addInput(TestUtil.constant(session, "Const1", 2))
              .addInput(TestUtil.constant(session, "Const2", 4))
              .build();
      try {
        add.getUnsafeNativeHandle(1);
        fail();
      } catch (IndexOutOfBoundsException e) {
        // expected
      }
      try {
        add.shape(1);
        fail();
      } catch (IndexOutOfBoundsException e) {
        // expected
      }
      try {
        add.dtype(1);
        fail();
      } catch (IndexOutOfBoundsException e) {
        // expected
      }
      try {
        add.tensor(1);
        fail();
      } catch (IndexOutOfBoundsException e) {
        // expected
      }
    }
  }

  private static EagerOperationBuilder opBuilder(EagerSession session, String type, String name) {
    return new EagerOperationBuilder(session, type, name);
  }
}
