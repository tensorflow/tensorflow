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

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collection;

/** Static utility functions. */
public class TestUtil {

  public static final class AutoCloseableList<E extends AutoCloseable> extends ArrayList<E>
      implements AutoCloseable {
    public AutoCloseableList(Collection<? extends E> c) {
      super(c);
    }

    @Override
    public void close() {
      Exception toThrow = null;
      for (AutoCloseable c : this) {
        try {
          c.close();
        } catch (Exception e) {
          toThrow = e;
        }
      }
      if (toThrow != null) {
        throw new RuntimeException(toThrow);
      }
    }
  }

  public static GraphOperation constantOp(Graph g, String name, Object value) {
    try (Tensor<?> t = Tensor.create(value)) {
      return g.opBuilder("Const", name).setAttr("dtype", t.dataType()).setAttr("value", t).build();
    }
  }

  public static <T> Output<T> constant(Graph g, String name, Object value) {
    return constantOp(g, name, value).<T>output(0);
  }

  public static <T> Output<T> placeholder(Graph g, String name, Class<T> type) {
    return g.opBuilder("Placeholder", name)
        .setAttr("dtype", DataType.fromClass(type))
        .build()
        .<T>output(0);
  }

  public static <T> Output<T> addN(Graph g, Output<?>... inputs) {
    return g.opBuilder("AddN", "AddN").addInputList(inputs).build().output(0);
  }

  public static <T> Output<T> matmul(
      Graph g, String name, Output<T> a, Output<T> b, boolean transposeA, boolean transposeB) {
    return g.opBuilder("MatMul", name)
        .addInput(a)
        .addInput(b)
        .setAttr("transpose_a", transposeA)
        .setAttr("transpose_b", transposeB)
        .build()
        .<T>output(0);
  }

  public static Operation split(Graph g, String name, int[] values, int numSplit) {
    return g.opBuilder("Split", name)
        .addInput(constant(g, "split_dim", 0))
        .addInput(constant(g, "values", values))
        .setAttr("num_split", numSplit)
        .build();
  }
  
  public static <T> Output<T> square(Graph g, String name, Output<T> value) {
    return g.opBuilder("Square", name)
        .addInput(value)
        .build()
        .<T>output(0);
  }

  public static void transpose_A_times_X(Graph g, int[][] a) {
    Output<Integer> aa = constant(g, "A", a);
    matmul(g, "Y", aa, placeholder(g, "X", Integer.class), true, false);
  }

  /**
   * Counts the total number of elements in an ND array.
   *
   * @param array the array to count the elements of
   * @return the number of elements
   */
  public static int flattenedNumElements(Object array) {
    int count = 0;
    for (int i = 0; i < Array.getLength(array); i++) {
      Object e = Array.get(array, i);
      if (!e.getClass().isArray()) {
        count += 1;
      } else {
        count += flattenedNumElements(e);
      }
    }
    return count;
  }

  /**
   * Flattens an ND-array into a 1D-array with the same elements.
   *
   * @param array the array to flatten
   * @param elementType the element class (e.g. {@code Integer.TYPE} for an {@code int[]})
   * @return a flattened array
   */
  public static Object flatten(Object array, Class<?> elementType) {
    Object out = Array.newInstance(elementType, flattenedNumElements(array));
    flatten(array, out, 0);
    return out;
  }

  private static int flatten(Object array, Object out, int next) {
    for (int i = 0; i < Array.getLength(array); i++) {
      Object e = Array.get(array, i);
      if (!e.getClass().isArray()) {
        Array.set(out, next++, e);
      } else {
        next = flatten(e, out, next);
      }
    }
    return next;
  }

  /**
   * Converts a {@code boolean[]} to a {@code byte[]}.
   *
   * <p>Suitable for creating tensors of type {@link DataType#BOOL} using {@link
   * java.nio.ByteBuffer}.
   */
  public static byte[] bool2byte(boolean[] array) {
    byte[] out = new byte[array.length];
    for (int i = 0; i < array.length; i++) {
      out[i] = array[i] ? (byte) 1 : (byte) 0;
    }
    return out;
  }

  /**
   * Converts a {@code byte[]} to a {@code boolean[]}.
   *
   * <p>Suitable for reading tensors of type {@link DataType#BOOL} using {@link
   * java.nio.ByteBuffer}.
   */
  public static boolean[] byte2bool(byte[] array) {
    boolean[] out = new boolean[array.length];
    for (int i = 0; i < array.length; i++) {
      out[i] = array[i] != 0;
    }
    return out;
  }

  private TestUtil() {}
}
