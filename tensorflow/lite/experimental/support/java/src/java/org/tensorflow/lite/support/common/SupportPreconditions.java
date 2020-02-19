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

package org.tensorflow.lite.support.common;

import org.checkerframework.checker.nullness.qual.Nullable;

/** Static error checking util methods. */
public final class SupportPreconditions {
  /**
   * Ensures that an object reference passed as a parameter to the calling method is not null.
   *
   * @param reference an object reference
   * @return the non-null reference that was validated
   * @throws NullPointerException if {@code reference} is null
   */
  public static <T extends Object> T checkNotNull(T reference) {
    if (reference == null) {
      throw new NullPointerException("The object reference is null.");
    }
    return reference;
  }

  /**
   * Ensures that an object reference passed as a parameter to the calling method is not null.
   *
   * @param reference an object reference
   * @param errorMessage the exception message to use if the check fails; will be converted to a
   *     string using {@link String#valueOf(Object)}
   * @return the non-null reference that was validated
   * @throws NullPointerException if {@code reference} is null
   */
  public static <T extends Object> T checkNotNull(T reference, @Nullable Object errorMessage) {
    if (reference == null) {
      throw new NullPointerException(String.valueOf(errorMessage));
    }
    return reference;
  }

  /**
   * Ensures that the given String is not empty and not null.
   *
   * @param string the String to test
   * @return the non-null non-empty String that was validated
   * @throws IllegalArgumentException if {@code string} is null or empty
   */
  public static String checkNotEmpty(String string) {
    if (string == null || string.length() == 0) {
      throw new IllegalArgumentException("Given String is empty or null.");
    }
    return string;
  }

  /**
   * Ensures that the given String is not empty and not null.
   *
   * @param string the String to test
   * @param errorMessage the exception message to use if the check fails; will be converted to a
   *     string using {@link String#valueOf(Object)}
   * @return the non-null non-empty String that was validated
   * @throws IllegalArgumentException if {@code string} is null or empty
   */
  public static String checkNotEmpty(String string, Object errorMessage) {
    if (string == null || string.length() == 0) {
      throw new IllegalArgumentException(String.valueOf(errorMessage));
    }
    return string;
  }

  /**
   * Ensures the truth of an expression involving one or more parameters to the calling method.
   *
   * @param expression a boolean expression.
   * @throws IllegalArgumentException if {@code expression} is false.
   */
  public static void checkArgument(boolean expression) {
    if (!expression) {
      throw new IllegalArgumentException();
    }
  }

  /**
   * Ensures the truth of an expression involving one or more parameters to the calling method.
   *
   * @param expression a boolean expression.
   * @param errorMessage the exception message to use if the check fails; will be converted to a
   *     string using {@link String#valueOf(Object)}.
   * @throws IllegalArgumentException if {@code expression} is false.
   */
  public static void checkArgument(boolean expression, @Nullable Object errorMessage) {
    if (!expression) {
      throw new IllegalArgumentException(String.valueOf(errorMessage));
    }
  }

  /**
   * Ensures that {@code index} specifies a valid <i>element</i> in an array, list or string of size
   * {@code size}. An element index may range from zero, inclusive, to {@code size}, exclusive.
   *
   * @param index a user-supplied index identifying an element of an array, list or string
   * @param size the size of that array, list or string
   * @return the value of {@code index}
   * @throws IndexOutOfBoundsException if {@code index} is negative or is not less than {@code size}
   * @throws IllegalArgumentException if {@code size} is negative
   */
  public static int checkElementIndex(int index, int size) {
    return checkElementIndex(index, size, "index");
  }

  /**
   * Ensures that {@code index} specifies a valid <i>element</i> in an array, list or string of size
   * {@code size}. An element index may range from zero, inclusive, to {@code size}, exclusive.
   *
   * @param index a user-supplied index identifying an element of an array, list or string
   * @param size the size of that array, list or string
   * @param desc the text to use to describe this index in an error message
   * @return the value of {@code index}
   * @throws IndexOutOfBoundsException if {@code index} is negative or is not less than {@code size}
   * @throws IllegalArgumentException if {@code size} is negative
   */
  public static int checkElementIndex(int index, int size, @Nullable String desc) {
    // Carefully optimized for execution by hotspot (explanatory comment above)
    if (index < 0 || index >= size) {
      throw new IndexOutOfBoundsException(badElementIndex(index, size, desc));
    }
    return index;
  }

  /**
   * Ensures the truth of an expression involving the state of the calling instance, but not
   * involving any parameters to the calling method.
   *
   * @param expression a boolean expression
   * @throws IllegalStateException if {@code expression} is false
   * @see Verify#verify Verify.verify()
   */
  public static void checkState(boolean expression) {
    if (!expression) {
      throw new IllegalStateException();
    }
  }

  /**
   * Ensures the truth of an expression involving the state of the calling instance, but not
   * involving any parameters to the calling method.
   *
   * @param expression a boolean expression
   * @param errorMessage the exception message to use if the check fails; will be converted to a
   *     string using {@link String#valueOf(Object)}
   * @throws IllegalStateException if {@code expression} is false
   * @see Verify#verify Verify.verify()
   */
  public static void checkState(boolean expression, @Nullable Object errorMessage) {
    if (!expression) {
      throw new IllegalStateException(String.valueOf(errorMessage));
    }
  }

  private static String badElementIndex(int index, int size, @Nullable String desc) {
    if (index < 0) {
      return String.format("%s (%s) must not be negative", desc, index);
    } else if (size < 0) {
      throw new IllegalArgumentException("negative size: " + size);
    } else { // index >= size
      return String.format("%s (%s) must be less than size (%s)", desc, index, size);
    }
  }

  private SupportPreconditions() {
    throw new AssertionError("SupportPreconditions is Uninstantiable.");
  }
}
