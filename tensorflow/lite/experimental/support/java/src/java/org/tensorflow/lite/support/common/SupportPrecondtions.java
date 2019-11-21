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

import android.text.TextUtils;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Static error checking util methods. */
public final class SupportPrecondtions {
  /**
   * Ensures that an object reference passed as a parameter to the calling method is not null.
   *
   * @param reference an object reference
   * @return the non-null reference that was validated
   * @throws NullPointerException if {@code reference} is null
   */
  public static <T extends @NonNull Object> T checkNotNull(T reference) {
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
  public static <T extends @NonNull Object> T checkNotNull(
      T reference, @Nullable Object errorMessage) {
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
    if (TextUtils.isEmpty(string)) {
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
    if (TextUtils.isEmpty(string)) {
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

  private SupportPrecondtions() {
    throw new AssertionError("SupportPrecondtions is Uninstantiable.");
  }
}
