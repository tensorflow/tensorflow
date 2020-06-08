/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.support.label;

import android.util.Log;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.support.common.SupportPreconditions;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/** Label operation utils. */
public class LabelUtil {
  /**
   * Maps an int value tensor to a list of string labels. It takes an array of strings as the
   * dictionary. Example: if the given tensor is [3, 1, 0], and given labels is ["background",
   * "apple", "banana", "cherry", "date"], the result will be ["date", "banana", "apple"].
   *
   * @param tensorBuffer: A tensor with index values. The values should be non-negative integers,
   *     and each value {@code x} will be converted to {@code labels[x + offset]}. If the tensor is
   *     given as a float {@link TensorBuffer}, values will be cast to integers. All values that are
   *     out of bound will map to empty string.
   * @param labels: A list of strings, used as a dictionary to look up. The index of the array
   *     element will be used as the key. To get better performance, use an object that implements
   *     RandomAccess, such as {@link ArrayList}.
   * @param offset: The offset value when look up int values in the {@code labels}.
   * @return the mapped strings. The length of the list is {@link TensorBuffer#getFlatSize}.
   * @throws IllegalArgumentException if {@code tensorBuffer} or {@code labels} is null.
   */
  public static List<String> mapValueToLabels(
      @NonNull TensorBuffer tensorBuffer, @NonNull List<String> labels, int offset) {
    SupportPreconditions.checkNotNull(tensorBuffer, "Given tensor should not be null");
    SupportPreconditions.checkNotNull(labels, "Given labels should not be null");
    int[] values = tensorBuffer.getIntArray();
    Log.d("values", Arrays.toString(values));
    List<String> result = new ArrayList<>();
    for (int v : values) {
      int index = v + offset;
      if (index < 0 || index >= labels.size()) {
        result.add("");
      } else {
        result.add(labels.get(index));
      }
    }
    return result;
  }

  // Private constructor to prevent initialization.
  private LabelUtil() {}
}
