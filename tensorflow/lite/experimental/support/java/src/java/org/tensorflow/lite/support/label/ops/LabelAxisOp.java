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

package org.tensorflow.lite.support.label.ops;

import android.content.Context;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.SupportPrecondtions;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/**
 * Labels TensorBuffer with axisLabels for outputs.
 *
 * <p>Apply on a {@code TensorBuffer} to get a {@code TensorLabel} that could output a Map, which is
 * a pair of the label name and the corresponding TensorBuffer value.
 */
public class LabelAxisOp {
  // Axis and its corresponding label names.
  private final Map<Integer, List<String>> axisLabels;

  protected LabelAxisOp(Builder builder) {
    axisLabels = builder.axisLabels;
  }

  public TensorLabel apply(@NonNull TensorBuffer buffer) {
    SupportPrecondtions.checkNotNull(buffer, "Tensor buffer cannot be null.");
    return new TensorLabel(axisLabels, buffer);
  }

  /** The inner builder class to build a LabelTensor Operator. */
  public static class Builder {
    private final Map<Integer, List<String>> axisLabels;

    protected Builder() {
      axisLabels = new HashMap<>();
    }

    public Builder addAxisLabel(@NonNull Context context, int axis, @NonNull String filePath)
        throws IOException {
      SupportPrecondtions.checkNotNull(context, "Context cannot be null.");
      SupportPrecondtions.checkNotNull(filePath, "File path cannot be null.");
      List<String> labels = FileUtil.loadLabels(context, filePath);
      axisLabels.put(axis, labels);
      return this;
    }

    public Builder addAxisLabel(int axis, @NonNull List<String> labels) {
      axisLabels.put(axis, labels);
      return this;
    }

    public LabelAxisOp build() {
      return new LabelAxisOp(this);
    }
  }
}
