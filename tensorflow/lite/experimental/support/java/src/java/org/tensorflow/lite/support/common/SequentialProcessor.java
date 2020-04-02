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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.checkerframework.checker.nullness.qual.NonNull;

/**
 * A processor base class that chains a serial of {@link Operator<T>} and executes them.
 *
 * <p>Typically, users could use its subclasses, e.g. {@link
 * org.tensorflow.lite.support.image.ImageProcessor} rather than directly use this one.
 *
 * @param <T> The type that the Operator is handling.
 */
public class SequentialProcessor<T> implements Processor<T> {

  /** List of operators added to this {@link SequentialProcessor}. */
  protected final List<Operator<T>> operatorList;
  /**
   * The {@link Map} between the operator name and the corresponding op indexes in {@code
   * operatorList}. An operator may be added multiple times into this {@link SequentialProcessor}.
   */
  protected final Map<String, List<Integer>> operatorIndex;

  protected SequentialProcessor(Builder<T> builder) {
    operatorList = builder.operatorList;
    operatorIndex = Collections.unmodifiableMap(builder.operatorIndex);
  }

  @Override
  public T process(T x) {
    for (Operator<T> op : operatorList) {
      x = op.apply(x);
    }
    return x;
  }

  /** The inner builder class to build a Sequential Processor. */
  protected static class Builder<T> {

    private final List<Operator<T>> operatorList;
    private final Map<String, List<Integer>> operatorIndex;

    protected Builder() {
      operatorList = new ArrayList<>();
      operatorIndex = new HashMap<>();
    }

    public Builder<T> add(@NonNull Operator<T> op) {
      SupportPreconditions.checkNotNull(op, "Adding null Op is illegal.");
      operatorList.add(op);
      String operatorName = op.getClass().getName();
      if (!operatorIndex.containsKey(operatorName)) {
        operatorIndex.put(operatorName, new ArrayList<Integer>());
      }
      operatorIndex.get(operatorName).add(operatorList.size() - 1);
      return this;
    }

    public SequentialProcessor<T> build() {
      return new SequentialProcessor<T>(this);
    }
  }
}
