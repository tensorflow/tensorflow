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

import java.util.Objects;

/**
 * Category is a util class, contains a label and a float value. Typically it's used as result of
 * classification tasks.
 */
public final class Category {
  private final String label;
  private final float score;

  /** Constructs a Category. */
  public Category(String label, float score) {
    this.label = label;
    this.score = score;
  }

  /** Gets the reference of category's label. */
  public String getLabel() {
    return label;
  }

  /** Gets the score of the category. */
  public float getScore() {
    return score;
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof Category) {
      Category other = (Category) o;
      return (other.getLabel().equals(this.label) && other.getScore() == this.score);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hash(label, score);
  }

  @Override
  public String toString() {
    return "<Category \"" + label + "\" (score=" + score + ")>";
  }
}
