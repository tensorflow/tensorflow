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

import org.hamcrest.TypeSafeMatcher;
import org.hamcrest.Description;

public class TensorMatcher {

  /**
  * Creates a matcher for (@link Tensor}s that only matches when the examined {@link Tensor} 
  * has a {@link DataType} that is equal to the specified <code>type</code>.
  * <p/>
  * For example:
  * <pre>assertThat(tensor, hasDataType(DataType.INT64);</pre>
  *
  * @param type 
  *     the type to compare against the DataType of the examined {@link Tensor} 
  */
  public static TypeSafeMatcher<Tensor> hasDataType(final DataType type) {
    return new TypeSafeMatcher<Tensor>() {
      @Override
      protected boolean matchesSafely(final Tensor item) {
        return type == item.dataType();
      }

      @Override
      public void describeTo(final Description description) {
        description.appendText("Tensor with data type of ").appendValue(type);
      }

      @Override
      public void describeMismatchSafely(final Tensor item, final Description description) {
        description.appendText(" was ");
        description.appendText("Tensor with data type of ").appendValue(item.dataType());
      }
    };
  }

  /**
  * Creates a matcher for {@link Tensor}s that only matches when the examined {@link Tensor}
  * has a {@link Tensor#numDimensions()} of zero, <strong>and</strong> a {@link Tensor#shape()}
  * length of zero.
  * <p/>
  * For example:
  * <pre>assertThat(tensor, is(scalar()));</pre>
  */
  public static TypeSafeMatcher<Tensor> scalar() {
    return new TypeSafeMatcher<Tensor>() {

      @Override
      protected boolean matchesSafely(final Tensor item) {
        final boolean hasScalarNumDimensions = 0 == item.numDimensions();
        final boolean hasScalarShape = 0 == item.shape().length;

        return hasScalarNumDimensions && hasScalarShape;
      }

      @Override
      public void describeTo(final Description description) {
        description.appendText("Tensor with shape length ").appendValue(0);
        description.appendText(" and number of dimensions ").appendValue(0);
      }

      @Override
      protected void describeMismatchSafely(final Tensor item, final Description description) {
        description.appendText(" was ");
        description.appendText("Tensor with shape length ").appendValue(item.shape().length);
        description.appendText(" and number of dimensions ").appendValue(item.numDimensions());
      }
    };
  }
}
