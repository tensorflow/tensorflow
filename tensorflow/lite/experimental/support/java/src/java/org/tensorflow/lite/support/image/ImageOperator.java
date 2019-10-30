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

package org.tensorflow.lite.support.image;

import org.tensorflow.lite.support.common.Operator;

/**
 * Operates a TensorImage object. Used in ImageProcessor.
 */
public interface ImageOperator extends Operator<TensorImage> {
  /** @see org.tensorflow.lite.support.common.Operator#apply(java.lang.Object) */
  @Override
  TensorImage apply(TensorImage image);
}
