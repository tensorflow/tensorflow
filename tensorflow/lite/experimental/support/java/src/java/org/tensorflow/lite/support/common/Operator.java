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

/**
 * The common interface for classes that carries an "apply" method, which converts T to another one.
 * @param <T> The class which Operator handles.
 */
public interface Operator<T> {

  /**
   * Applies an operation on a T object, returning a T object.
   *
   * <p>Note: The returned object could probably be the same one with given input, and given input
   * could probably be changed.
   */
  T apply(T x);
}
