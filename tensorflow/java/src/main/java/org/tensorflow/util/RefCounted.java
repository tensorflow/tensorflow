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

package org.tensorflow.util;

/**
 * A reference-counted object.
 *
 * <p>Newly-instantiated {@RefCounted} objects have a reference count of one. Once the reference count
 * reaches zero, the object is considered deallocated. The general rule-of-thumb is that the party that
 * last accesses a reference-counted object releases it.
 *
 * <p>Note: see {@link org.tensorflow.Tensor} for a usage example.
 */
public interface RefCounted {

  /**
    * Returns the reference count of this object.   A value of zero means that the object has been deallocated.
    * @return the current reference count.
    */
  int refCount();

  /**
   * Increments the reference count by one.
   * @return this instance.
   */
  RefCounted ref();

  /**
   * Decrements the reference count by one and deallocates the object if the reference count reaches zero.
   * @return true if the reference count becomes zero, false otherwise.
   * @throws IllegalStateException if the reference count is already zero.
   */
  boolean unref();
}
