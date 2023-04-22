/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

/**
 * Defines classes that represent TensorFlow data types. For each possible data type that can be
 * used in a tensor, there is a corresponding class that is used to represent it. For example, the
 * TensorFlow int32 type is represented by the type {@link java.lang.Integer} and by the class
 * object {@code Integer.class}. The former is used to support compile-time checking of tensor
 * element types and the latter is used for run-time checking of element types. Classes appearing in
 * this package, such as UInt8, represent TensorFlow data types for which there is no existing Java
 * equivalent.
 *
 * <p>TensorFlow element types are also separately represented by the {@link
 * org.tensorflow.DataType} enum, with one enum value per element type. The enum representation is
 * not usually needed, but can be obtained using {@link org.tensorflow.DataType#fromClass}.
 */
package org.tensorflow.types;
