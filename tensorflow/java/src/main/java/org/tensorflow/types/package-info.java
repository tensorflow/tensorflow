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
 * Defines classes that represent TensorFlow data types. For each possible data type
 * that can be used in a tensor, there is a corresponding class in this package that
 * is used to represent it. For example, the TensorFlow int32 type is represented by
 * the type TFInt32 and by the class object TFInt32.class. The former is used to
 * support compile-time checking of tensor data types and the latter is used for
 * run-time checking of data types. All such classes implement the TFType interface.
 *`
 * <p><b>WARNING</b>: The API is currently experimental and is not covered by TensorFlow <a
 * href="https://www.tensorflow.org/programmers_guide/version_semantics">API stability
 * guarantees</a>. See <a
 * href="https://www.tensorflow.org/code/tensorflow/java/README.md">README.md</a> for installation
 * instructions.
 *
 */
package org.tensorflow.types;
