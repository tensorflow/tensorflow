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

package org.tensorflow.op;

/**
 * A marker interface for all operation wrappers.
 *
 * <p>Operation wrappers provide strongly typed interfaces for building and execution operations
 * without the use of literals and indexes, as required in the core classes.
 *
 * <p>This interface allows keeping references to any operation wrapper using a common type.
 *
 * <pre>{@code
 * // All values returned by an Ops call can be referred as a Op
 * Op split = ops.array().split(...);
 * Op shape = ops.array().shape(...);
 *
 * // All operations could be added to an Op collection
 * Collection<Op> allOps = Arrays.asList(split, shape);
 * }
 */
public interface Op {}
