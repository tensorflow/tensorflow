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

/**
 * A tensor that can be used as an operand for operation wrappers that require variables.
 *
 * <p>This interface allows operation wrappers that require a variable tensor enforce it at
 * compile-time by requiring this interface as an operand. Note that a {@code VariableInput} can
 * also be used as an {@code Input}, but not the other way around.
 *
 * <pre>{@code
 * VariableInput var = ....;
 * ops.math.mean(var, ...); // will work
 * ops.training.applyGradientDescent(var, ...); // will work
 *
 * Input input = ...;
 * ops.math.mean(input, ...); // will work
 * ops.training.applyGradientDescent(input, ...); // won't work
 * }</pre>
 */
public interface VariableInput extends Input {}
