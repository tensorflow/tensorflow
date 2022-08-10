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

package org.tensorflow.lite;

import java.nio.ByteBuffer;

/**
 * Extension of NativeInterpreterWrapper that adds support for experimental methods.
 *
 * <p><b>WARNING:</b> Resources consumed by the {@code NativeInterpreterWrapperExperimental} object
 * must be explicitly freed by invoking the {@link #close()} method when the {@code
 * NativeInterpreterWrapperExperimental} object is no longer needed.
 *
 * <p>Note: This class is not thread safe.
 */
final class NativeInterpreterWrapperExperimental extends NativeInterpreterWrapper {

  NativeInterpreterWrapperExperimental(String modelPath) {
    super(modelPath);
  }

  NativeInterpreterWrapperExperimental(ByteBuffer byteBuffer) {
    super(byteBuffer);
  }

  NativeInterpreterWrapperExperimental(String modelPath, InterpreterImpl.Options options) {
    super(modelPath, options);
  }

  NativeInterpreterWrapperExperimental(ByteBuffer buffer, InterpreterImpl.Options options) {
    super(buffer, options);
  }

  void resetVariableTensors() {
    resetVariableTensors(interpreterHandle, errorHandle);
  }

  private static native void resetVariableTensors(long interpreterHandle, long errorHandle);
}
