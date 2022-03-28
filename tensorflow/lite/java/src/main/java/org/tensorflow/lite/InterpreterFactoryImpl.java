/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

import java.io.File;
import java.nio.ByteBuffer;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.annotations.UsedByReflection;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegateImpl;

/** Package-private factory class for constructing InterpreterApi instances. */
@UsedByReflection("InterpreterFactory.java")
class InterpreterFactoryImpl implements InterpreterFactoryApi {
  public InterpreterFactoryImpl() {}

  @Override
  public InterpreterApi create(@NonNull File modelFile, InterpreterApi.Options options) {
    return new InterpreterImpl(
        modelFile, options == null ? null : new InterpreterImpl.Options(options));
  }

  @Override
  public InterpreterApi create(@NonNull ByteBuffer byteBuffer, InterpreterApi.Options options) {
    return new InterpreterImpl(
        byteBuffer, options == null ? null : new InterpreterImpl.Options(options));
  }

  @Override
  public String runtimeVersion() {
    TensorFlowLite.init();
    return nativeRuntimeVersion();
  }

  @Override
  public String schemaVersion() {
    TensorFlowLite.init();
    return nativeSchemaVersion();
  }

  private static native String nativeRuntimeVersion();

  private static native String nativeSchemaVersion();

  @Override
  public NnApiDelegate.PrivateInterface createNnApiDelegateImpl(NnApiDelegate.Options options) {
    return new NnApiDelegateImpl(options);
  }
}
