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
import org.tensorflow.lite.nnapi.NnApiDelegate;

/**
 * Private interface specifying factory for constructing InterpreterApi instances. This interface is
 * an implementation detail of InterpreterFactory and should only be used from within the TensorFlow
 * Lite implementation. We can't make it package-private, though, because it is used from both
 * org.tensorflow.lite.InterpreterFactoryImpl and
 * com.google.android.gms.tflite.InterpreterFactoryImpl.
 *
 * @hide
 */
public interface InterpreterFactoryApi {
  /**
   * Constructs an {@link InterpreterApi} instance, using the specified model and options. The model
   * will be loaded from a file.
   *
   * @param modelFile A file containing a pre-trained TF Lite model.
   * @param options A set of options for customizing interpreter behavior.
   * @throws IllegalArgumentException if {@code modelFile} does not encode a valid TensorFlow Lite
   *     model.
   */
  InterpreterApi create(@NonNull File modelFile, InterpreterApi.Options options);

  /**
   * Constructs an {@link InterpreterApi} instance, using the specified model and options. The model
   * will be read from a {@code ByteBuffer}.
   *
   * @param byteBuffer A pre-trained TF Lite model, in binary serialized form. The ByteBuffer should
   *     not be modified after the construction of an {@link InterpreterApi} instance. The {@code
   *     ByteBuffer} can be either a {@code MappedByteBuffer} that memory-maps a model file, or a
   *     direct {@code ByteBuffer} of nativeOrder() that contains the bytes content of a model.
   * @param options A set of options for customizing interpreter behavior.
   * @throws IllegalArgumentException if {@code byteBuffer} is not a {@code MappedByteBuffer} nor a
   *     direct {@code ByteBuffer} of nativeOrder.
   */
  InterpreterApi create(@NonNull ByteBuffer byteBuffer, InterpreterApi.Options options);

  /** Returns the version of the underlying TensorFlowLite runtime. */
  String runtimeVersion();

  /**
   * Returns the version of the TensorFlowLite model schema that is supported by the underlying
   * TensorFlowLite runtime.
   */
  String schemaVersion();

  /**
   * Instance method for constructing an NNAPI delegate implementation, using the TF Lite runtime
   * from the InterpreterFactoryApi.
   */
  NnApiDelegate.PrivateInterface createNnApiDelegateImpl(NnApiDelegate.Options options);
}
