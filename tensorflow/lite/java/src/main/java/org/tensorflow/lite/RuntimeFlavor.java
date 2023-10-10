/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;

/**
 * Represents a TFLite runtime. In contrast to {@link TfLiteRuntime}, this enum represents the
 * actual runtime that is being used, whereas the latter represents a preference for which runtime
 * should be used.
 */
public enum RuntimeFlavor {
  /** A TFLite runtime built directly into the application. */
  APPLICATION,
  /** A TFLite runtime provided by the system (TFLite in Google Play Services). */
  SYSTEM,
}
