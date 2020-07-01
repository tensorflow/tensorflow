/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.support.metadata;

/** Information about the metadata parser that this metadata extractor library is depending on. */
public final class MetadataParser {
  /**
   * The version of the metadata parser that this metadata extractor library is depending on. The
   * value should match the value of "Schema Semantic version" in metadata_schema.fbs.
   */
  public static final String VERSION = "1.0.1";

  private MetadataParser() {}
}
