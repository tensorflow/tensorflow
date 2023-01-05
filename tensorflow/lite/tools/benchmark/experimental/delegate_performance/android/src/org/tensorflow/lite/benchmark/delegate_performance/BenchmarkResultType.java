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

package org.tensorflow.lite.benchmark.delegateperformance;

/** Enumerates the possible benchmark result values. */
public enum BenchmarkResultType {
  /** Unknown benchmark result, possibly due to internal failures. */
  UNKONWN("UNKNOWN"),
  /** The benchmark activity skips the Pass/Fail result generation. */
  SKIP("SKIP"),
  /** All benchmark results don't breach the thresholds specified in the criteria file. */
  PASS("PASS"),
  /** Some benchmark results breach the thresholds specified in the criteria file. */
  FAIL("FAIL");

  private final String name;

  BenchmarkResultType(String name) {
    this.name = name;
  }

  @Override
  public String toString() {
    return name;
  }
}
