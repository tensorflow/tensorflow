/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.flatbuffers;

import java.nio.ByteBuffer;

/// @cond FLATBUFFERS_INTERNAL

/**
 * All structs in the generated code derive from this class, and add their own accessors.
 */
public class Struct {
  /** Used to hold the position of the `bb` buffer. */
  protected int bb_pos;
  /** The underlying ByteBuffer to hold the data of the Struct. */
  protected ByteBuffer bb;

  /**
   * Re-init the internal state with an external buffer {@code ByteBuffer} and an offset within.
   *
   * This method exists primarily to allow recycling Table instances without risking memory leaks
   * due to {@code ByteBuffer} references.
   */
  protected void __reset(int _i, ByteBuffer _bb) { 
    bb = _bb;
    if (bb != null) {
      bb_pos = _i;
    } else {
      bb_pos = 0;
    }
  }

  /**
   * Resets internal state with a null {@code ByteBuffer} and a zero position.
   *
   * This method exists primarily to allow recycling Struct instances without risking memory leaks
   * due to {@code ByteBuffer} references. The instance will be unusable until it is assigned
   * again to a {@code ByteBuffer}.
   *
   * @param struct the instance to reset to initial state
   */
  public void __reset() {
    __reset(0, null);
  }
}

/// @endcond
