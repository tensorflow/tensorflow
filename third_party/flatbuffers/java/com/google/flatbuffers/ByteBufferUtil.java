/*
 * Copyright 2017 Google Inc. All rights reserved.
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

import static com.google.flatbuffers.Constants.*;

import java.nio.ByteBuffer;

/// @file
/// @addtogroup flatbuffers_java_api
/// @{

/**
 * Class that collects utility functions around `ByteBuffer`.
 */
public class ByteBufferUtil {

	/**
     * Extract the size prefix from a `ByteBuffer`.
     * 
     * @param bb a size-prefixed buffer
     * @return the size prefix
     */
    public static int getSizePrefix(ByteBuffer bb) {
        return bb.getInt(bb.position());
    }

	/**
     * Create a duplicate of a size-prefixed `ByteBuffer` that has its position
     * advanced just past the size prefix.
     * 
     * @param bb a size-prefixed buffer
     * @return a new buffer on the same underlying data that has skipped the
     *         size prefix
     */
    public static ByteBuffer removeSizePrefix(ByteBuffer bb) {
        ByteBuffer s = bb.duplicate();
        s.position(s.position() + SIZE_PREFIX_LENGTH);
        return s;
    }

}

/// @}
