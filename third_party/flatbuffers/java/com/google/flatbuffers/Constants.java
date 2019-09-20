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

/// @cond FLATBUFFERS_INTERNAL

/**
 * Class that holds shared constants
 */
public class Constants {
    // Java doesn't seem to have these.
    /** The number of bytes in an `byte`. */
    static final int SIZEOF_BYTE = 1;
    /** The number of bytes in a `short`. */
    static final int SIZEOF_SHORT = 2;
    /** The number of bytes in an `int`. */
    static final int SIZEOF_INT = 4;
    /** The number of bytes in an `float`. */
    static final int SIZEOF_FLOAT = 4;
    /** The number of bytes in an `long`. */
    static final int SIZEOF_LONG = 8;
    /** The number of bytes in an `double`. */
    static final int SIZEOF_DOUBLE = 8;
    /** The number of bytes in a file identifier. */
    static final int FILE_IDENTIFIER_LENGTH = 4;
    /** The number of bytes in a size prefix. */
    public static final int SIZE_PREFIX_LENGTH = 4;
    /** A version identifier to force a compile error if someone
    accidentally tries to build generated code with a runtime of
    two mismatched version. Versions need to always match, as
    the runtime and generated code are modified in sync.
    Changes to the Java implementation need to be sure to change
    the version here and in the code generator on every possible
    incompatible change */
    public static void FLATBUFFERS_1_11_1() {}
}

/// @endcond
