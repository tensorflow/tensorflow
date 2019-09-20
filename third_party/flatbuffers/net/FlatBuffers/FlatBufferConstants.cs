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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FlatBuffers
{
    public static class FlatBufferConstants
    {
        public const int FileIdentifierLength = 4;
        public const int SizePrefixLength = 4;
        /** A version identifier to force a compile error if someone
        accidentally tries to build generated code with a runtime of
        two mismatched version. Versions need to always match, as
        the runtime and generated code are modified in sync.
        Changes to the C# implementation need to be sure to change
        the version here and in the code generator on every possible
        incompatible change */
        public static void FLATBUFFERS_1_11_1() {}
    }
}
