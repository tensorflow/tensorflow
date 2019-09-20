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

namespace FlatBuffers
{
    /// <summary>
    /// All structs in the generated code derive from this class, and add their own accessors.
    /// </summary>
    public struct Struct
    {
        public int bb_pos { get; private set; }
        public ByteBuffer bb { get; private set; }

        // Re-init the internal state with an external buffer {@code ByteBuffer} and an offset within.
        public Struct(int _i, ByteBuffer _bb)
        {
            bb = _bb;
            bb_pos = _i;
        }
    }
}
