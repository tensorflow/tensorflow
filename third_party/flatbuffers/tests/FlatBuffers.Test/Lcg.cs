/*
 * Copyright 2016 Google Inc. All rights reserved.
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

namespace FlatBuffers.Test
{
    /// <summary>
    /// Lcg Pseudo RNG
    /// </summary>
    internal sealed class Lcg
    {
        private const uint InitialValue = 10000;
        private uint _state;

        public Lcg()
        {
            _state = InitialValue;
        }

        public uint Next()
        {
            return (_state = 69069 * _state + 362437);
        }

        public void Reset()
        {
            _state = InitialValue;
        }
    }
}