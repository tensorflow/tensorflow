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
    /// A test Table object that gives easy access to the slot data
    /// </summary>
    internal struct TestTable
    {
        Table t;

        public TestTable(ByteBuffer bb, int pos)
        {
          t = new Table(pos, bb);
        }

        public bool GetSlot(int slot, bool def)
        {
            var off = t.__offset(slot);

            if (off == 0)
            {
                return def;
            }
            return t.bb.GetSbyte(t.bb_pos + off) != 0;
        }

        public sbyte GetSlot(int slot, sbyte def)
        {
            var off = t.__offset(slot);

            if (off == 0)
            {
                return def;
            }
            return t.bb.GetSbyte(t.bb_pos + off);
        }

        public byte GetSlot(int slot, byte def)
        {
            var off = t.__offset(slot);

            if (off == 0)
            {
                return def;
            }
            return t.bb.Get(t.bb_pos + off);
        }

        public short GetSlot(int slot, short def)
        {
            var off = t.__offset(slot);

            if (off == 0)
            {
                return def;
            }
            return t.bb.GetShort(t.bb_pos + off);
        }

        public ushort GetSlot(int slot, ushort def)
        {
            var off = t.__offset(slot);

            if (off == 0)
            {
                return def;
            }
            return t.bb.GetUshort(t.bb_pos + off);
        }

        public int GetSlot(int slot, int def)
        {
            var off = t.__offset(slot);

            if (off == 0)
            {
                return def;
            }
            return t.bb.GetInt(t.bb_pos + off);
        }

        public uint GetSlot(int slot, uint def)
        {
            var off = t.__offset(slot);

            if (off == 0)
            {
                return def;
            }
            return t.bb.GetUint(t.bb_pos + off);
        }

        public long GetSlot(int slot, long def)
        {
            var off = t.__offset(slot);

            if (off == 0)
            {
                return def;
            }
            return t.bb.GetLong(t.bb_pos + off);
        }

        public ulong GetSlot(int slot, ulong def)
        {
            var off = t.__offset(slot);

            if (off == 0)
            {
                return def;
            }
            return t.bb.GetUlong(t.bb_pos + off);
        }

        public float GetSlot(int slot, float def)
        {
            var off = t.__offset(slot);

            if (off == 0)
            {
                return def;
            }
            return t.bb.GetFloat(t.bb_pos + off);
        }

        public double GetSlot(int slot, double def)
        {
            var off = t.__offset(slot);

            if (off == 0)
            {
                return def;
            }
            return t.bb.GetDouble(t.bb_pos + off);
        }
    }
}
