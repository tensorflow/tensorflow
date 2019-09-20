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

import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

import static com.google.flatbuffers.FlexBuffers.*;
import static com.google.flatbuffers.FlexBuffers.Unsigned.byteToUnsignedInt;
import static com.google.flatbuffers.FlexBuffers.Unsigned.intToUnsignedLong;
import static com.google.flatbuffers.FlexBuffers.Unsigned.shortToUnsignedInt;

/// @file
/// @addtogroup flatbuffers_java_api
/// @{

/**
 * Helper class that builds FlexBuffers
 * <p> This class presents all necessary APIs to create FlexBuffers. A `ByteBuffer` will be used to store the
 * data. It can be created internally, or passed down in the constructor.</p>
 *
 * <p>There are some limitations when compared to original implementation in C++. Most notably:
 * <ul>
 *   <li><p> No support for mutations (might change in the future).</p></li>
 *   <li><p> Buffer size limited to {@link Integer#MAX_VALUE}</p></li>
 *   <li><p> Since Java does not support unsigned type, all unsigned operations accepts an immediate higher representation
 *   of similar type.</p></li>
 * </ul>
 * </p>
 */
public class FlexBuffersBuilder {

    /**
     * No keys or strings will be shared
     */
    public static final int BUILDER_FLAG_NONE = 0;
    /**
     * Keys will be shared between elements. Identical keys will only be serialized once, thus possibly saving space.
     * But serialization performance might be slower and consumes more memory.
     */
    public static final int BUILDER_FLAG_SHARE_KEYS = 1;
    /**
     * Strings will be shared between elements. Identical strings will only be serialized once, thus possibly saving space.
     * But serialization performance might be slower and consumes more memory. This is ideal if you expect many repeated
     * strings on the message.
     */
    public static final int BUILDER_FLAG_SHARE_STRINGS = 1;
    /**
     * Strings and keys will be shared between elements.
     */
    public static final int BUILDER_FLAG_SHARE_KEYS_AND_STRINGS = 3;
    /**
     * Reserved for the future.
     */
    public static final int BUILDER_FLAG_SHARE_KEY_VECTORS = 4;
    /**
     * Reserved for the future.
     */
    public static final int BUILDER_FLAG_SHARE_ALL = 7;

    /// @cond FLATBUFFERS_INTERNAL
    private static final int WIDTH_8 = 0;
    private static final int WIDTH_16 = 1;
    private static final int WIDTH_32 = 2;
    private static final int WIDTH_64 = 3;
    private final ByteBuffer bb;
    private final ArrayList<Value> stack = new ArrayList<>();
    private final HashMap<String, Integer> keyPool = new HashMap<>();
    private final HashMap<String, Integer> stringPool = new HashMap<>();
    private final int flags;
    private boolean finished = false;

    // A lambda to sort map keys
    private Comparator<Value> keyComparator = new Comparator<Value>() {
        @Override
        public int compare(Value o1, Value o2) {
            int ia = o1.key;
            int io =  o2.key;
            byte c1, c2;
            do {
                c1 = bb.get(ia);
                c2 = bb.get(io);
                if (c1 == 0)
                    return c1 - c2;
                ia++;
                io++;
            }
            while (c1 == c2);
            return c1 - c2;
        }
    };
    /// @endcond

    /**
     * Constructs a newly allocated {@code FlexBuffersBuilder} with {@link #BUILDER_FLAG_SHARE_KEYS} set.
     * @param bufSize size of buffer in bytes.
     */
    public FlexBuffersBuilder(int bufSize) {
        this(ByteBuffer.allocate(bufSize), BUILDER_FLAG_SHARE_KEYS);
    }

    /**
     * Constructs a newly allocated {@code FlexBuffersBuilder} with {@link #BUILDER_FLAG_SHARE_KEYS} set.
     */
    public FlexBuffersBuilder() {
        this(256);
    }

    /**
     * Constructs a newly allocated {@code FlexBuffersBuilder}.
     *
     * @param bb    `ByteBuffer` that will hold the message
     * @param flags Share flags
     */
    public FlexBuffersBuilder(ByteBuffer bb, int flags) {
        this.bb = bb;
        this.flags = flags;
        bb.order(ByteOrder.LITTLE_ENDIAN);
        bb.position(0);
    }

    /**
     * Constructs a newly allocated {@code FlexBuffersBuilder}.
     * By default same keys will be serialized only once
     * @param bb `ByteBuffer` that will hold the message
     */
    public FlexBuffersBuilder(ByteBuffer bb) {
        this(bb, BUILDER_FLAG_SHARE_KEYS);
    }

    /**
     * Return `ByteBuffer` containing FlexBuffer message. {@code #finish()} must be called before calling this
     * function otherwise an assert will trigger.
     *
     * @return `ByteBuffer` with finished message
     */
    public ByteBuffer getBuffer() {
        assert (finished);
        return bb;
    }

    /**
     * Insert a single boolean into the buffer
     * @param val true or false
     */
    public void putBoolean(boolean val) {
        putBoolean(null, val);
    }

    /**
     * Insert a single boolean into the buffer
     * @param key key used to store element in map
     * @param val true or false
     */
    public void putBoolean(String key, boolean val) {
        stack.add(Value.bool(putKey(key), val));
    }

    private int putKey(String key) {
        if (key == null) {
            return -1;
        }
        int pos = bb.position();
        if ((flags & BUILDER_FLAG_SHARE_KEYS) != 0) {
            if (keyPool.get(key) == null) {
                bb.put(key.getBytes(StandardCharsets.UTF_8));
                bb.put((byte) 0);
                keyPool.put(key, pos);
            } else {
                pos = keyPool.get(key);
            }
        } else {
            bb.put(key.getBytes(StandardCharsets.UTF_8));
            bb.put((byte) 0);
            keyPool.put(key, pos);
        }
        return pos;
    }

    /**
     * Adds a integer into the buff
     * @param val integer
     */
    public void putInt(int val) {
        putInt(null, val);
    }

    /**
     * Adds a integer into the buff
     * @param key key used to store element in map
     * @param val integer
     */
    public void putInt(String key, int val) {
        putInt(key, (long) val);
    }

    /**
     * Adds a integer into the buff
     * @param key key used to store element in map
     * @param val 64-bit integer
     */
    public void putInt(String key, long val) {
        int iKey = putKey(key);
        if (Byte.MIN_VALUE <= val && val <= Byte.MAX_VALUE) {
            stack.add(Value.int8(iKey, (int) val));
        } else if (Short.MIN_VALUE <= val && val <= Short.MAX_VALUE) {
            stack.add(Value.int16(iKey, (int) val));
        } else if (Integer.MIN_VALUE <= val && val <= Integer.MAX_VALUE) {
            stack.add(Value.int32(iKey, (int) val));
        } else {
            stack.add(Value.int64(iKey, val));
        }
    }

    /**
     * Adds a 64-bit integer into the buff
     * @param value integer
     */
    public void putInt(long value) {
        putInt(null, value);
    }

    /**
     * Adds a unsigned integer into the buff.
     * @param value integer representing unsigned value
     */
    public void putUInt(int value) {
        putUInt(null, (long) value);
    }

    /**
     * Adds a unsigned integer (stored in a signed 64-bit integer) into the buff.
     * @param value integer representing unsigned value
     */
    public void putUInt(long value) {
        putUInt(null, value);
    }

    /**
     * Adds a 64-bit unsigned integer (stored as {@link BigInteger}) into the buff.
     * Warning: This operation might be very slow.
     * @param value integer representing unsigned value
     */
    public void putUInt64(BigInteger value) {
        putUInt64(null, value.longValue());
    }

    private void putUInt64(String key, long value) {
        stack.add(Value.uInt64(putKey(key), value));
    }

    private void putUInt(String key, long value) {
        int iKey = putKey(key);
        Value vVal;

        int width = widthUInBits(value);

        if (width == WIDTH_8) {
            vVal = Value.uInt8(iKey, (int)value);
        } else if (width == WIDTH_16) {
            vVal = Value.uInt16(iKey, (int)value);
        } else if (width == WIDTH_32) {
            vVal = Value.uInt32(iKey, (int)value);
        } else {
            vVal = Value.uInt64(iKey, value);
        }
        stack.add(vVal);
    }

    /**
     * Adds a 32-bit float into the buff.
     * @param value float representing value
     */
    public void putFloat(float value) {
        putFloat(null, value);
    }

    /**
     * Adds a 32-bit float into the buff.
     * @param key key used to store element in map
     * @param value float representing value
     */
    public void putFloat(String key, float val) {
        stack.add(Value.float32(putKey(key), val));
    }

    /**
     * Adds a 64-bit float into the buff.
     * @param value float representing value
     */
    public void putFloat(double value) {
        putFloat(null, value);
    }

    /**
     * Adds a 64-bit float into the buff.
     * @param key key used to store element in map
     * @param value float representing value
     */
    public void putFloat(String key, double val) {
        stack.add(Value.float64(putKey(key), val));
    }

    /**
     * Adds a String into the buffer
     * @param value string
     * @return start position of string in the buffer
     */
    public int putString(String value) {
        return putString(null, value);
    }

    /**
     * Adds a String into the buffer
     * @param key key used to store element in map
     * @param value string
     * @return start position of string in the buffer
     */
    public int putString(String key, String val) {
        int iKey = putKey(key);
        if ((flags & FlexBuffersBuilder.BUILDER_FLAG_SHARE_STRINGS) != 0) {
            Integer i = stringPool.get(val);
            if (i == null) {
                Value value = writeString(iKey, val);
                stringPool.put(val, (int) value.iValue);
                stack.add(value);
                return (int) value.iValue;
            } else {
                int bitWidth = widthUInBits(val.length());
                stack.add(Value.blob(iKey, i, FBT_STRING, bitWidth));
                return i;
            }
        } else {
            Value value = writeString(iKey, val);
            stack.add(value);
            return (int) value.iValue;
        }
    }

    private Value writeString(int key, String s) {
        return writeBlob(key, s.getBytes(StandardCharsets.UTF_8), FBT_STRING);
    }

    // in bits to fit a unsigned int
    private static int widthUInBits(long len) {
        if (len <= byteToUnsignedInt((byte)0xff)) return WIDTH_8;
        if (len <= shortToUnsignedInt((short)0xffff)) return WIDTH_16;
        if (len <= intToUnsignedLong(0xffff_ffff)) return WIDTH_32;
        return WIDTH_64;
    }

    private Value writeBlob(int key, byte[] blob, int type) {
        int bitWidth = widthUInBits(blob.length);
        int byteWidth = align(bitWidth);
        writeInt(blob.length, byteWidth);
        int sloc = bb.position();
        bb.put(blob);
        if (type == FBT_STRING) {
            bb.put((byte) 0);
        }
        return Value.blob(key, sloc, type, bitWidth);
    }

    // Align to prepare for writing a scalar with a certain size.
    private int align(int alignment) {
        int byteWidth = 1 << alignment;
        int padBytes = Value.paddingBytes(bb.capacity(), byteWidth);
        while (padBytes-- != 0) {
            bb.put((byte) 0);
        }
        return byteWidth;
    }

    private void writeInt(long value, int byteWidth) {
        switch (byteWidth) {
            case 1: bb.put((byte) value); break;
            case 2: bb.putShort((short) value); break;
            case 4: bb.putInt((int) value); break;
            case 8: bb.putLong(value); break;
        }
    }

    /**
     * Adds a byte array into the message
     * @param value byte array
     * @return position in buffer as the start of byte array
     */
    public int putBlob(byte[] value) {
        return putBlob(null, value);
    }

    /**
     * Adds a byte array into the message
     * @param key key used to store element in map
     * @param value byte array
     * @return position in buffer as the start of byte array
     */
    public int putBlob(String key, byte[] val) {
        int iKey = putKey(key);
        Value value = writeBlob(iKey, val, FBT_BLOB);
        stack.add(value);
        return (int) value.iValue;
    }

    /**
     * Start a new vector in the buffer.
     * @return a reference indicating position of the vector in buffer. This
     * reference must be passed along when the vector is finished using endVector()
     */
    public int startVector() {
        return stack.size();
    }

    /**
     * Finishes a vector, but writing the information in the buffer
     * @param key   key used to store element in map
     * @param start reference for begining of the vector. Returned by {@link startVector()}
     * @param typed boolean indicating wether vector is typed
     * @param fixed boolean indicating wether vector is fixed
     * @return      Reference to the vector
     */
    public int endVector(String key, int start, boolean typed, boolean fixed) {
        int iKey = putKey(key);
        Value vec = createVector(iKey, start, stack.size() - start, typed, fixed, null);
        // Remove temp elements and return vector.
        while (stack.size() > start) {
            stack.remove(stack.size() - 1);
        }
        stack.add(vec);
        return (int) vec.iValue;
    }

    /**
     * Finish writing the message into the buffer. After that no other element must
     * be inserted into the buffer. Also, you must call this function before start using the
     * FlexBuffer message
     * @return `ByteBuffer` containing the FlexBuffer message
     */
    public ByteBuffer finish() {
        // If you hit this assert, you likely have objects that were never included
        // in a parent. You need to have exactly one root to finish a buffer.
        // Check your Start/End calls are matched, and all objects are inside
        // some other object.
        assert (stack.size() == 1);
        // Write root value.
        int byteWidth = align(stack.get(0).elemWidth(bb.position(), 0));
        writeAny(stack.get(0), byteWidth);
        // Write root type.
        bb.put(stack.get(0).storedPackedType());
        // Write root size. Normally determined by parent, but root has no parent :)
        bb.put((byte) byteWidth);
        bb.limit(bb.position());
        this.finished = true;
        return bb;
    }

    /*
     * Create a vector based on the elements stored in the stack
     *
     * @param key    reference to its key
     * @param start  element in the stack
     * @param length size of the vector
     * @param typed  whether is TypedVector or not
     * @param fixed  whether is Fixed vector or not
     * @param keys   Value representing key vector
     * @return Value representing the created vector
     */
    private Value createVector(int key, int start, int length, boolean typed, boolean fixed, Value keys) {
        assert (!fixed || typed); // typed=false, fixed=true combination is not supported.
        // Figure out smallest bit width we can store this vector with.
        int bitWidth = Math.max(WIDTH_8, widthUInBits(length));
        int prefixElems = 1;
        if (keys != null) {
            // If this vector is part of a map, we will pre-fix an offset to the keys
            // to this vector.
            bitWidth = Math.max(bitWidth, keys.elemWidth(bb.position(), 0));
            prefixElems += 2;
        }
        int vectorType = FBT_KEY;
        // Check bit widths and types for all elements.
        for (int i = start; i < stack.size(); i++) {
            int elemWidth = stack.get(i).elemWidth(bb.position(), i + prefixElems);
            bitWidth = Math.max(bitWidth, elemWidth);
            if (typed) {
                if (i == start) {
                    vectorType = stack.get(i).type;
                } else {
                    // If you get this assert, you are writing a typed vector with
                    // elements that are not all the same type.
                    assert (vectorType == stack.get(i).type);
                }
            }
        }
        // If you get this assert, your fixed types are not one of:
        // Int / UInt / Float / Key.
        assert (!fixed || FlexBuffers.isTypedVectorElementType(vectorType));

        int byteWidth = align(bitWidth);
        // Write vector. First the keys width/offset if available, and size.
        if (keys != null) {
            writeOffset(keys.iValue, byteWidth);
            writeInt(1L << keys.minBitWidth, byteWidth);
        }
        if (!fixed) {
            writeInt(length, byteWidth);
        }
        // Then the actual data.
        int vloc = bb.position();
        for (int i = start; i < stack.size(); i++) {
            writeAny(stack.get(i), byteWidth);
        }
        // Then the types.
        if (!typed) {
            for (int i = start; i < stack.size(); i++) {
                bb.put(stack.get(i).storedPackedType(bitWidth));
            }
        }
        return new Value(key, keys != null ? FBT_MAP
                : (typed ? FlexBuffers.toTypedVector(vectorType, fixed ? length : 0)
                : FBT_VECTOR), bitWidth, vloc);
    }

    private void writeOffset(long val, int byteWidth) {
        int reloff = (int) (bb.position() - val);
        assert (byteWidth == 8 || reloff < 1L << (byteWidth * 8));
        writeInt(reloff, byteWidth);
    }

    private void writeAny(final Value val, int byteWidth) {
        switch (val.type) {
            case FBT_NULL:
            case FBT_BOOL:
            case FBT_INT:
            case FBT_UINT:
                writeInt(val.iValue, byteWidth);
                break;
            case FBT_FLOAT:
                writeDouble(val.dValue, byteWidth);
                break;
            default:
                writeOffset(val.iValue, byteWidth);
                break;
        }
    }

    private void writeDouble(double val, int byteWidth) {
        if (byteWidth == 4) {
            bb.putFloat((float) val);
        } else if (byteWidth == 8) {
            bb.putDouble(val);
        }
    }

    /**
     * Start a new map in the buffer.
     * @return a reference indicating position of the map in buffer. This
     * reference must be passed along when the map is finished using endMap()
     */
    public int startMap() {
        return stack.size();
    }

    /**
     * Finishes a map, but writing the information in the buffer
     * @param key   key used to store element in map
     * @param start reference for begining of the map. Returned by {@link startMap()}
     * @return      Reference to the map
     */
    public int endMap(String key, int start) {
        int iKey = putKey(key);

        Collections.sort(stack.subList(start, stack.size()), keyComparator);

        Value keys = createKeyVector(start, stack.size() - start);
        Value vec = createVector(iKey, start, stack.size() - start, false, false, keys);
        // Remove temp elements and return map.
        while (stack.size() > start) {
            stack.remove(stack.size() - 1);
        }
        stack.add(vec);
        return (int) vec.iValue;
    }

    private Value createKeyVector(int start, int length) {
        // Figure out smallest bit width we can store this vector with.
        int bitWidth = Math.max(WIDTH_8, widthUInBits(length));
        int prefixElems = 1;
        // Check bit widths and types for all elements.
        for (int i = start; i < stack.size(); i++) {
            int elemWidth = Value.elemWidth(FBT_KEY, WIDTH_8, stack.get(i).key, bb.position(), i + prefixElems);
            bitWidth = Math.max(bitWidth, elemWidth);
        }

        int byteWidth = align(bitWidth);
        // Write vector. First the keys width/offset if available, and size.
        writeInt(length, byteWidth);
        // Then the actual data.
        int vloc = bb.position();
        for (int i = start; i < stack.size(); i++) {
            int pos = stack.get(i).key;
            assert(pos != -1);
            writeOffset(stack.get(i).key, byteWidth);
        }
        // Then the types.
        return new Value(-1, FlexBuffers.toTypedVector(FBT_KEY,0), bitWidth, vloc);
    }

    private static class Value {
        final int type;
        // for scalars, represents scalar size in bytes
        // for vectors, represents the size
        // for string, length
        final int minBitWidth;
        // float value
        final double dValue;
        // integer value
        long iValue;
        // position of the key associated with this value in buffer
        int key;

        Value(int key, int type, int bitWidth, long iValue) {
            this.key = key;
            this.type = type;
            this.minBitWidth = bitWidth;
            this.iValue = iValue;
            this.dValue = Double.MIN_VALUE;
        }

        Value(int key, int type, int bitWidth, double dValue) {
            this.key = key;
            this.type = type;
            this.minBitWidth = bitWidth;
            this.dValue = dValue;
            this.iValue = Long.MIN_VALUE;
        }

        static Value bool(int key, boolean b) {
            return new Value(key, FBT_BOOL, WIDTH_8, b ? 1 : 0);
        }

        static Value blob(int key, int position, int type, int bitWidth) {
            return new Value(key, type, WIDTH_8, position);
        }

        static Value int8(int key, int value) {
            return new Value(key, FBT_INT, WIDTH_8, value);
        }

        static Value int16(int key, int value) {
            return new Value(key, FBT_INT, WIDTH_16, value);
        }

        static Value int32(int key, int value) {
            return new Value(key, FBT_INT, WIDTH_32, value);
        }

        static Value int64(int key, long value) {
            return new Value(key, FBT_INT, WIDTH_64, value);
        }

        static Value uInt8(int key, int value) {
            return new Value(key, FBT_UINT, WIDTH_8, value);
        }

        static Value uInt16(int key, int value) {
            return new Value(key, FBT_UINT, WIDTH_16, value);
        }

        static Value uInt32(int key, int value) {
            return new Value(key, FBT_UINT, WIDTH_32, value);
        }

        static Value uInt64(int key, long value) {
            return new Value(key, FBT_UINT, WIDTH_64, value);
        }

        static Value float32(int key, float value) {
            return new Value(key, FBT_FLOAT, WIDTH_32, value);
        }

        static Value float64(int key, double value) {
            return new Value(key, FBT_FLOAT, WIDTH_64, value);
        }

        private byte storedPackedType() {
            return storedPackedType(WIDTH_8);
        }

        private byte storedPackedType(int parentBitWidth) {
            return packedType(storedWidth(parentBitWidth), type);
        }

        private static byte packedType(int bitWidth, int type) {
            return (byte) (bitWidth | (type << 2));
        }

        private int storedWidth(int parentBitWidth) {
            if (FlexBuffers.isTypeInline(type)) {
                return Math.max(minBitWidth, parentBitWidth);
            } else {
                return minBitWidth;
            }
        }

        private int elemWidth(int bufSize, int elemIndex) {
            return elemWidth(type, minBitWidth, iValue, bufSize, elemIndex);
        }

        private static int elemWidth(int type, int minBitWidth, long iValue, int bufSize, int elemIndex) {
            if (FlexBuffers.isTypeInline(type)) {
                return minBitWidth;
            } else {
                // We have an absolute offset, but want to store a relative offset
                // elem_index elements beyond the current buffer end. Since whether
                // the relative offset fits in a certain byte_width depends on
                // the size of the elements before it (and their alignment), we have
                // to test for each size in turn.

                // Original implementation checks for largest scalar
                // which is long unsigned int
                for (int byteWidth = 1; byteWidth <= 32; byteWidth *= 2) {
                    // Where are we going to write this offset?
                    int offsetLoc = bufSize + paddingBytes(bufSize, byteWidth) + (elemIndex * byteWidth);
                    // Compute relative offset.
                    long offset = offsetLoc - iValue;
                    // Does it fit?
                    int bitWidth = widthUInBits((int) offset);
                    if (((1L) << bitWidth) == byteWidth)
                        return bitWidth;
                }
                assert (false);  // Must match one of the sizes above.
                return WIDTH_64;
            }
        }

        private static int paddingBytes(int bufSize, int scalarSize) {
            return ((~bufSize) + 1) & (scalarSize - 1);
        }
    }
}

/// @}
