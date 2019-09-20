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


import static com.google.flatbuffers.FlexBuffers.Unsigned.byteToUnsignedInt;
import static com.google.flatbuffers.FlexBuffers.Unsigned.intToUnsignedLong;
import static com.google.flatbuffers.FlexBuffers.Unsigned.shortToUnsignedInt;

import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/// @file
/// @addtogroup flatbuffers_java_api
/// @{

/**
 * This class can be used to parse FlexBuffer messages.
 * <p>
 * For generating FlexBuffer messages, use {@link FlexBuffersBuilder}.
 * <p>
 * Example of usage:
 * <pre>
 * ByteBuffer bb = ... // load message from file or network
 * FlexBuffers.Reference r = FlexBuffers.getRoot(bb); // Reads the root element
 * FlexBuffers.Map map = r.asMap(); // We assumed root object is a map
 * System.out.println(map.get("name").asString()); // prints element with key "name"
 * </pre>
 */
public class FlexBuffers {

    // These are used as the upper 6 bits of a type field to indicate the actual
    // type.
    /** Represent a null type */
    public static final int FBT_NULL = 0;
    /** Represent a signed integer type */
    public static final int FBT_INT = 1;
    /** Represent a unsigned type */
    public static final int FBT_UINT = 2;
    /** Represent a float type */
    public static final int FBT_FLOAT = 3; // Types above stored inline, types below store an offset.
    /** Represent a key to a map type */
    public static final int FBT_KEY = 4;
    /** Represent a string type */
    public static final int FBT_STRING = 5;
    /** Represent a indirect signed integer type */
    public static final int FBT_INDIRECT_INT = 6;
    /** Represent a indirect unsigned integer type */
    public static final int FBT_INDIRECT_UINT = 7;
    /** Represent a indirect float type */
    public static final int FBT_INDIRECT_FLOAT = 8;
    /** Represent a map type */
    public static final int FBT_MAP = 9;
    /** Represent a vector type */
    public static final int FBT_VECTOR = 10; // Untyped.
    /** Represent a vector of signed integers type */
    public static final int FBT_VECTOR_INT = 11;  // Typed any size  = stores no type table).
    /** Represent a vector of unsigned integers type */
    public static final int FBT_VECTOR_UINT = 12;
    /** Represent a vector of floats type */
    public static final int FBT_VECTOR_FLOAT = 13;
    /** Represent a vector of keys type */
    public static final int FBT_VECTOR_KEY = 14;
    /** Represent a vector of strings type */
    public static final int FBT_VECTOR_STRING = 15;

    /// @cond FLATBUFFERS_INTERNAL
    public static final int FBT_VECTOR_INT2 = 16;  // Typed tuple  = no type table; no size field).
    public static final int FBT_VECTOR_UINT2 = 17;
    public static final int FBT_VECTOR_FLOAT2 = 18;
    public static final int FBT_VECTOR_INT3 = 19;  // Typed triple  = no type table; no size field).
    public static final int FBT_VECTOR_UINT3 = 20;
    public static final int FBT_VECTOR_FLOAT3 = 21;
    public static final int FBT_VECTOR_INT4 = 22;  // Typed quad  = no type table; no size field).
    public static final int FBT_VECTOR_UINT4 = 23;
    public static final int FBT_VECTOR_FLOAT4 = 24;
    /// @endcond FLATBUFFERS_INTERNAL

    /** Represent a blob type */
    public static final int FBT_BLOB = 25;
    /** Represent a boolean type */
    public static final int FBT_BOOL = 26;
    /** Represent a vector of booleans type */
    public static final int FBT_VECTOR_BOOL = 36;  // To Allow the same type of conversion of type to vector type

    private static final ByteBuffer EMPTY_BB = ByteBuffer.allocate(0).asReadOnlyBuffer();

    /**
     * Checks where a type is a typed vector
     *
     * @param type type to be checked
     * @return true if typed vector
     */
    static boolean isTypedVector(int type) {
        return (type >= FBT_VECTOR_INT && type <= FBT_VECTOR_STRING) || type == FBT_VECTOR_BOOL;
    }

    /**
     * Check whether you can access type directly (no indirection) or not.
     *
     * @param type type to be checked
     * @return true if inline type
     */
    static boolean isTypeInline(int type) {
        return type <= FBT_FLOAT || type == FBT_BOOL;
    }

    static int toTypedVectorElementType(int original_type) {
        return original_type - FBT_VECTOR_INT + FBT_INT;
    }

    /**
     * Return a vector type our of a original element type
     *
     * @param type        element type
     * @param fixedLength size of element
     * @return typed vector type
     */
    static int toTypedVector(int type, int fixedLength) {
        assert (isTypedVectorElementType(type));
        switch (fixedLength) {
            case 0: return type - FBT_INT + FBT_VECTOR_INT;
            case 2: return type - FBT_INT + FBT_VECTOR_INT2;
            case 3: return type - FBT_INT + FBT_VECTOR_INT3;
            case 4: return type - FBT_INT + FBT_VECTOR_INT4;
            default:
                assert (false);
                return FBT_NULL;
        }
    }

    static boolean isTypedVectorElementType(int type) {
        return (type >= FBT_INT && type <= FBT_STRING) || type == FBT_BOOL;
    }

    // return position of the element that the offset is pointing to
    private static int indirect(ByteBuffer bb, int offset, int byteWidth) {
        // we assume all offset fits on a int, since ByteBuffer operates with that assumption
        return (int) (offset - readUInt(bb, offset, byteWidth));
    }

    // read unsigned int with size byteWidth and return as a 64-bit integer
    private static long readUInt(ByteBuffer buff, int end, int byteWidth) {
        switch (byteWidth) {
            case 1: return byteToUnsignedInt(buff.get(end));
            case 2: return shortToUnsignedInt(buff.getShort(end));
            case 4: return intToUnsignedLong(buff.getInt(end));
            case 8: return buff.getLong(end); // We are passing signed long here. Losing information (user should know)
            default: return -1; // we should never reach here
        }
    }

    // read signed int of size byteWidth and return as 32-bit int
    private static int readInt(ByteBuffer buff, int end, int byteWidth) {
        return (int) readLong(buff, end, byteWidth);
    }

    // read signed int of size byteWidth and return as 64-bit int
    private static long readLong(ByteBuffer buff, int end, int byteWidth) {
        switch (byteWidth) {
            case 1: return buff.get(end);
            case 2: return buff.getShort(end);
            case 4: return buff.getInt(end);
            case 8: return buff.getLong(end);
            default: return -1; // we should never reach here
        }
    }

    private static double readDouble(ByteBuffer buff, int end, int byteWidth) {
        switch (byteWidth) {
            case 4: return buff.getFloat(end);
            case 8: return buff.getDouble(end);
            default: return -1; // we should never reach here
        }
    }

    /**
     * Reads a FlexBuffer message in ByteBuffer and returns {@link Reference} to
     * the root element.
     * @param buffer ByteBuffer containing FlexBuffer message
     * @return {@link Reference} to the root object
     */
    public static Reference getRoot(ByteBuffer buffer) {
        // See Finish() below for the serialization counterpart of this.
        // The root ends at the end of the buffer, so we parse backwards from there.
        int end = buffer.limit();
        int byteWidth = buffer.get(--end);
        int packetType = byteToUnsignedInt(buffer.get(--end));
        end -= byteWidth;  // The root data item.
        return new Reference(buffer, end, byteWidth, packetType);
    }

    /**
     * Represents an generic element in the buffer.
     */
    public static class Reference {

        private static final Reference NULL_REFERENCE = new Reference(EMPTY_BB, 0, 1, 0);
        private ByteBuffer bb;
        private int end;
        private int parentWidth;
        private int byteWidth;
        private int type;

        Reference(ByteBuffer bb, int end, int parentWidth, int packedType) {
            this(bb, end, parentWidth, (1 << (packedType & 3)), packedType >> 2);
        }

        Reference(ByteBuffer bb, int end, int parentWidth, int byteWidth, int type) {
            this.bb = bb;
            this.end = end;
            this.parentWidth = parentWidth;
            this.byteWidth = byteWidth;
            this.type = type;
        }

        /**
         * Return element type
         * @return element type as integer
         */
        public int getType() {
            return type;
        }

        /**
         * Checks whether the element is null type
         * @return true if null type
         */
        public boolean isNull() {
            return type == FBT_NULL;
        }
         
        /**
         * Checks whether the element is boolean type
         * @return true if boolean type
         */
        public boolean isBoolean() {
            return type == FBT_BOOL;
        }

        /**
         * Checks whether the element type is numeric (signed/unsigned integers and floats)
         * @return true if numeric type
         */
        public boolean isNumeric() {
            return isIntOrUInt() || isFloat();
        }

        /**
         * Checks whether the element type is signed or unsigned integers
         * @return true if an integer type
         */
        public boolean isIntOrUInt() {
            return isInt() || isUInt();
        }

        /**
         * Checks whether the element type is float
         * @return true if a float type
         */
        public boolean isFloat() {
            return type == FBT_FLOAT || type == FBT_INDIRECT_FLOAT;
        }

        /**
         * Checks whether the element type is signed integer
         * @return true if a signed integer type
         */
        public boolean isInt() {
            return type == FBT_INT || type == FBT_INDIRECT_INT;
        }

        /**
         * Checks whether the element type is signed integer
         * @return true if a signed integer type
         */
        public boolean isUInt() {
            return type == FBT_UINT || type == FBT_INDIRECT_UINT;
        }

        /**
         * Checks whether the element type is string
         * @return true if a string type
         */
        public boolean isString() {
            return type == FBT_STRING;
        }

        /**
         * Checks whether the element type is key
         * @return true if a key type
         */
        public boolean isKey() {
            return type == FBT_KEY;
        }

        /**
         * Checks whether the element type is vector
         * @return true if a vector type
         */
        public boolean isVector() {
            return type == FBT_VECTOR || type == FBT_MAP;
        }

        /**
         * Checks whether the element type is typed vector
         * @return true if a typed vector type
         */
        public boolean isTypedVector() {
            return (type >= FBT_VECTOR_INT && type <= FBT_VECTOR_STRING) ||
                    type == FBT_VECTOR_BOOL;
        }

        /**
         * Checks whether the element type is a map
         * @return true if a map type
         */
        public boolean isMap() {
            return type == FBT_MAP;
        }

        /**
         * Checks whether the element type is a blob
         * @return true if a blob type
         */
        public boolean isBlob() {
            return type == FBT_BLOB;
        }

        /**
         * Returns element as 32-bit integer.
         * <p> For vector element, it will return size of the vector</p>
         * <p> For String element, it will type to be parsed as integer</p>
         * <p> Unsigned elements will become negative</p>
         * <p> Float elements will be casted to integer </p>
         * @return 32-bit integer or 0 if fail to convert element to integer.
         */
        public int asInt() {
            if (type == FBT_INT) {
                // A fast path for the common case.
                return readInt(bb, end, parentWidth);
            } else
                switch (type) {
                    case FBT_INDIRECT_INT: return readInt(bb, indirect(bb, end, parentWidth), byteWidth);
                    case FBT_UINT: return (int) readUInt(bb, end, parentWidth);
                    case FBT_INDIRECT_UINT: return (int) readUInt(bb, indirect(bb, end, parentWidth), parentWidth);
                    case FBT_FLOAT: return (int) readDouble(bb, end, parentWidth);
                    case FBT_INDIRECT_FLOAT: return (int) readDouble(bb, indirect(bb, end, parentWidth), byteWidth);
                    case FBT_NULL: return 0;
                    case FBT_STRING: return Integer.parseInt(asString());
                    case FBT_VECTOR: return asVector().size();
                    case FBT_BOOL: return readInt(bb, end, parentWidth);
                    default:
                        // Convert other things to int.
                        return 0;
                }
        }

        /**
         * Returns element as unsigned 64-bit integer.
         * <p> For vector element, it will return size of the vector</p>
         * <p> For String element, it will type to be parsed as integer</p>
         * <p> Negative signed elements will become unsigned counterpart</p>
         * <p> Float elements will be casted to integer </p>
         * @return 64-bit integer or 0 if fail to convert element to integer.
         */
        public long asUInt() {
            if (type == FBT_UINT) {
                // A fast path for the common case.
                return readUInt(bb, end, parentWidth);
            } else
                switch (type) {
                    case FBT_INDIRECT_UINT: return readUInt(bb, indirect(bb, end, parentWidth), byteWidth);
                    case FBT_INT: return readLong(bb, end, parentWidth);
                    case FBT_INDIRECT_INT: return readLong(bb, indirect(bb, end, parentWidth), byteWidth);
                    case FBT_FLOAT: return (long) readDouble(bb, end, parentWidth);
                    case FBT_INDIRECT_FLOAT: return (long) readDouble(bb,  indirect(bb, end, parentWidth), parentWidth);
                    case FBT_NULL: return 0;
                    case FBT_STRING: return Long.parseLong(asString());
                    case FBT_VECTOR: return asVector().size();
                    case FBT_BOOL: readInt(bb, end, parentWidth);
                    default:
                        // Convert other things to uint.
                        return 0;
                }
        }

        /**
         * Returns element as 64-bit integer.
         * <p> For vector element, it will return size of the vector</p>
         * <p> For String element, it will type to be parsed as integer</p>
         * <p> Unsigned elements will become negative</p>
         * <p> Float elements will be casted to integer </p>
         * @return 64-bit integer or 0 if fail to convert element to long.
         */
        public long asLong() {
            if (type == FBT_INT) {
                // A fast path for the common case.
                return readLong(bb, end, parentWidth);
            } else
                switch (type) {
                    case FBT_INDIRECT_INT: return readLong(bb, indirect(bb, end, parentWidth), byteWidth);
                    case FBT_UINT: return readUInt(bb, end, parentWidth);
                    case FBT_INDIRECT_UINT: return readUInt(bb, indirect(bb, end, parentWidth), parentWidth);
                    case FBT_FLOAT: return (long) readDouble(bb, end, parentWidth);
                    case FBT_INDIRECT_FLOAT: return (long) readDouble(bb, indirect(bb, end, parentWidth), byteWidth);
                    case FBT_NULL: return 0;
                    case FBT_STRING: {
                        try {
                            return Long.parseLong(asString());
                        } catch (NumberFormatException nfe) {
                            return 0; //same as C++ implementation
                        }
                    }
                    case FBT_VECTOR: return asVector().size();
                    case FBT_BOOL: return readInt(bb, end, parentWidth);
                    default:
                        // Convert other things to int.
                        return 0;
                }
        }

        /**
         * Returns element as 64-bit integer.
         * <p> For vector element, it will return size of the vector</p>
         * <p> For String element, it will type to be parsed as integer</p>
         * @return 64-bit integer or 0 if fail to convert element to long.
         */
        public double asFloat() {
            if (type == FBT_FLOAT) {
                // A fast path for the common case.
                return readDouble(bb, end, parentWidth);
            } else
                switch (type) {
                    case FBT_INDIRECT_FLOAT: return readDouble(bb, indirect(bb, end, parentWidth), byteWidth);
                    case FBT_INT: return readInt(bb, end, parentWidth);
                    case FBT_UINT:
                    case FBT_BOOL:
                        return readUInt(bb, end, parentWidth);
                    case FBT_INDIRECT_INT: return readInt(bb, indirect(bb, end, parentWidth), byteWidth);
                    case FBT_INDIRECT_UINT: return readUInt(bb, indirect(bb, end, parentWidth), byteWidth);
                    case FBT_NULL: return 0.0;
                    case FBT_STRING: return Double.parseDouble(asString());
                    case FBT_VECTOR: return asVector().size();
                    default:
                        // Convert strings and other things to float.
                        return 0;
                }
        }

        /**
         * Returns element as a {@link Key}
         * @return key or {@link Key#empty()} if element is not a key
         */
        public Key asKey() {
            if (isKey()) {
                return new Key(bb, indirect(bb, end, parentWidth), byteWidth);
            } else {
                return Key.empty();
            }
        }

        /**
         * Returns element as a `String`
         * @return element as `String` or empty `String` if fail
         */
        public String asString() {
            if (isString()) {
                int start = indirect(bb, end, byteWidth);
                int size = readInt(bb, start - byteWidth, byteWidth);
                return Utf8.getDefault().decodeUtf8(bb, start, size);
            }
            else if (isKey()){
                int start = indirect(bb, end, byteWidth);
                for (int i = start; ; i++) {
                    if (bb.get(i) == 0) {
                        return Utf8.getDefault().decodeUtf8(bb, start, i - start);
                    }
                }
            } else {
                return "";
            }
        }

        /**
         * Returns element as a {@link Map}
         * @return element as {@link Map} or empty {@link Map} if fail
         */
        public Map asMap() {
            if (isMap()) {
                return new Map(bb, indirect(bb, end, parentWidth), byteWidth);
            } else {
                return Map.empty();
            }
        }

        /**
         * Returns element as a {@link Vector}
         * @return element as {@link Vector} or empty {@link Vector} if fail
         */
        public Vector asVector() {
            if (isVector()) {
                return new Vector(bb, indirect(bb, end, parentWidth), byteWidth);
            } else if (FlexBuffers.isTypedVector(type)) {
                return new TypedVector(bb, indirect(bb, end, parentWidth), byteWidth, FlexBuffers.toTypedVectorElementType(type));
            } else {
                return Vector.empty();
            }
        }

        /**
         * Returns element as a {@link Blob}
         * @return element as {@link Blob} or empty {@link Blob} if fail
         */
        public Blob asBlob() {
            if (isBlob() || isString()) {
                return new Blob(bb, indirect(bb, end, parentWidth), byteWidth);
            } else {
                return Blob.empty();
            }
        }

        /**
         * Returns element as a boolean
         * <p>If element type is not boolean, it will be casted to integer and compared against 0</p>
         * @return element as boolean
         */
        public boolean asBoolean() {
            if (isBoolean()) {
                return bb.get(end) != 0;
            }
            return asUInt() != 0;
        }

        /**
         * Returns text representation of the element (JSON)
         * @return String containing text representation of the element
         */
        @Override
        public String toString() {
            return toString(new StringBuilder(128)).toString();
        }

        /**
         * Appends a text(JSON) representation to a `StringBuilder`
         */
        StringBuilder toString(StringBuilder sb) {
            //TODO: Original C++ implementation escape strings.
            // probably we should do it as well.
            switch (type) {
                case FBT_NULL:
                    return sb.append("null");
                case FBT_INT:
                case FBT_INDIRECT_INT:
                    return sb.append(asLong());
                case FBT_UINT:
                case FBT_INDIRECT_UINT:
                    return sb.append(asUInt());
                case FBT_INDIRECT_FLOAT:
                case FBT_FLOAT:
                    return sb.append(asFloat());
                case FBT_KEY:
                    return asKey().toString(sb.append('"')).append('"');
                case FBT_STRING:
                    return sb.append('"').append(asString()).append('"');
                case FBT_MAP:
                    return asMap().toString(sb);
                case FBT_VECTOR:
                    return asVector().toString(sb);
                case FBT_BLOB:
                    return asBlob().toString(sb);
                case FBT_BOOL:
                    return sb.append(asBoolean());
                case FBT_VECTOR_INT:
                case FBT_VECTOR_UINT:
                case FBT_VECTOR_FLOAT:
                case FBT_VECTOR_KEY:
                case FBT_VECTOR_STRING:
                case FBT_VECTOR_BOOL:
                    return sb.append(asVector());
                case FBT_VECTOR_INT2:
                case FBT_VECTOR_UINT2:
                case FBT_VECTOR_FLOAT2:
                case FBT_VECTOR_INT3:
                case FBT_VECTOR_UINT3:
                case FBT_VECTOR_FLOAT3:
                case FBT_VECTOR_INT4:
                case FBT_VECTOR_UINT4:
                case FBT_VECTOR_FLOAT4:

                    throw new FlexBufferException("not_implemented:" + type);
                default:
                    return sb;
            }
        }
    }

    /**
     * Base class of all types below.
     * Points into the data buffer and allows access to one type.
     */
    private static abstract class Object {
        ByteBuffer bb;
        int end;
        int byteWidth;

        Object(ByteBuffer buff, int end, int byteWidth) {
            this.bb = buff;
            this.end = end;
            this.byteWidth = byteWidth;
        }

        @Override
        public String toString() {
            return toString(new StringBuilder(128)).toString();
        }

        public abstract StringBuilder toString(StringBuilder sb);
    }

    // Stores size in `byte_width_` bytes before end position.
    private static abstract class Sized extends Object {
        Sized(ByteBuffer buff, int end, int byteWidth) {
            super(buff, end, byteWidth);
        }

        public int size() {
            return readInt(bb, end - byteWidth, byteWidth);
        }
    }

    /**
     * Represents a array of bytes element in the buffer
     *
     * <p>It can be converted to `ByteBuffer` using {@link data()},
     * copied into a byte[] using {@link getBytes()} or
     * have individual bytes accessed individually using {@link get(int)}</p>
     */
    public static class Blob extends Sized {
        static final Blob EMPTY = new Blob(EMPTY_BB, 0, 1);

        Blob(ByteBuffer buff, int end, int byteWidth) {
            super(buff, end, byteWidth);
        }

        /** Return an empty {@link Blob} */
        public static Blob empty() {
            return EMPTY;
        }

        /**
         * Return {@link Blob} as `ByteBuffer`
         * @return blob as `ByteBuffer`
         */
        public ByteBuffer data() {
            ByteBuffer dup = bb.duplicate();
            dup.position(end);
            dup.limit(end + size());
            return dup.asReadOnlyBuffer().slice();
        }

        /**
         * Copy blob into a byte[]
         * @return blob as a byte[]
         */
        public byte[] getBytes() {
            int size = size();
            byte[] result = new byte[size];
            for (int i = 0; i < size; i++) {
                result[i] = bb.get(end + i);
            }
            return result;
        }

        /**
         * Return individual byte at a given position
         * @param pos position of the byte to be read
         */
        public byte get(int pos) {
            assert pos >=0 && pos <= size();
            return bb.get(end + pos);
        }

        /**
         * Returns a text(JSON) representation of the {@link Blob}
         */
        @Override
        public String toString() {
            return Utf8.getDefault().decodeUtf8(bb, end, size());
        }

        /**
         * Append a text(JSON) representation of the {@link Blob} into a `StringBuilder`
         */
        @Override
        public StringBuilder toString(StringBuilder sb) {
            sb.append('"');
            sb.append(Utf8.getDefault().decodeUtf8(bb, end, size()));
            return sb.append('"');
        }
    }

    /**
     * Represents a key element in the buffer. Keys are
     * used to reference objects in a {@link Map}
     */
    public static class Key extends Object {

        private static final Key EMPTY = new Key(EMPTY_BB, 0, 0);

        Key(ByteBuffer buff, int end, int byteWidth) {
            super(buff, end, byteWidth);
        }

        /**
         * Return an empty {@link Key}
         * @return empty {@link Key}
         * */
        public static Key empty() {
            return Key.EMPTY;
        }

        /**
         * Appends a text(JSON) representation to a `StringBuilder`
         */
        @Override
        public StringBuilder toString(StringBuilder sb) {
            int size;
            for (int i = end; ; i++) {
                if (bb.get(i) == 0) {
                    size = i - end;
                    break;
                }
            }
            sb.append(Utf8.getDefault().decodeUtf8(bb, end, size));
            return sb;
        }

        int compareTo(byte[] other) {
            int ia = end;
            int io = 0;
            byte c1, c2;
            do {
                c1 = bb.get(ia);
                c2 = other[io];
                if (c1 == '\0')
                    return c1 - c2;
                ia++;
                io++;
                if (io == other.length) {
                    // in our buffer we have an additional \0 byte
                    // but this does not exist in regular Java strings, so we return now
                    return c1 - c2;
                }
            }
            while (c1 == c2);
            return c1 - c2;
        }

        /**
         *  Compare keys
         *  @param obj other key to compare
         *  @return true if keys are the same
         */
        @Override
        public boolean equals(java.lang.Object obj) {
            if (!(obj instanceof Key))
                return false;

            return ((Key) obj).end == end && ((Key) obj).byteWidth == byteWidth;
        }
    }

    /**
     * Map object representing a set of key-value pairs.
     */
    public static class Map extends Vector {
        private static final Map EMPTY_MAP = new Map(EMPTY_BB, 0, 0);

        Map(ByteBuffer bb, int end, int byteWidth) {
            super(bb, end, byteWidth);
        }

        /**
         * Returns an empty {@link Map}
         * @return an empty {@link Map}
         */
        public static Map empty() {
            return EMPTY_MAP;
        }

        /**
         * @param key access key to element on map
         * @return reference to value in map
         */
        public Reference get(String key) {
            return get(key.getBytes(StandardCharsets.UTF_8));
        }

        /**
         * @param key access key to element on map. Keys are assumed to be encoded in UTF-8
         * @return reference to value in map
         */
        public Reference get(byte[] key) {
            KeyVector keys = keys();
            int size = keys.size();
            int index = binarySearch(keys, key);
            if (index >= 0 && index < size) {
                return get(index);
            }
            return Reference.NULL_REFERENCE;
        }

        /**
         * Get a vector or keys in the map
         *
         * @return vector of keys
         */
        public KeyVector keys() {
            final int num_prefixed_fields = 3;
            int keysOffset = end - (byteWidth * num_prefixed_fields);
            return new KeyVector(new TypedVector(bb,
                    indirect(bb, keysOffset, byteWidth),
                    readInt(bb, keysOffset + byteWidth, byteWidth),
                    FBT_KEY));
        }

        /**
         * @return {@code Vector} of values from map
         */
        public Vector values() {
            return new Vector(bb, end, byteWidth);
        }

        /**
         * Writes text (json) representation of map in a {@code StringBuilder}.
         *
         * @param builder {@code StringBuilder} to be appended to
         * @return Same {@code StringBuilder} with appended text
         */
        public StringBuilder toString(StringBuilder builder) {
            builder.append("{ ");
            KeyVector keys = keys();
            int size = size();
            Vector vals = values();
            for (int i = 0; i < size; i++) {
                builder.append('"')
                        .append(keys.get(i).toString())
                        .append("\" : ");
                builder.append(vals.get(i).toString());
                if (i != size - 1)
                    builder.append(", ");
            }
            builder.append(" }");
            return builder;
        }

        // Performs a binary search on a key vector and return index of the key in key vector
        private int binarySearch(KeyVector keys, byte[] searchedKey) {
            int low = 0;
            int high = keys.size() - 1;

            while (low <= high) {
                int mid = (low + high) >>> 1;
                Key k = keys.get(mid);
                int cmp = k.compareTo(searchedKey);
                if (cmp < 0)
                    low = mid + 1;
                else if (cmp > 0)
                    high = mid - 1;
                else
                    return mid; // key found
            }
            return -(low + 1);  // key not found
        }
    }

    /**
     * Object that represents a set of elements in the buffer
     */
    public static class Vector extends Sized {

        private static final Vector EMPTY_VECTOR = new Vector(ByteBuffer.allocate(0), 1, 1);

        Vector(ByteBuffer bb, int end, int byteWidth) {
            super(bb, end, byteWidth);
        }

        /**
         * Returns an empty {@link Map}
         * @return an empty {@link Map}
         */
        public static Vector empty() {
            return EMPTY_VECTOR;
        }

        /**
         * Checks if the vector is empty
         * @return true if vector is empty
         */
        public boolean isEmpty() {
            return this == EMPTY_VECTOR;
        }

        /**
         * Appends a text(JSON) representation to a `StringBuilder`
         */
        @Override
        public StringBuilder toString(StringBuilder sb) {
            sb.append("[ ");
            int size = size();
            for (int i = 0; i < size; i++) {
                get(i).toString(sb);
                if (i != size - 1) {
                    sb.append(", ");
                }
            }
            sb.append(" ]");
            return sb;
        }

        /**
         * Get a element in a vector by index
         *
         * @param index position of the element
         * @return {@code Reference} to the element
         */
        public Reference get(int index) {
            long len = size();
            if (index >= len) {
                return Reference.NULL_REFERENCE;
            }
            int packedType = byteToUnsignedInt(bb.get((int) (end + (len * byteWidth) + index)));
            int obj_end = end + index * byteWidth;
            return new Reference(bb, obj_end, byteWidth, packedType);
        }
    }

    /**
     * Object that represents a set of elements with the same type
     */
    public static class TypedVector extends Vector {

        private static final TypedVector EMPTY_VECTOR = new TypedVector(EMPTY_BB, 0, 1, FBT_INT);

        private final int elemType;

        TypedVector(ByteBuffer bb, int end, int byteWidth, int elemType) {
            super(bb, end, byteWidth);
            this.elemType = elemType;
        }

        public static TypedVector empty() {
            return EMPTY_VECTOR;
        }

        /**
         * Returns whether the vector is empty
         *
         * @return true if empty
         */
        public boolean isEmptyVector() {
            return this == EMPTY_VECTOR;
        }

        /**
         * Return element type for all elements in the vector
         *
         * @return element type
         */
        public int getElemType() {
            return elemType;
        }

        /**
         * Get reference to an object in the {@code Vector}
         *
         * @param pos position of the object in {@code Vector}
         * @return reference to element
         */
        @Override
        public Reference get(int pos) {
            int len = size();
            if (pos >= len) return Reference.NULL_REFERENCE;
            int childPos = end + pos * byteWidth;
            return new Reference(bb, childPos, byteWidth, 1, elemType);
        }
    }

    /**
     * Represent a vector of keys in a map
     */
    public static class KeyVector {

        private final TypedVector vec;

        KeyVector(TypedVector vec) {
            this.vec = vec;
        }

        /**
         * Return key
         *
         * @param pos position of the key in key vector
         * @return key
         */
        public Key get(int pos) {
            int len = size();
            if (pos >= len) return Key.EMPTY;
            int childPos = vec.end + pos * vec.byteWidth;
            return new Key(vec.bb, indirect(vec.bb, childPos, vec.byteWidth), 1);
        }

        /**
         * Returns size of key vector
         *
         * @return size
         */
        public int size() {
            return vec.size();
        }

        /**
         * Returns a text(JSON) representation
         */
        public String toString() {
            StringBuilder b = new StringBuilder();
            b.append('[');
            for (int i = 0; i < vec.size(); i++) {
                vec.get(i).toString(b);
                if (i != vec.size() - 1) {
                    b.append(", ");
                }
            }
            return b.append("]").toString();
        }
    }

    public static class FlexBufferException extends RuntimeException {
        FlexBufferException(String msg) {
            super(msg);
        }
    }

    static class Unsigned {

        static int byteToUnsignedInt(byte x) {
            return ((int) x) & 0xff;
        }

        static int shortToUnsignedInt(short x) {
            return ((int) x) & 0xffff;
        }

        static long intToUnsignedLong(int x) {
            return ((long) x) & 0xffffffffL;
        }
    }
}
/// @}
