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

import static com.google.flatbuffers.Constants.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;

/// @cond FLATBUFFERS_INTERNAL

/**
 * All tables in the generated code derive from this class, and add their own accessors.
 */
public class Table {
  public final static ThreadLocal<Charset> UTF8_CHARSET = new ThreadLocal<Charset>() {
    @Override
    protected Charset initialValue() {
      return Charset.forName("UTF-8");
    }
  };
  /** Used to hold the position of the `bb` buffer. */
  protected int bb_pos;
  /** The underlying ByteBuffer to hold the data of the Table. */
  protected ByteBuffer bb;
  /** Used to hold the vtable position. */
  private int vtable_start;
  /** Used to hold the vtable size. */
  private int vtable_size;
  Utf8 utf8 = Utf8.getDefault();

  /**
   * Get the underlying ByteBuffer.
   *
   * @return Returns the Table's ByteBuffer.
   */
  public ByteBuffer getByteBuffer() { return bb; }

  /**
   * Look up a field in the vtable.
   *
   * @param vtable_offset An `int` offset to the vtable in the Table's ByteBuffer.
   * @return Returns an offset into the object, or `0` if the field is not present.
   */
  protected int __offset(int vtable_offset) {
    return vtable_offset < vtable_size ? bb.getShort(vtable_start + vtable_offset) : 0;
  }

  protected static int __offset(int vtable_offset, int offset, ByteBuffer bb) {
    int vtable = bb.capacity() - offset;
    return bb.getShort(vtable + vtable_offset - bb.getInt(vtable)) + vtable;
  }

  /**
   * Retrieve a relative offset.
   *
   * @param offset An `int` index into the Table's ByteBuffer containing the relative offset.
   * @return Returns the relative offset stored at `offset`.
   */
  protected int __indirect(int offset) {
    return offset + bb.getInt(offset);
  }

  protected static int __indirect(int offset, ByteBuffer bb) {
    return offset + bb.getInt(offset);
  }

  /**
   * Create a Java `String` from UTF-8 data stored inside the FlatBuffer.
   *
   * This allocates a new string and converts to wide chars upon each access,
   * which is not very efficient. Instead, each FlatBuffer string also comes with an
   * accessor based on __vector_as_bytebuffer below, which is much more efficient,
   * assuming your Java program can handle UTF-8 data directly.
   *
   * @param offset An `int` index into the Table's ByteBuffer.
   * @return Returns a `String` from the data stored inside the FlatBuffer at `offset`.
   */
  protected String __string(int offset) {
    offset += bb.getInt(offset);
    int length = bb.getInt(offset);
    return utf8.decodeUtf8(bb, offset + SIZEOF_INT, length);
  }

  /**
   * Get the length of a vector.
   *
   * @param offset An `int` index into the Table's ByteBuffer.
   * @return Returns the length of the vector whose offset is stored at `offset`.
   */
  protected int __vector_len(int offset) {
    offset += bb_pos;
    offset += bb.getInt(offset);
    return bb.getInt(offset);
  }

  /**
   * Get the start data of a vector.
   *
   * @param offset An `int` index into the Table's ByteBuffer.
   * @return Returns the start of the vector data whose offset is stored at `offset`.
   */
  protected int __vector(int offset) {
    offset += bb_pos;
    return offset + bb.getInt(offset) + SIZEOF_INT;  // data starts after the length
  }

  /**
   * Get a whole vector as a ByteBuffer.
   *
   * This is efficient, since it only allocates a new {@link ByteBuffer} object,
   * but does not actually copy the data, it still refers to the same bytes
   * as the original ByteBuffer. Also useful with nested FlatBuffers, etc.
   *
   * @param vector_offset The position of the vector in the byte buffer
   * @param elem_size The size of each element in the array
   * @return The {@link ByteBuffer} for the array
   */
  protected ByteBuffer __vector_as_bytebuffer(int vector_offset, int elem_size) {
    int o = __offset(vector_offset);
    if (o == 0) return null;
    ByteBuffer bb = this.bb.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    int vectorstart = __vector(o);
    bb.position(vectorstart);
    bb.limit(vectorstart + __vector_len(o) * elem_size);
    return bb;
  }

  /**
   * Initialize vector as a ByteBuffer.
   *
   * This is more efficient than using duplicate, since it doesn't copy the data
   * nor allocattes a new {@link ByteBuffer}, creating no garbage to be collected.
   *
   * @param bb The {@link ByteBuffer} for the array
   * @param vector_offset The position of the vector in the byte buffer
   * @param elem_size The size of each element in the array
   * @return The {@link ByteBuffer} for the array
   */
  protected ByteBuffer __vector_in_bytebuffer(ByteBuffer bb, int vector_offset, int elem_size) {
    int o = this.__offset(vector_offset);
    if (o == 0) return null;
    int vectorstart = __vector(o);
    bb.rewind();
    bb.limit(vectorstart + __vector_len(o) * elem_size);
    bb.position(vectorstart);
    return bb;
  }

  /**
   * Initialize any Table-derived type to point to the union at the given `offset`.
   *
   * @param t A `Table`-derived type that should point to the union at `offset`.
   * @param offset An `int` index into the Table's ByteBuffer.
   * @return Returns the Table that points to the union at `offset`.
   */
  protected Table __union(Table t, int offset) {
    offset += bb_pos;
    t.bb_pos = offset + bb.getInt(offset);
    t.bb = bb;
    t.vtable_start = t.bb_pos - bb.getInt(t.bb_pos);
    t.vtable_size = bb.getShort(t.vtable_start);
    return t;
  }

  /**
   * Check if a {@link ByteBuffer} contains a file identifier.
   *
   * @param bb A {@code ByteBuffer} to check if it contains the identifier
   * `ident`.
   * @param ident A `String` identifier of the FlatBuffer file.
   * @return True if the buffer contains the file identifier
   */
  protected static boolean __has_identifier(ByteBuffer bb, String ident) {
    if (ident.length() != FILE_IDENTIFIER_LENGTH)
        throw new AssertionError("FlatBuffers: file identifier must be length " +
                                 FILE_IDENTIFIER_LENGTH);
    for (int i = 0; i < FILE_IDENTIFIER_LENGTH; i++) {
      if (ident.charAt(i) != (char)bb.get(bb.position() + SIZEOF_INT + i)) return false;
    }
    return true;
  }

  /**
   * Sort tables by the key.
   *
   * @param offsets An 'int' indexes of the tables into the bb.
   * @param bb A {@code ByteBuffer} to get the tables.
   */
  protected void sortTables(int[] offsets, final ByteBuffer bb) {
    Integer[] off = new Integer[offsets.length];
    for (int i = 0; i < offsets.length; i++) off[i] = offsets[i];
    java.util.Arrays.sort(off, new java.util.Comparator<Integer>() {
      public int compare(Integer o1, Integer o2) {
        return keysCompare(o1, o2, bb);
      }
    });
    for (int i = 0; i < offsets.length; i++) offsets[i] = off[i];
  }

  /**
   * Compare two tables by the key.
   *
   * @param o1 An 'Integer' index of the first key into the bb.
   * @param o2 An 'Integer' index of the second key into the bb.
   * @param bb A {@code ByteBuffer} to get the keys.
   */
  protected int keysCompare(Integer o1, Integer o2, ByteBuffer bb) { return 0; }

  /**
   * Compare two strings in the buffer.
   *
   * @param offset_1 An 'int' index of the first string into the bb.
   * @param offset_2 An 'int' index of the second string into the bb.
   * @param bb A {@code ByteBuffer} to get the strings.
   */
  protected static int compareStrings(int offset_1, int offset_2, ByteBuffer bb) {
    offset_1 += bb.getInt(offset_1);
    offset_2 += bb.getInt(offset_2);
    int len_1 = bb.getInt(offset_1);
    int len_2 = bb.getInt(offset_2);
    int startPos_1 = offset_1 + SIZEOF_INT;
    int startPos_2 = offset_2 + SIZEOF_INT;
    int len = Math.min(len_1, len_2);
    for(int i = 0; i < len; i++) {
      if (bb.get(i + startPos_1) != bb.get(i + startPos_2))
        return bb.get(i + startPos_1) - bb.get(i + startPos_2);
    }
    return len_1 - len_2;
  }

  /**
   * Compare string from the buffer with the 'String' object.
   *
   * @param offset_1 An 'int' index of the first string into the bb.
   * @param key Second string as a byte array.
   * @param bb A {@code ByteBuffer} to get the first string.
   */
  protected static int compareStrings(int offset_1, byte[] key, ByteBuffer bb) {
    offset_1 += bb.getInt(offset_1);
    int len_1 = bb.getInt(offset_1);
    int len_2 = key.length;
    int startPos_1 = offset_1 + Constants.SIZEOF_INT;
    int len = Math.min(len_1, len_2);
    for (int i = 0; i < len; i++) {
      if (bb.get(i + startPos_1) != key[i])
        return bb.get(i + startPos_1) - key[i];
    }
    return len_1 - len_2;
  }

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
      vtable_start = bb_pos - bb.getInt(bb_pos);
      vtable_size = bb.getShort(vtable_start);
    } else {
      bb_pos = 0;
      vtable_start = 0;
      vtable_size = 0;
    }
  }

  /**
   * Resets the internal state with a null {@code ByteBuffer} and a zero position.
   *
   * This method exists primarily to allow recycling Table instances without risking memory leaks
   * due to {@code ByteBuffer} references. The instance will be unusable until it is assigned
   * again to a {@code ByteBuffer}.
   */
  public void __reset() {
    __reset(0, null);
  }
}

/// @endcond
