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

import java.io.IOException;
import java.io.InputStream;
import java.nio.*;
import java.util.Arrays;

/// @file
/// @addtogroup flatbuffers_java_api
/// @{

/**
 * Class that helps you build a FlatBuffer.  See the section
 * "Use in Java/C#" in the main FlatBuffers documentation.
 */
public class FlatBufferBuilder {
    /// @cond FLATBUFFERS_INTERNAL
    ByteBuffer bb;                  // Where we construct the FlatBuffer.
    int space;                      // Remaining space in the ByteBuffer.
    int minalign = 1;               // Minimum alignment encountered so far.
    int[] vtable = null;            // The vtable for the current table.
    int vtable_in_use = 0;          // The amount of fields we're actually using.
    boolean nested = false;         // Whether we are currently serializing a table.
    boolean finished = false;       // Whether the buffer is finished.
    int object_start;               // Starting offset of the current struct/table.
    int[] vtables = new int[16];    // List of offsets of all vtables.
    int num_vtables = 0;            // Number of entries in `vtables` in use.
    int vector_num_elems = 0;       // For the current vector being built.
    boolean force_defaults = false; // False omits default values from the serialized data.
    ByteBufferFactory bb_factory;   // Factory for allocating the internal buffer
    final Utf8 utf8;                // UTF-8 encoder to use
    /// @endcond

    /**
     * Start with a buffer of size `initial_size`, then grow as required.
     *
     * @param initial_size The initial size of the internal buffer to use.
     * @param bb_factory The factory to be used for allocating the internal buffer
     */
    public FlatBufferBuilder(int initial_size, ByteBufferFactory bb_factory) {
        this(initial_size, bb_factory, null, Utf8.getDefault());
    }

    /**
     * Start with a buffer of size `initial_size`, then grow as required.
     *
     * @param initial_size The initial size of the internal buffer to use.
     * @param bb_factory The factory to be used for allocating the internal buffer
     * @param existing_bb The byte buffer to reuse.
     * @param utf8 The Utf8 codec
     */
    public FlatBufferBuilder(int initial_size, ByteBufferFactory bb_factory,
                             ByteBuffer existing_bb, Utf8 utf8) {
        if (initial_size <= 0) {
          initial_size = 1;
        }
        this.bb_factory = bb_factory;
        if (existing_bb != null) {
          bb = existing_bb;
          bb.clear();
          bb.order(ByteOrder.LITTLE_ENDIAN);
        } else {
          bb = bb_factory.newByteBuffer(initial_size);
        }
        this.utf8 = utf8;
        space = bb.capacity();
    }

   /**
    * Start with a buffer of size `initial_size`, then grow as required.
    *
    * @param initial_size The initial size of the internal buffer to use.
    */
    public FlatBufferBuilder(int initial_size) {
        this(initial_size, HeapByteBufferFactory.INSTANCE, null, Utf8.getDefault());
    }

    /**
     * Start with a buffer of 1KiB, then grow as required.
     */
    public FlatBufferBuilder() {
        this(1024);
    }

    /**
     * Alternative constructor allowing reuse of {@link ByteBuffer}s.  The builder
     * can still grow the buffer as necessary.  User classes should make sure
     * to call {@link #dataBuffer()} to obtain the resulting encoded message.
     *
     * @param existing_bb The byte buffer to reuse.
     * @param bb_factory The factory to be used for allocating a new internal buffer if
     *                   the existing buffer needs to grow
     */
    public FlatBufferBuilder(ByteBuffer existing_bb, ByteBufferFactory bb_factory) {
        this(existing_bb.capacity(), bb_factory, existing_bb, Utf8.getDefault());
    }

    /**
     * Alternative constructor allowing reuse of {@link ByteBuffer}s.  The builder
     * can still grow the buffer as necessary.  User classes should make sure
     * to call {@link #dataBuffer()} to obtain the resulting encoded message.
     *
     * @param existing_bb The byte buffer to reuse.
     */
    public FlatBufferBuilder(ByteBuffer existing_bb) {
        this(existing_bb, new HeapByteBufferFactory());
    }

    /**
     * Alternative initializer that allows reusing this object on an existing
     * `ByteBuffer`. This method resets the builder's internal state, but keeps
     * objects that have been allocated for temporary storage.
     *
     * @param existing_bb The byte buffer to reuse.
     * @param bb_factory The factory to be used for allocating a new internal buffer if
     *                   the existing buffer needs to grow
     * @return Returns `this`.
     */
    public FlatBufferBuilder init(ByteBuffer existing_bb, ByteBufferFactory bb_factory){
        this.bb_factory = bb_factory;
        bb = existing_bb;
        bb.clear();
        bb.order(ByteOrder.LITTLE_ENDIAN);
        minalign = 1;
        space = bb.capacity();
        vtable_in_use = 0;
        nested = false;
        finished = false;
        object_start = 0;
        num_vtables = 0;
        vector_num_elems = 0;
        return this;
    }

    /**
     * An interface that provides a user of the FlatBufferBuilder class the ability to specify
     * the method in which the internal buffer gets allocated. This allows for alternatives
     * to the default behavior, which is to allocate memory for a new byte-array
     * backed `ByteBuffer` array inside the JVM.
     *
     * The FlatBufferBuilder class contains the HeapByteBufferFactory class to
     * preserve the default behavior in the event that the user does not provide
     * their own implementation of this interface.
     */
    public static abstract class ByteBufferFactory {
        /**
         * Create a `ByteBuffer` with a given capacity.
         * The returned ByteBuf must have a ByteOrder.LITTLE_ENDIAN ByteOrder.
         *
         * @param capacity The size of the `ByteBuffer` to allocate.
         * @return Returns the new `ByteBuffer` that was allocated.
         */
        public abstract ByteBuffer newByteBuffer(int capacity);

        /**
         * Release a ByteBuffer. Current {@link FlatBufferBuilder}
         * released any reference to it, so it is safe to dispose the buffer
         * or return it to a pool.
         * It is not guaranteed that the buffer has been created
         * with {@link #newByteBuffer(int) }.
         *
         * @param bb the buffer to release
         */
        public void releaseByteBuffer(ByteBuffer bb) {
        }
    }

    /**
     * An implementation of the ByteBufferFactory interface that is used when
     * one is not provided by the user.
     *
     * Allocate memory for a new byte-array backed `ByteBuffer` array inside the JVM.
     */
    public static final class HeapByteBufferFactory extends ByteBufferFactory {

        public static final HeapByteBufferFactory INSTANCE = new HeapByteBufferFactory();

        @Override
        public ByteBuffer newByteBuffer(int capacity) {
            return ByteBuffer.allocate(capacity).order(ByteOrder.LITTLE_ENDIAN);
        }
    }

   /**
   * Helper function to test if a field is present in the table
   *
   * @param table Flatbuffer table
   * @param offset virtual table offset
   * @return true if the filed is present
   */
   public static boolean isFieldPresent(Table table, int offset) {
     return table.__offset(offset) != 0;
   }

    /**
     * Reset the FlatBufferBuilder by purging all data that it holds.
     */
    public void clear(){
        space = bb.capacity();
        bb.clear();
        minalign = 1;
        while(vtable_in_use > 0) vtable[--vtable_in_use] = 0;
        vtable_in_use = 0;
        nested = false;
        finished = false;
        object_start = 0;
        num_vtables = 0;
        vector_num_elems = 0;
    }

    /**
     * Doubles the size of the backing {@link ByteBuffer} and copies the old data towards the
     * end of the new buffer (since we build the buffer backwards).
     *
     * @param bb The current buffer with the existing data.
     * @param bb_factory The factory to be used for allocating the new internal buffer
     * @return A new byte buffer with the old data copied copied to it.  The data is
     * located at the end of the buffer.
     */
    static ByteBuffer growByteBuffer(ByteBuffer bb, ByteBufferFactory bb_factory) {
        int old_buf_size = bb.capacity();
        if ((old_buf_size & 0xC0000000) != 0)  // Ensure we don't grow beyond what fits in an int.
            throw new AssertionError("FlatBuffers: cannot grow buffer beyond 2 gigabytes.");
        int new_buf_size = old_buf_size == 0 ? 1 : old_buf_size << 1;
        bb.position(0);
        ByteBuffer nbb = bb_factory.newByteBuffer(new_buf_size);
        nbb.position(new_buf_size - old_buf_size);
        nbb.put(bb);
        return nbb;
    }

   /**
    * Offset relative to the end of the buffer.
    *
    * @return Offset relative to the end of the buffer.
    */
    public int offset() {
        return bb.capacity() - space;
    }

   /**
    * Add zero valued bytes to prepare a new entry to be added.
    *
    * @param byte_size Number of bytes to add.
    */
    public void pad(int byte_size) {
        for (int i = 0; i < byte_size; i++) bb.put(--space, (byte)0);
    }

   /**
    * Prepare to write an element of `size` after `additional_bytes`
    * have been written, e.g. if you write a string, you need to align such
    * the int length field is aligned to {@link com.google.flatbuffers.Constants#SIZEOF_INT}, and
    * the string data follows it directly.  If all you need to do is alignment, `additional_bytes`
    * will be 0.
    *
    * @param size This is the of the new element to write.
    * @param additional_bytes The padding size.
    */
    public void prep(int size, int additional_bytes) {
        // Track the biggest thing we've ever aligned to.
        if (size > minalign) minalign = size;
        // Find the amount of alignment needed such that `size` is properly
        // aligned after `additional_bytes`
        int align_size = ((~(bb.capacity() - space + additional_bytes)) + 1) & (size - 1);
        // Reallocate the buffer if needed.
        while (space < align_size + size + additional_bytes) {
            int old_buf_size = bb.capacity();
            ByteBuffer old = bb;
            bb = growByteBuffer(old, bb_factory);
            if (old != bb) {
                bb_factory.releaseByteBuffer(old);
            }
            space += bb.capacity() - old_buf_size;
        }
        pad(align_size);
    }

    /**
     * Add a `boolean` to the buffer, backwards from the current location. Doesn't align nor
     * check for space.
     *
     * @param x A `boolean` to put into the buffer.
     */
    public void putBoolean(boolean x) { bb.put      (space -= Constants.SIZEOF_BYTE, (byte)(x ? 1 : 0)); }

    /**
     * Add a `byte` to the buffer, backwards from the current location. Doesn't align nor
     * check for space.
     *
     * @param x A `byte` to put into the buffer.
     */
    public void putByte   (byte    x) { bb.put      (space -= Constants.SIZEOF_BYTE, x); }

    /**
     * Add a `short` to the buffer, backwards from the current location. Doesn't align nor
     * check for space.
     *
     * @param x A `short` to put into the buffer.
     */
    public void putShort  (short   x) { bb.putShort (space -= Constants.SIZEOF_SHORT, x); }

    /**
     * Add an `int` to the buffer, backwards from the current location. Doesn't align nor
     * check for space.
     *
     * @param x An `int` to put into the buffer.
     */
    public void putInt    (int     x) { bb.putInt   (space -= Constants.SIZEOF_INT, x); }

    /**
     * Add a `long` to the buffer, backwards from the current location. Doesn't align nor
     * check for space.
     *
     * @param x A `long` to put into the buffer.
     */
    public void putLong   (long    x) { bb.putLong  (space -= Constants.SIZEOF_LONG, x); }

    /**
     * Add a `float` to the buffer, backwards from the current location. Doesn't align nor
     * check for space.
     *
     * @param x A `float` to put into the buffer.
     */
    public void putFloat  (float   x) { bb.putFloat (space -= Constants.SIZEOF_FLOAT, x); }

    /**
     * Add a `double` to the buffer, backwards from the current location. Doesn't align nor
     * check for space.
     *
     * @param x A `double` to put into the buffer.
     */
    public void putDouble (double  x) { bb.putDouble(space -= Constants.SIZEOF_DOUBLE, x); }
    /// @endcond

    /**
     * Add a `boolean` to the buffer, properly aligned, and grows the buffer (if necessary).
     *
     * @param x A `boolean` to put into the buffer.
     */
    public void addBoolean(boolean x) { prep(Constants.SIZEOF_BYTE, 0); putBoolean(x); }

    /**
     * Add a `byte` to the buffer, properly aligned, and grows the buffer (if necessary).
     *
     * @param x A `byte` to put into the buffer.
     */
    public void addByte   (byte    x) { prep(Constants.SIZEOF_BYTE, 0); putByte   (x); }

    /**
     * Add a `short` to the buffer, properly aligned, and grows the buffer (if necessary).
     *
     * @param x A `short` to put into the buffer.
     */
    public void addShort  (short   x) { prep(Constants.SIZEOF_SHORT, 0); putShort  (x); }

    /**
     * Add an `int` to the buffer, properly aligned, and grows the buffer (if necessary).
     *
     * @param x An `int` to put into the buffer.
     */
    public void addInt    (int     x) { prep(Constants.SIZEOF_INT, 0); putInt    (x); }

    /**
     * Add a `long` to the buffer, properly aligned, and grows the buffer (if necessary).
     *
     * @param x A `long` to put into the buffer.
     */
    public void addLong   (long    x) { prep(Constants.SIZEOF_LONG, 0); putLong   (x); }

    /**
     * Add a `float` to the buffer, properly aligned, and grows the buffer (if necessary).
     *
     * @param x A `float` to put into the buffer.
     */
    public void addFloat  (float   x) { prep(Constants.SIZEOF_FLOAT, 0); putFloat  (x); }

    /**
     * Add a `double` to the buffer, properly aligned, and grows the buffer (if necessary).
     *
     * @param x A `double` to put into the buffer.
     */
    public void addDouble (double  x) { prep(Constants.SIZEOF_DOUBLE, 0); putDouble (x); }

   /**
    * Adds on offset, relative to where it will be written.
    *
    * @param off The offset to add.
    */
    public void addOffset(int off) {
        prep(SIZEOF_INT, 0);  // Ensure alignment is already done.
        assert off <= offset();
        off = offset() - off + SIZEOF_INT;
        putInt(off);
    }

   /// @cond FLATBUFFERS_INTERNAL
   /**
    * Start a new array/vector of objects.  Users usually will not call
    * this directly.  The `FlatBuffers` compiler will create a start/end
    * method for vector types in generated code.
    * <p>
    * The expected sequence of calls is:
    * <ol>
    * <li>Start the array using this method.</li>
    * <li>Call {@link #addOffset(int)} `num_elems` number of times to set
    * the offset of each element in the array.</li>
    * <li>Call {@link #endVector()} to retrieve the offset of the array.</li>
    * </ol>
    * <p>
    * For example, to create an array of strings, do:
    * <pre>{@code
    * // Need 10 strings
    * FlatBufferBuilder builder = new FlatBufferBuilder(existingBuffer);
    * int[] offsets = new int[10];
    *
    * for (int i = 0; i < 10; i++) {
    *   offsets[i] = fbb.createString(" " + i);
    * }
    *
    * // Have the strings in the buffer, but don't have a vector.
    * // Add a vector that references the newly created strings:
    * builder.startVector(4, offsets.length, 4);
    *
    * // Add each string to the newly created vector
    * // The strings are added in reverse order since the buffer
    * // is filled in back to front
    * for (int i = offsets.length - 1; i >= 0; i--) {
    *   builder.addOffset(offsets[i]);
    * }
    *
    * // Finish off the vector
    * int offsetOfTheVector = fbb.endVector();
    * }</pre>
    *
    * @param elem_size The size of each element in the array.
    * @param num_elems The number of elements in the array.
    * @param alignment The alignment of the array.
    */
    public void startVector(int elem_size, int num_elems, int alignment) {
        notNested();
        vector_num_elems = num_elems;
        prep(SIZEOF_INT, elem_size * num_elems);
        prep(alignment, elem_size * num_elems); // Just in case alignment > int.
        nested = true;
    }

   /**
    * Finish off the creation of an array and all its elements.  The array
    * must be created with {@link #startVector(int, int, int)}.
    *
    * @return The offset at which the newly created array starts.
    * @see #startVector(int, int, int)
    */
    public int endVector() {
        if (!nested)
            throw new AssertionError("FlatBuffers: endVector called without startVector");
        nested = false;
        putInt(vector_num_elems);
        return offset();
    }
    /// @endcond

    /**
     * Create a new array/vector and return a ByteBuffer to be filled later.
     * Call {@link #endVector} after this method to get an offset to the beginning
     * of vector.
     *
     * @param elem_size the size of each element in bytes.
     * @param num_elems number of elements in the vector.
     * @param alignment byte alignment.
     * @return ByteBuffer with position and limit set to the space allocated for the array.
     */
    public ByteBuffer createUnintializedVector(int elem_size, int num_elems, int alignment) {
        int length = elem_size * num_elems;
        startVector(elem_size, num_elems, alignment);

        bb.position(space -= length);

        // Slice and limit the copy vector to point to the 'array'
        ByteBuffer copy = bb.slice().order(ByteOrder.LITTLE_ENDIAN);
        copy.limit(length);
        return copy;
    }

   /**
     * Create a vector of tables.
     *
     * @param offsets Offsets of the tables.
     * @return Returns offset of the vector.
     */
    public int createVectorOfTables(int[] offsets) {
        notNested();
        startVector(Constants.SIZEOF_INT, offsets.length, Constants.SIZEOF_INT);
        for(int i = offsets.length - 1; i >= 0; i--) addOffset(offsets[i]);
        return endVector();
    }

    /**
     * Create a vector of sorted by the key tables.
     *
     * @param obj Instance of the table subclass.
     * @param offsets Offsets of the tables.
     * @return Returns offset of the sorted vector.
     */
    public <T extends Table> int createSortedVectorOfTables(T obj, int[] offsets) {
        obj.sortTables(offsets, bb);
        return createVectorOfTables(offsets);
    }

   /**
    * Encode the string `s` in the buffer using UTF-8.  If {@code s} is
    * already a {@link CharBuffer}, this method is allocation free.
    *
    * @param s The string to encode.
    * @return The offset in the buffer where the encoded string starts.
    */
    public int createString(CharSequence s) {
        int length = utf8.encodedLength(s);
        addByte((byte)0);
        startVector(1, length, 1);
        bb.position(space -= length);
        utf8.encodeUtf8(s, bb);
        return endVector();
    }

   /**
    * Create a string in the buffer from an already encoded UTF-8 string in a ByteBuffer.
    *
    * @param s An already encoded UTF-8 string as a `ByteBuffer`.
    * @return The offset in the buffer where the encoded string starts.
    */
    public int createString(ByteBuffer s) {
        int length = s.remaining();
        addByte((byte)0);
        startVector(1, length, 1);
        bb.position(space -= length);
        bb.put(s);
        return endVector();
    }

    /**
     * Create a byte array in the buffer.
     *
     * @param arr A source array with data
     * @return The offset in the buffer where the encoded array starts.
     */
    public int createByteVector(byte[] arr) {
        int length = arr.length;
        startVector(1, length, 1);
        bb.position(space -= length);
        bb.put(arr);
        return endVector();
    }

   /// @cond FLATBUFFERS_INTERNAL
   /**
    * Should not be accessing the final buffer before it is finished.
    */
    public void finished() {
        if (!finished)
            throw new AssertionError(
                "FlatBuffers: you can only access the serialized buffer after it has been" +
                " finished by FlatBufferBuilder.finish().");
    }

   /**
    * Should not be creating any other object, string or vector
    * while an object is being constructed.
    */
    public void notNested() {
        if (nested)
            throw new AssertionError("FlatBuffers: object serialization must not be nested.");
    }

   /**
    * Structures are always stored inline, they need to be created right
    * where they're used.  You'll get this assertion failure if you
    * created it elsewhere.
    *
    * @param obj The offset of the created object.
    */
    public void Nested(int obj) {
        if (obj != offset())
            throw new AssertionError("FlatBuffers: struct must be serialized inline.");
    }

   /**
    * Start encoding a new object in the buffer.  Users will not usually need to
    * call this directly. The `FlatBuffers` compiler will generate helper methods
    * that call this method internally.
    * <p>
    * For example, using the "Monster" code found on the "landing page". An
    * object of type `Monster` can be created using the following code:
    *
    * <pre>{@code
    * int testArrayOfString = Monster.createTestarrayofstringVector(fbb, new int[] {
    *   fbb.createString("test1"),
    *   fbb.createString("test2")
    * });
    *
    * Monster.startMonster(fbb);
    * Monster.addPos(fbb, Vec3.createVec3(fbb, 1.0f, 2.0f, 3.0f, 3.0,
    *   Color.Green, (short)5, (byte)6));
    * Monster.addHp(fbb, (short)80);
    * Monster.addName(fbb, str);
    * Monster.addInventory(fbb, inv);
    * Monster.addTestType(fbb, (byte)Any.Monster);
    * Monster.addTest(fbb, mon2);
    * Monster.addTest4(fbb, test4);
    * Monster.addTestarrayofstring(fbb, testArrayOfString);
    * int mon = Monster.endMonster(fbb);
    * }</pre>
    * <p>
    * Here:
    * <ul>
    * <li>The call to `Monster#startMonster(FlatBufferBuilder)` will call this
    * method with the right number of fields set.</li>
    * <li>`Monster#endMonster(FlatBufferBuilder)` will ensure {@link #endObject()} is called.</li>
    * </ul>
    * <p>
    * It's not recommended to call this method directly.  If it's called manually, you must ensure
    * to audit all calls to it whenever fields are added or removed from your schema.  This is
    * automatically done by the code generated by the `FlatBuffers` compiler.
    *
    * @param numfields The number of fields found in this object.
    */
    public void startTable(int numfields) {
        notNested();
        if (vtable == null || vtable.length < numfields) vtable = new int[numfields];
        vtable_in_use = numfields;
        Arrays.fill(vtable, 0, vtable_in_use, 0);
        nested = true;
        object_start = offset();
    }

    /**
     * Add a `boolean` to a table at `o` into its vtable, with value `x` and default `d`.
     *
     * @param o The index into the vtable.
     * @param x A `boolean` to put into the buffer, depending on how defaults are handled. If
     * `force_defaults` is `false`, compare `x` against the default value `d`. If `x` contains the
     * default value, it can be skipped.
     * @param d A `boolean` default value to compare against when `force_defaults` is `false`.
     */
    public void addBoolean(int o, boolean x, boolean d) { if(force_defaults || x != d) { addBoolean(x); slot(o); } }

    /**
     * Add a `byte` to a table at `o` into its vtable, with value `x` and default `d`.
     *
     * @param o The index into the vtable.
     * @param x A `byte` to put into the buffer, depending on how defaults are handled. If
     * `force_defaults` is `false`, compare `x` against the default value `d`. If `x` contains the
     * default value, it can be skipped.
     * @param d A `byte` default value to compare against when `force_defaults` is `false`.
     */
    public void addByte   (int o, byte    x, int     d) { if(force_defaults || x != d) { addByte   (x); slot(o); } }

    /**
     * Add a `short` to a table at `o` into its vtable, with value `x` and default `d`.
     *
     * @param o The index into the vtable.
     * @param x A `short` to put into the buffer, depending on how defaults are handled. If
     * `force_defaults` is `false`, compare `x` against the default value `d`. If `x` contains the
     * default value, it can be skipped.
     * @param d A `short` default value to compare against when `force_defaults` is `false`.
     */
    public void addShort  (int o, short   x, int     d) { if(force_defaults || x != d) { addShort  (x); slot(o); } }

    /**
     * Add an `int` to a table at `o` into its vtable, with value `x` and default `d`.
     *
     * @param o The index into the vtable.
     * @param x An `int` to put into the buffer, depending on how defaults are handled. If
     * `force_defaults` is `false`, compare `x` against the default value `d`. If `x` contains the
     * default value, it can be skipped.
     * @param d An `int` default value to compare against when `force_defaults` is `false`.
     */
    public void addInt    (int o, int     x, int     d) { if(force_defaults || x != d) { addInt    (x); slot(o); } }

    /**
     * Add a `long` to a table at `o` into its vtable, with value `x` and default `d`.
     *
     * @param o The index into the vtable.
     * @param x A `long` to put into the buffer, depending on how defaults are handled. If
     * `force_defaults` is `false`, compare `x` against the default value `d`. If `x` contains the
     * default value, it can be skipped.
     * @param d A `long` default value to compare against when `force_defaults` is `false`.
     */
    public void addLong   (int o, long    x, long    d) { if(force_defaults || x != d) { addLong   (x); slot(o); } }

    /**
     * Add a `float` to a table at `o` into its vtable, with value `x` and default `d`.
     *
     * @param o The index into the vtable.
     * @param x A `float` to put into the buffer, depending on how defaults are handled. If
     * `force_defaults` is `false`, compare `x` against the default value `d`. If `x` contains the
     * default value, it can be skipped.
     * @param d A `float` default value to compare against when `force_defaults` is `false`.
     */
    public void addFloat  (int o, float   x, double  d) { if(force_defaults || x != d) { addFloat  (x); slot(o); } }

    /**
     * Add a `double` to a table at `o` into its vtable, with value `x` and default `d`.
     *
     * @param o The index into the vtable.
     * @param x A `double` to put into the buffer, depending on how defaults are handled. If
     * `force_defaults` is `false`, compare `x` against the default value `d`. If `x` contains the
     * default value, it can be skipped.
     * @param d A `double` default value to compare against when `force_defaults` is `false`.
     */
    public void addDouble (int o, double  x, double  d) { if(force_defaults || x != d) { addDouble (x); slot(o); } }

    /**
     * Add an `offset` to a table at `o` into its vtable, with value `x` and default `d`.
     *
     * @param o The index into the vtable.
     * @param x An `offset` to put into the buffer, depending on how defaults are handled. If
     * `force_defaults` is `false`, compare `x` against the default value `d`. If `x` contains the
     * default value, it can be skipped.
     * @param d An `offset` default value to compare against when `force_defaults` is `false`.
     */
    public void addOffset (int o, int     x, int     d) { if(force_defaults || x != d) { addOffset (x); slot(o); } }

    /**
     * Add a struct to the table. Structs are stored inline, so nothing additional is being added.
     *
     * @param voffset The index into the vtable.
     * @param x The offset of the created struct.
     * @param d The default value is always `0`.
     */
    public void addStruct(int voffset, int x, int d) {
        if(x != d) {
            Nested(x);
            slot(voffset);
        }
    }

    /**
     * Set the current vtable at `voffset` to the current location in the buffer.
     *
     * @param voffset The index into the vtable to store the offset relative to the end of the
     * buffer.
     */
    public void slot(int voffset) {
        vtable[voffset] = offset();
    }

   /**
    * Finish off writing the object that is under construction.
    *
    * @return The offset to the object inside {@link #dataBuffer()}.
    * @see #startTable(int)
    */
    public int endTable() {
        if (vtable == null || !nested)
            throw new AssertionError("FlatBuffers: endTable called without startTable");
        addInt(0);
        int vtableloc = offset();
        // Write out the current vtable.
        int i = vtable_in_use - 1;
        // Trim trailing zeroes.
        for (; i >= 0 && vtable[i] == 0; i--) {}
        int trimmed_size = i + 1;
        for (; i >= 0 ; i--) {
            // Offset relative to the start of the table.
            short off = (short)(vtable[i] != 0 ? vtableloc - vtable[i] : 0);
            addShort(off);
        }

        final int standard_fields = 2; // The fields below:
        addShort((short)(vtableloc - object_start));
        addShort((short)((trimmed_size + standard_fields) * SIZEOF_SHORT));

        // Search for an existing vtable that matches the current one.
        int existing_vtable = 0;
        outer_loop:
        for (i = 0; i < num_vtables; i++) {
            int vt1 = bb.capacity() - vtables[i];
            int vt2 = space;
            short len = bb.getShort(vt1);
            if (len == bb.getShort(vt2)) {
                for (int j = SIZEOF_SHORT; j < len; j += SIZEOF_SHORT) {
                    if (bb.getShort(vt1 + j) != bb.getShort(vt2 + j)) {
                        continue outer_loop;
                    }
                }
                existing_vtable = vtables[i];
                break outer_loop;
            }
        }

        if (existing_vtable != 0) {
            // Found a match:
            // Remove the current vtable.
            space = bb.capacity() - vtableloc;
            // Point table to existing vtable.
            bb.putInt(space, existing_vtable - vtableloc);
        } else {
            // No match:
            // Add the location of the current vtable to the list of vtables.
            if (num_vtables == vtables.length) vtables = Arrays.copyOf(vtables, num_vtables * 2);
            vtables[num_vtables++] = offset();
            // Point table to current vtable.
            bb.putInt(bb.capacity() - vtableloc, offset() - vtableloc);
        }

        nested = false;
        return vtableloc;
    }

    /**
     * Checks that a required field has been set in a given table that has
     * just been constructed.
     *
     * @param table The offset to the start of the table from the `ByteBuffer` capacity.
     * @param field The offset to the field in the vtable.
     */
    public void required(int table, int field) {
        int table_start = bb.capacity() - table;
        int vtable_start = table_start - bb.getInt(table_start);
        boolean ok = bb.getShort(vtable_start + field) != 0;
        // If this fails, the caller will show what field needs to be set.
        if (!ok)
            throw new AssertionError("FlatBuffers: field " + field + " must be set");
    }
    /// @endcond

    /**
     * Finalize a buffer, pointing to the given `root_table`.
     *
     * @param root_table An offset to be added to the buffer.
     * @param size_prefix Whether to prefix the size to the buffer.
     */
    protected void finish(int root_table, boolean size_prefix) {
        prep(minalign, SIZEOF_INT + (size_prefix ? SIZEOF_INT : 0));
        addOffset(root_table);
        if (size_prefix) {
            addInt(bb.capacity() - space);
        }
        bb.position(space);
        finished = true;
    }

    /**
     * Finalize a buffer, pointing to the given `root_table`.
     *
     * @param root_table An offset to be added to the buffer.
     */
    public void finish(int root_table) {
        finish(root_table, false);
    }

    /**
     * Finalize a buffer, pointing to the given `root_table`, with the size prefixed.
     *
     * @param root_table An offset to be added to the buffer.
     */
    public void finishSizePrefixed(int root_table) {
        finish(root_table, true);
    }

    /**
     * Finalize a buffer, pointing to the given `root_table`.
     *
     * @param root_table An offset to be added to the buffer.
     * @param file_identifier A FlatBuffer file identifier to be added to the buffer before
     * `root_table`.
     * @param size_prefix Whether to prefix the size to the buffer.
     */
    protected void finish(int root_table, String file_identifier, boolean size_prefix) {
        prep(minalign, SIZEOF_INT + FILE_IDENTIFIER_LENGTH + (size_prefix ? SIZEOF_INT : 0));
        if (file_identifier.length() != FILE_IDENTIFIER_LENGTH)
            throw new AssertionError("FlatBuffers: file identifier must be length " +
                                     FILE_IDENTIFIER_LENGTH);
        for (int i = FILE_IDENTIFIER_LENGTH - 1; i >= 0; i--) {
            addByte((byte)file_identifier.charAt(i));
        }
        finish(root_table, size_prefix);
    }

    /**
     * Finalize a buffer, pointing to the given `root_table`.
     *
     * @param root_table An offset to be added to the buffer.
     * @param file_identifier A FlatBuffer file identifier to be added to the buffer before
     * `root_table`.
     */
    public void finish(int root_table, String file_identifier) {
        finish(root_table, file_identifier, false);
    }

    /**
     * Finalize a buffer, pointing to the given `root_table`, with the size prefixed.
     *
     * @param root_table An offset to be added to the buffer.
     * @param file_identifier A FlatBuffer file identifier to be added to the buffer before
     * `root_table`.
     */
    public void finishSizePrefixed(int root_table, String file_identifier) {
        finish(root_table, file_identifier, true);
    }

    /**
     * In order to save space, fields that are set to their default value
     * don't get serialized into the buffer. Forcing defaults provides a
     * way to manually disable this optimization.
     *
     * @param forceDefaults When set to `true`, always serializes default values.
     * @return Returns `this`.
     */
    public FlatBufferBuilder forceDefaults(boolean forceDefaults){
        this.force_defaults = forceDefaults;
        return this;
    }

    /**
     * Get the ByteBuffer representing the FlatBuffer. Only call this after you've
     * called `finish()`. The actual data starts at the ByteBuffer's current position,
     * not necessarily at `0`.
     *
     * @return The {@link ByteBuffer} representing the FlatBuffer
     */
    public ByteBuffer dataBuffer() {
        finished();
        return bb;
    }

   /**
    * The FlatBuffer data doesn't start at offset 0 in the {@link ByteBuffer}, but
    * now the {@code ByteBuffer}'s position is set to that location upon {@link #finish(int)}.
    *
    * @return The {@link ByteBuffer#position() position} the data starts in {@link #dataBuffer()}
    * @deprecated This method should not be needed anymore, but is left
    * here for the moment to document this API change. It will be removed in the future.
    */
    @Deprecated
    private int dataStart() {
        finished();
        return space;
    }

   /**
    * A utility function to copy and return the ByteBuffer data from `start` to
    * `start` + `length` as a `byte[]`.
    *
    * @param start Start copying at this offset.
    * @param length How many bytes to copy.
    * @return A range copy of the {@link #dataBuffer() data buffer}.
    * @throws IndexOutOfBoundsException If the range of bytes is ouf of bound.
    */
    public byte[] sizedByteArray(int start, int length){
        finished();
        byte[] array = new byte[length];
        bb.position(start);
        bb.get(array);
        return array;
    }

   /**
    * A utility function to copy and return the ByteBuffer data as a `byte[]`.
    *
    * @return A full copy of the {@link #dataBuffer() data buffer}.
    */
    public byte[] sizedByteArray() {
        return sizedByteArray(space, bb.capacity() - space);
    }

    /**
     * A utility function to return an InputStream to the ByteBuffer data
     *
     * @return An InputStream that starts at the beginning of the ByteBuffer data
     *         and can read to the end of it.
     */
    public InputStream sizedInputStream() {
        finished();
        ByteBuffer duplicate = bb.duplicate();
        duplicate.position(space);
        duplicate.limit(bb.capacity());
        return new ByteBufferBackedInputStream(duplicate);
    }

    /**
     * A class that allows a user to create an InputStream from a ByteBuffer.
     */
    static class ByteBufferBackedInputStream extends InputStream {

        ByteBuffer buf;

        public ByteBufferBackedInputStream(ByteBuffer buf) {
            this.buf = buf;
        }

        public int read() throws IOException {
            try {
                return buf.get() & 0xFF;
            } catch(BufferUnderflowException e) {
                return -1;
            }
        }
    }

}

/// @}
