/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow;

import java.lang.reflect.Array;
import java.nio.Buffer;
import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import org.tensorflow.types.Types;
import org.tensorflow.types.TFType;
import org.tensorflow.types.TFInt32;
import org.tensorflow.types.TFFloat;
import org.tensorflow.types.TFDouble;
import org.tensorflow.types.TFInt64;

/**
 * A statically typed multi-dimensional array whose elements are of a type described by T.
 * Appropriate types T are declared in the package {@link org.tensorflow.types} and are
 * subtypes of {@link org.tensorflow.types.TFType}.
 *
 * <p>Instances of a Tensor are <b>not</b> thread-safe.
 *
 * <p><b>WARNING:</b> Resources consumed by the Tensor object <b>must</b> be explicitly freed by
 * invoking the {@link #close()} method when the object is no longer needed. For example, using a
 * try-with-resources block:
 *
 * <pre>{@code
 * try (Tensor t = Tensor.create(...)) {
 *   doSomethingWith(t);
 * }
 * }</pre>
 */
public final class Tensor<T> implements AutoCloseable {

  /**
   * Creates a Tensor from a Java object.
   *
   * <p>A {@code Tensor} is a multi-dimensional array of elements of a limited set of types ({@link
   * types}), so not all Java objects can be converted to a {@code Tensor}. In particular, the
   * argument {@code obj} must be either a primitive (float, double, int, long, boolean, byte) or
   * a multi-dimensional array of one of those primitives. The argument {@code type} specifies how
   * to interpret the first argument as a TensorFlow type. For example:
   *
   * <pre>{@code
   * // Valid: A 64-bit integer scalar.
   * Tensor<TFInt64> s = Tensor.create(42L, TFInt64.class);
   *
   * // Valid: A 3x2 matrix of floats.
   * float[][] matrix = new float[3][2];
   * Tensor<TFFloat> m = Tensor.create(matrix, TFFloat.class);
   *
   * // Invalid: Will throw an IllegalArgumentException as an arbitrary Object
   * // does not fit into the TensorFlow type system.
   * Tensor<?> o = Tensor.create(new Object(), ...);
   *
   * // Invalid: Will throw an IllegalArgumentException since there are
   * // a differing number of elements in each row of this 2-D array.
   * int[][] twoD = new int[2][];
   * twoD[0] = new int[1];
   * twoD[1] = new int[2];
   * Tensor<TFInt32> x = Tensor.create(twoD, TFInt32.class);
   * }</pre>
   *
   * {@link types.TFString} typed Tensors are multi-dimensional arrays of
   * arbitrary byte sequences, so can be initialized from arrays of
   * {@code byte[]} elements. For example:
   *
   * <pre>{@code
   * // Valid: A TFString tensor.
   * Tensor<TFString> s = Tensor.create(new byte[]{1, 2, 3}, TFString.class);
   *
   * // Java Strings will need to be encoded into a byte-sequence.
   * String mystring = "foo";
   * Tensor<TFString> s = Tensor.create(mystring.getBytes("UTF-8"), TFString.class);
   *
   * // Valid: Matrix of TFString tensors.
   * // Each element might have a different length.
   * byte[][][] matrix = new byte[2][2][];
   * matrix[0][0] = "this".getBytes("UTF-8");
   * matrix[0][1] = "is".getBytes("UTF-8");
   * matrix[1][0] = "a".getBytes("UTF-8");
   * matrix[1][1] = "matrix".getBytes("UTF-8");
   * Tensor<TFString> m = Tensor.create(matrix, TFString.class);
   * }</pre>
   *
   * @param obj The object to convert to a Tensor<T>. Note that whether the it
   *     is compatible with the type T is not checked by the type system. For
   *     type-safe creation of tensors, use {@link op.Tensors}.
   *
   * @param type The class object representing the type T.
   *
   * @throws IllegalArgumentException if {@code obj} is not compatible with the TensorFlow type
   *     system.
   *
   * @see org.tensorflow.op.Tensors
   */
  public static <T extends TFType> Tensor<T> create(Object obj, Class<T> type) {
    DataType dt2 = Types.dataType(type);
    if (!objectCompatWithType(obj, dt2)) {
      throw new IllegalArgumentException("Data type of object does not match T (expected " + dt2 + ", got " + dataTypeOf(obj) + ")");
    }
    return create(obj, dt2);
  }

  /** Returns whether the object {@code obj} can represent a tensor with
   *  data type {@code dtype}.
   */
  private static boolean objectCompatWithType(Object obj, DataType dtype) {
   /*  TODO(andrewmyers): Probably should not be built using dataTypeOf,
    *  which is somewhat questionable once we allow a give Java type
    *  to represent multiple tensor types.
    */
    DataType dto = dataTypeOf(obj);
    if (dto.equals(dtype)) {
      return true;
    }
    if (dto == DataType.STRING && dtype == DataType.UINT8) {
      return true;
    }
    return false;
  }
  
  /**
   * Creates a tensor from an object whose class is inspected to figure out what the underlying data
   * type should be. The parameter {@code T} must match the data type of the object, but this is <em>not</em> checked.
   * Not all types {@code T} can be distinguished based on the run-time class of {@code obj}. For example,
   * a {@code byte[]} array could be either a 1-D tensor of uint8 or a scalar string.
   * Use the class {@link org.tensorflow.Tensors} for methods that create Tensors in a fully type-safe
   * way.
   * @throws IllegalArgumentException if {@code obj} is not compatible with the TensorFlow type
   *     system.
   */
  public static <T extends TFType> Tensor<T> create(Object obj) {
    return create(obj, dataTypeOf(obj));
  }
  
  /**
   * Creates a tensor from an object.
   * Requires the parameter {@code T} to match {@code type}, but this condition is not
   * checked statically. Use class {@link org.tensorflow.Tensors} to
   * create Tensors in a fully type-safe way.
   * @param obj the object supplying the tensor data. It type must be compatible with T.
   * @param dtype the DataType representation of the type T
   * @return the new tensor
   */
  static <T extends TFType> Tensor<T> create(Object obj, DataType dtype) {
    Tensor<T> t = new Tensor<T>(dtype);
    t.shapeCopy = new long[numDimensions(obj, dtype)];
    fillShape(obj, 0, t.shapeCopy);
    if (t.dtype != DataType.STRING) {
      int byteSize = elemByteSize(t.dtype) * numElements(t.shapeCopy);
      t.nativeHandle = allocate(t.dtype.c(), t.shapeCopy, byteSize);
      setValue(t.nativeHandle, obj);
    } else if (t.shapeCopy.length != 0) {
      t.nativeHandle = allocateNonScalarBytes(t.shapeCopy, (Object[]) obj);
    } else {
      t.nativeHandle = allocateScalarBytes((byte[]) obj);
    }
    return t;
  }

  /**
   * Create a {@link TFInt32} Tensor with data from the given buffer.
   *
   * <p>Creates a Tensor with the given shape by copying elements from the buffer (starting from its
   * current position) into the tensor. For example, if {@code shape = {2,3} } (which represents a
   * 2x3 matrix) then the buffer must have 6 elements remaining, which will be consumed by this
   * method.
   *
   * @param shape the tensor shape.
   * @param data a buffer containing the tensor data.
   * @throws IllegalArgumentException If the tensor shape is not compatible with the buffer
   */
  public static Tensor<TFInt32> create(long[] shape, IntBuffer data) {
    Tensor<TFInt32> t = allocateForBuffer(DataType.INT32, shape, data.remaining());
    t.buffer().asIntBuffer().put(data);
    return t;
  }

  /**
   * Create a {@link TFFloat} Tensor with data from the given buffer.
   *
   * <p>Creates a Tensor with the given shape by copying elements from the buffer (starting from its
   * current position) into the tensor. For example, if {@code shape = {2,3} } (which represents a
   * 2×3 matrix) then the buffer must have 6 elements remaining, which will be consumed by this
   * method.
   *
   * @param shape the tensor shape.
   * @param data a buffer containing the tensor data.
   * @throws IllegalArgumentException If the tensor shape is not compatible with the buffer
   */
  public static Tensor<TFFloat> create(long[] shape, FloatBuffer data) {
    Tensor<TFFloat> t = allocateForBuffer(DataType.FLOAT, shape, data.remaining());
    t.buffer().asFloatBuffer().put(data);
    return t;
  }

  /**
   * Create a {@link TFDouble} Tensor with data from the given buffer.
   *
   * <p>Creates a Tensor with the given shape by copying elements from the buffer (starting from its
   * current position) into the tensor. For example, if {@code shape = {2,3} } (which represents a
   * 2x3 matrix) then the buffer must have 6 elements remaining, which will be consumed by this
   * method.
   *
   * @param shape the tensor shape.
   * @param data a buffer containing the tensor data.
   * @throws IllegalArgumentException If the tensor shape is not compatible with the buffer
   */
  public static Tensor<TFDouble> create(long[] shape, DoubleBuffer data) {
    Tensor<TFDouble> t = allocateForBuffer(DataType.DOUBLE, shape, data.remaining());
    t.buffer().asDoubleBuffer().put(data);
    return t;
  }

  /**
   * Create an {@link TFInt64} Tensor with data from the given buffer.
   *
   * <p>Creates a Tensor with the given shape by copying elements from the buffer (starting from its
   * current position) into the tensor. For example, if {@code shape = {2,3} } (which represents a
   * 2x3 matrix) then the buffer must have 6 elements remaining, which will be consumed by this
   * method.
   *
   * @param shape the tensor shape.
   * @param data a buffer containing the tensor data.
   * @throws IllegalArgumentException If the tensor shape is not compatible with the buffer
   */
  public static Tensor<TFInt64> create(long[] shape, LongBuffer data) {
    Tensor<TFInt64> t = allocateForBuffer(DataType.INT64, shape, data.remaining());
    t.buffer().asLongBuffer().put(data);
    return t;
  }

  /**
   * Create a Tensor of any type with data from the given buffer.
   *
   * <p>Creates a Tensor with the provided shape of any type where the tensor's data has been
   * encoded into {@code data} as per the specification of the TensorFlow <a
   * href="https://www.tensorflow.org/code/tensorflow/c/c_api.h">C
   * API</a>.
   *
   * @param <T> the tensor element type
   * @param type the tensor element type, represented as a class object.
   * @param shape the tensor shape.
   * @param data a buffer containing the tensor data.
   * @throws IllegalArgumentException If the tensor datatype or shape is not compatible with the
   *     buffer
   */
  public static <T extends TFType> Tensor<T> create(Class<T> type, long[] shape, ByteBuffer data) {
    return create(Types.dataType(type), shape, data);
  }

  /**
   * Creates a Tensor of any type with data from the given buffer.
   *
   * <p>Creates a Tensor with the provided shape of any type where the tensor's data has been
   * encoded into {@code data} as per the specification of the TensorFlow <a
   * href="https://www.tensorflow.org/code/tensorflow/c/c_api.h">C API</a>.
   *
   * @param <T> The tensor element type
   * @param type the tensor element type, specified as a DataType. This must
   *     agree with T.
   * @param shape the tensor shape.
   * @param data a buffer containing the tensor data.
   * @throws IllegalArgumentException If the tensor datatype or shape is not compatible with the
   *     buffer
   * @deprecated The type-safe version of {@code create} is preferred.
   */
  @Deprecated
  public static <T> Tensor<T> create(DataType dtype, long[] shape, ByteBuffer data) {
    int nremaining = 0;
    if (dtype != DataType.STRING) {
      int elemBytes = elemByteSize(dtype);
      if (data.remaining() % elemBytes != 0) {
        throw new IllegalArgumentException(
            String.format(
                "ByteBuffer with %d bytes is not compatible with a %s Tensor (%d bytes/element)",
                data.remaining(), dtype.toString(), elemBytes));
      }
      nremaining = data.remaining() / elemBytes;
    } else {
      nremaining = data.remaining();
    }
    Tensor<T> t = allocateForBuffer(dtype, shape, nremaining);
    t.buffer().put(data);
    return t;
  }
  
  /** Returns this Tensor object with the type {@code Tensor<U>}. An exception is
   * thrown if the actual data type of this object does not match the type {@code U}.
   * This method is useful when given a value of type {@code Tensor<?>}.
   * @param type any (non-null) array of the correct type.
   * @return this
   */
 @SuppressWarnings("unchecked") public <U extends TFType> Tensor<U> expect(Class<U> type) {
    DataType dt = Types.dataType(type);
    if (!dt.equals(dtype)) {
      throw new IllegalArgumentException("Cannot cast from tensor of " + dtype + " to tensor of " + dt);
    }
    return ((Tensor<U>) this); 
  }

  // Helper function to allocate a Tensor for the create() methods that create a Tensor from
  // a java.nio.Buffer.
  // Requires: dataType matches T
  private static <T> Tensor<T> allocateForBuffer(DataType dataType, long[] shape, int nBuffered) {
    final int nflattened = numElements(shape);
    int nbytes = 0;
    if (dataType != DataType.STRING) {
      if (nBuffered != nflattened) {
        throw incompatibleBuffer(nBuffered, shape);
      }
      nbytes = nflattened * elemByteSize(dataType);
    } else {
      // DT_STRING tensor encoded in a ByteBuffer.
      nbytes = nBuffered;
    }
    Tensor<T> t = new Tensor<T>(dataType);
    t.shapeCopy = Arrays.copyOf(shape, shape.length);
    t.nativeHandle = allocate(t.dtype.c(), t.shapeCopy, nbytes);
    return t;
  }

  /**
   * Release resources associated with the Tensor.
   *
   * <p><b>WARNING:</b>If not invoked, memory will be leaked.
   *
   * <p>The Tensor object is no longer usable after {@code close} returns.
   */
  @Override
  public void close() {
    if (nativeHandle != 0) {
      delete(nativeHandle);
      nativeHandle = 0;
    }
  }

  /** Returns the {@link DataType} of elements stored in the Tensor. */
  public DataType dataType() {
    return dtype;
  }

  /**
   * Returns the number of dimensions (sometimes referred to as <a
   * href="https://www.tensorflow.org/resources/dims_types.html#rank">rank</a>) of the Tensor.
   *
   * <p>Will be 0 for a scalar, 1 for a vector, 2 for a matrix, 3 for a 3-dimensional tensor etc.
   */
  public int numDimensions() {
    return shapeCopy.length;
  }

  /** Returns the size, in bytes, of the tensor data. */
  public int numBytes() {
    return buffer().remaining();
  }

  /** Returns the number of elements in a flattened (1-D) view of the tensor. */
  public int numElements() {
    return numElements(shapeCopy);
  }

  /**
   * Returns the <a href="https://www.tensorflow.org/resources/dims_types.html#shape">shape</a> of
   * the Tensor, i.e., the sizes of each dimension.
   *
   * @return an array where the i-th element is the size of the i-th dimension of the tensor.
   */
  public long[] shape() {
    return shapeCopy;
  }

  /**
   * Returns the value in a scalar {@link TFFloat} tensor.
   *
   * @throws IllegalArgumentException if the Tensor does not represent a float scalar.
   */
  public float floatValue() {
    return scalarFloat(nativeHandle);
  }

  /**
   * Returns the value in a scalar {@link TFDouble} tensor.
   *
   * @throws IllegalArgumentException if the Tensor does not represent a double scalar.
   */
  public double doubleValue() {
    return scalarDouble(nativeHandle);
  }

  /**
   * Returns the value in a scalar {@link TFInt32} tensor.
   *
   * @throws IllegalArgumentException if the Tensor does not represent a int scalar.
   */
  public int intValue() {
    return scalarInt(nativeHandle);
  }

  /**
   * Returns the value in a scalar {@link TFInt64} tensor.
   *
   * @throws IllegalArgumentException if the Tensor does not represent a long scalar.
   */
  public long longValue() {
    return scalarLong(nativeHandle);
  }

  /**
   * Returns the value in a scalar {@link TFBool} tensor.
   *
   * @throws IllegalArgumentException if the Tensor does not represent a boolean scalar.
   */
  public boolean booleanValue() {
    return scalarBoolean(nativeHandle);
  }

  /**
   * Returns the value in a scalar {@link TFString} tensor.
   *
   * @throws IllegalArgumentException if the Tensor does not represent a boolean scalar.
   */
  public byte[] bytesValue() {
    return scalarBytes(nativeHandle);
  }

  /**
   * Copies the contents of the tensor to {@code dst} and returns {@code dst}.
   *
   * <p>For non-scalar tensors, this method copies the contents of the underlying tensor to a Java
   * array. For scalar tensors, use one of {@link #bytesValue()}, {@link #floatValue()}, {@link
   * #doubleValue()}, {@link #intValue()}, {@link #longValue()} or {@link #booleanValue()} instead.
   * The type and shape of {@code dst} must be compatible with the tensor. For example:
   *
   * <pre>{@code
   * int matrix[2][2] = {{1,2},{3,4}};
   * try(Tensor t = Tensor.create(matrix)) {
   *   // Succeeds and prints "3"
   *   int[][] copy = new int[2][2];
   *   System.out.println(t.copyTo(copy)[1][0]);
   *
   *   // Throws IllegalArgumentException since the shape of dst does not match the shape of t.
   *   int[][] dst = new int[4][1];
   *   t.copyTo(dst);
   * }
   * }</pre>
   *
   * @throws IllegalArgumentException if the tensor is a scalar or if {@code dst} is not compatible
   *     with the tensor (for example, mismatched data types or shapes).
   */
  public <U> U copyTo(U dst) {
    throwExceptionIfTypeIsIncompatible(dst);
    readNDArray(nativeHandle, dst);
    return dst;
  }

  /**
   * Write the data of a {@link TFInt32} tensor into the given buffer.
   *
   * <p>Copies {@code numElements()} elements to the buffer.
   *
   * @param dst the destination buffer
   * @throws BufferOverflowException If there is insufficient space in the given buffer for the data
   *     in this tensor
   * @throws IllegalArgumentException If the tensor datatype is not {@link TFInt32}
   */
  public void writeTo(IntBuffer dst) {
    if (dtype != DataType.INT32) {
      throw incompatibleBuffer(dst, dtype);
    }
    ByteBuffer src = buffer();
    dst.put(src.asIntBuffer());
  }

  /**
   * Write the data of a {@link TFFloat} tensor into the given buffer.
   *
   * <p>Copies {@code numElements()} elements to the buffer.
   *
   * @param dst the destination buffer
   * @throws BufferOverflowException If there is insufficient space in the given buffer for the data
   *     in this tensor
   * @throws IllegalArgumentException If the tensor datatype is not {@link TFFloat}
   */
  public void writeTo(FloatBuffer dst) {
    if (dtype != DataType.FLOAT) {
      throw incompatibleBuffer(dst, dtype);
    }
    ByteBuffer src = buffer();
    dst.put(src.asFloatBuffer());
  }

  /**
   * Write the data of a {@link TFDouble} tensor into the given buffer.
   *
   * <p>Copies {@code numElements()} elements to the buffer.
   *
   * @param dst the destination buffer
   * @throws BufferOverflowException If there is insufficient space in the given buffer for the data
   *     in this tensor
   * @throws IllegalArgumentException If the tensor datatype is not {@link TFDouble}
   */
  public void writeTo(DoubleBuffer dst) {
    if (dtype != DataType.DOUBLE) {
      throw incompatibleBuffer(dst, dtype);
    }
    ByteBuffer src = buffer();
    dst.put(src.asDoubleBuffer());
  }

  /**
   * Write the data of a {@link TFInt64} tensor into the given buffer.
   *
   * <p>Copies {@code numElements()} elements to the buffer.
   *
   * @param dst the destination buffer
   * @throws BufferOverflowException If there is insufficient space in the given buffer for the data
   *     in this tensor
   * @throws IllegalArgumentException If the tensor datatype is not {@link TFInt64}
   */
  public void writeTo(LongBuffer dst) {
    if (dtype != DataType.INT64) {
      throw incompatibleBuffer(dst, dtype);
    }
    ByteBuffer src = buffer();
    dst.put(src.asLongBuffer());
  }

  /**
   * Write the tensor data into the given buffer.
   *
   * <p>Copies {@code numBytes()} bytes to the buffer in native byte order for primitive types.
   *
   * @param dst the destination buffer
   * @throws BufferOverflowException If there is insufficient space in the given buffer for the data
   *     in this tensor
   */
  public void writeTo(ByteBuffer dst) {
    ByteBuffer src = buffer();
    dst.put(src);
  }

  /** Returns a string describing the type and shape of the Tensor. */
  @Override
  public String toString() {
    return String.format("%s tensor with shape %s", dtype.toString(), Arrays.toString(shape()));
  }

  /**
   * Create a Tensor object from a handle to the C TF_Tensor object.
   *
   * <p>Takes ownership of the handle.</p>
   */
  static Tensor<?> fromHandle(long handle) {
    @SuppressWarnings("rawtypes")
    Tensor<?> t = new Tensor(DataType.fromC(dtype(handle)));
    t.shapeCopy = shape(handle);
    t.nativeHandle = handle;
    return t;
  }

  /**
   * Create a Tensor object from a handle to the C TF_Tensor object.
   *
   * <p>Takes ownership of the handle. Requires the type of
   *   the created tensor to be {@code type}, i.e., the same as
   *   {@code T}.</p>
   */
  static <T extends TFType> Tensor<T> fromHandle(long handle, Class<T> type) {
    Tensor<T> t = new Tensor<T>(DataType.fromC(dtype(handle)));
    t.shapeCopy = shape(handle);
    t.nativeHandle = handle;
    return t;
  }

  long getNativeHandle() {
    return nativeHandle;
  }

  private long nativeHandle;
  private DataType dtype;
  private long[] shapeCopy = null;

  private Tensor(DataType t) {
    dtype = t;
  }
  
  private ByteBuffer buffer() {
    return buffer(nativeHandle).order(ByteOrder.nativeOrder());
  }

  private static IllegalArgumentException incompatibleBuffer(Buffer buf, DataType dataType) {
    return new IllegalArgumentException(
        String.format("cannot use %s with Tensor of type %s", buf.getClass().getName(), dataType));
  }

  private static IllegalArgumentException incompatibleBuffer(int numElements, long[] shape) {
    return new IllegalArgumentException(
        String.format(
            "buffer with %d elements is not compatible with a Tensor with shape %s",
            numElements, Arrays.toString(shape)));
  }

  private static int numElements(long[] shape) {
    // assumes a fully-known shape
    int n = 1;
    for (int i = 0; i < shape.length; i++) {
      n *= (int) shape[i];
    }
    return n;
  }

  private static int elemByteSize(DataType dataType) {
    switch (dataType) {
      case FLOAT:
      case INT32:
        return 4;
      case DOUBLE:
      case INT64:
        return 8;
      case BOOL:
      case UINT8:
        return 1;
      case STRING:
        throw new IllegalArgumentException("STRING tensors do not have a fixed element size");
    }
    throw new IllegalArgumentException("DataType " + dataType + " is not supported yet");
  }

  private static void throwExceptionIfNotByteOfByteArrays(Object array) {
    if (!array.getClass().getName().equals("[[B")) {
      throw new IllegalArgumentException(
          "object cannot be converted to a Tensor as it includes an array with null elements");
    }
  }

  /**
   * The default TensorFlow data type that Java object o corresponds to. Some Java objects
   * represent more than one TensorFlow data type; for example, 'byte' can represent
   * both {@code uint8} and {@code string}, with the latter being the default interpretation.
   */
  private static DataType dataTypeOf(Object o) {
    if (o.getClass().isArray()) {
      if (Array.getLength(o) == 0) {
        throw new IllegalArgumentException("cannot create Tensors with a 0 dimension");
      }
      // byte[] is a DataType.STRING scalar.
      Object e = Array.get(o, 0);
      if (e == null) {
        throwExceptionIfNotByteOfByteArrays(o);
        return DataType.STRING;
      }
      if (Byte.class.isInstance(e) || byte.class.isInstance(e)) {
        return DataType.STRING;
      }
      return dataTypeOf(e);
    }
    if (Float.class.isInstance(o) || float.class.isInstance(o)) {
      return DataType.FLOAT;
    } else if (Double.class.isInstance(o) || double.class.isInstance(o)) {
      return DataType.DOUBLE;
    } else if (Integer.class.isInstance(o) || int.class.isInstance(o)) {
      return DataType.INT32;
    } else if (Long.class.isInstance(o) || long.class.isInstance(o)) {
      return DataType.INT64;
    } else if (Boolean.class.isInstance(o) || boolean.class.isInstance(o)) {
      return DataType.BOOL;
    } else {
      throw new IllegalArgumentException("cannot create Tensors of " + o.getClass().getName());
    }
  }

  /** Return the number of dimensions of the tensor that object {@code o}
   *  represents as a tensor whose datatype is {@code dtype}. Normally
   *  this is the same as the number of dimensions of o itself, but
   *  is one smaller for tensors of strings.
   *  @param o The object to inspect. It must be a valid representation
   *           of the given data type.
   *  @param dtype The expected data type of the tensor.
   */
  private static int numDimensions(Object o, DataType dtype) {
    int ret = numArrayDimensions(o);
    if (dtype == DataType.STRING) {
      return ret - 1;
    }
    return ret;
  }
  
  /** Returns the number of dimensions of the array object o.
   *  Returns 0 if o is not an array.
   */
  private static int numArrayDimensions(Object o) {
    Class<?> c = o.getClass();
    if (!c.isArray()) return 0;
    String s = c.getName();
    int i = 0;
    while (s.charAt(i) == '[') {
      i++;
    }
    return i;
  }

  private static void fillShape(Object o, int dim, long[] shape) {
    if (shape == null || dim == shape.length) {
      return;
    }
    final int len = Array.getLength(o);
    if (shape[dim] == 0) {
      shape[dim] = len;
    } else if (shape[dim] != len) {
      throw new IllegalArgumentException(
          String.format("mismatched lengths (%d and %d) in dimension %d", shape[dim], len, dim));
    }
    for (int i = 0; i < len; ++i) {
      fillShape(Array.get(o, i), dim + 1, shape);
    }
  }

  private void throwExceptionIfTypeIsIncompatible(Object o) {
    final int rank = numDimensions();
    final int oRank = numDimensions(o, dtype);
    if (oRank != rank) {
      throw new IllegalArgumentException(
          String.format(
              "cannot copy Tensor with %d dimensions into an object with %d", rank, oRank));
    }
    if (!objectCompatWithType(o, dtype)) {
      throw new IllegalArgumentException(
          String.format(
              "cannot copy Tensor with DataType %s into an object of type %s",
              dtype.toString(), o.getClass().getName()));
    }
    long[] oShape = new long[rank];
    fillShape(o, 0, oShape);
    for (int i = 0; i < oShape.length; ++i) {
      if (oShape[i] != shape()[i]) {
        throw new IllegalArgumentException(
            String.format(
                "cannot copy Tensor with shape %s into object with shape %s",
                Arrays.toString(shape()), Arrays.toString(oShape)));
      }
    }
  }

  private static native long allocate(int dtype, long[] shape, long byteSize);

  private static native long allocateScalarBytes(byte[] value);

  private static native long allocateNonScalarBytes(long[] shape, Object[] value);

  private static native void delete(long handle);

  private static native ByteBuffer buffer(long handle);

  private static native int dtype(long handle);

  private static native long[] shape(long handle);

  private static native void setValue(long handle, Object value);

  private static native float scalarFloat(long handle);

  private static native double scalarDouble(long handle);

  private static native int scalarInt(long handle);

  private static native long scalarLong(long handle);

  private static native boolean scalarBoolean(long handle);

  private static native byte[] scalarBytes(long handle);

  private static native void readNDArray(long handle, Object value);

  static {
    TensorFlow.init();
  }
}
