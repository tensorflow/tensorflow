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
import java.util.HashMap;

/**
 * A statically typed multi-dimensional array whose elements are of a type described by T.
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
   * <p>A {@code Tensor} is a multi-dimensional array of elements of a limited set of types. Not all
   * Java objects can be converted to a {@code Tensor}. In particular, the argument {@code obj} must
   * be either a primitive (float, double, int, long, boolean, byte) or a multi-dimensional array of
   * one of those primitives. The argument {@code type} specifies how to interpret the first
   * argument as a TensorFlow type. For example:
   *
   * <pre>{@code
   * // Valid: A 64-bit integer scalar.
   * Tensor<Long> s = Tensor.create(42L, Long.class);
   *
   * // Valid: A 3x2 matrix of floats.
   * float[][] matrix = new float[3][2];
   * Tensor<Float> m = Tensor.create(matrix, Float.class);
   *
   * // Invalid: Will throw an IllegalArgumentException as an arbitrary Object
   * // does not fit into the TensorFlow type system.
   * Tensor<?> o = Tensor.create(new Object())
   *
   * // Invalid: Will throw an IllegalArgumentException since there are
   * // a differing number of elements in each row of this 2-D array.
   * int[][] twoD = new int[2][];
   * twoD[0] = new int[1];
   * twoD[1] = new int[2];
   * Tensor<Integer> x = Tensor.create(twoD, Integer.class);
   * }</pre>
   *
   * {@link String}-typed Tensors are multi-dimensional arrays of arbitrary byte sequences, so can
   * be initialized from arrays of {@code byte[]} elements. For example:
   *
   * <pre>{@code
   * // Valid: A String tensor.
   * Tensor<String> s = Tensor.create(new byte[]{1, 2, 3}, String.class);
   *
   * // Java Strings will need to be encoded into a byte-sequence.
   * String mystring = "foo";
   * Tensor<String> s = Tensor.create(mystring.getBytes("UTF-8"), String.class);
   *
   * // Valid: Matrix of String tensors.
   * // Each element might have a different length.
   * byte[][][] matrix = new byte[2][2][];
   * matrix[0][0] = "this".getBytes("UTF-8");
   * matrix[0][1] = "is".getBytes("UTF-8");
   * matrix[1][0] = "a".getBytes("UTF-8");
   * matrix[1][1] = "matrix".getBytes("UTF-8");
   * Tensor<String> m = Tensor.create(matrix, String.class);
   * }</pre>
   *
   * @param obj The object to convert to a {@code Tensor<T>}. Note that whether it is compatible
   *     with the type T is not checked by the type system. For type-safe creation of tensors, use
   *     {@link Tensors}.
   * @param type The class object representing the type T.
   * @throws IllegalArgumentException if {@code obj} is not compatible with the TensorFlow type
   *     system.
   */
  @SuppressWarnings("unchecked")
  public static <T> Tensor<T> create(Object obj, Class<T> type) {
    DataType dtype = DataType.fromClass(type);
    if (!objectCompatWithType(obj, dtype)) {
      throw new IllegalArgumentException(
          "DataType of object does not match T (expected "
              + dtype
              + ", got "
              + dataTypeOf(obj)
              + ")");
    }
    return (Tensor<T>) create(obj, dtype);
  }

  /**
   * Creates a tensor from an object whose class is inspected to figure out what the underlying data
   * type should be.
   *
   * @throws IllegalArgumentException if {@code obj} is not compatible with the TensorFlow type
   *     system.
   */
  public static Tensor<?> create(Object obj) {
    return create(obj, dataTypeOf(obj));
  }

  /**
   * Create a Tensor of data type {@code dtype} from a Java object. Requires the parameter {@code T}
   * to match {@code type}, but this condition is not checked.
   *
   * @param obj the object supplying the tensor data.
   * @param dtype the data type of the tensor to create. It must be compatible with the run-time
   *     type of the object.
   * @return the new tensor
   */
  private static Tensor<?> create(Object obj, DataType dtype) {
    @SuppressWarnings("rawtypes")
    Tensor<?> t = new Tensor(dtype);
    t.shapeCopy = new long[numDimensions(obj, dtype)];
    fillShape(obj, 0, t.shapeCopy);
    long nativeHandle;
    if (t.dtype != DataType.STRING) {
      int byteSize = elemByteSize(t.dtype) * numElements(t.shapeCopy);
      nativeHandle = allocate(t.dtype.c(), t.shapeCopy, byteSize);
      setValue(nativeHandle, obj);
    } else if (t.shapeCopy.length != 0) {
      nativeHandle = allocateNonScalarBytes(t.shapeCopy, (Object[]) obj);
    } else {
      nativeHandle = allocateScalarBytes((byte[]) obj);
    }
    t.nativeRef = new NativeReference(nativeHandle);
    return t;
  }

  /**
   * Create a {@link Integer} Tensor with data from the given buffer.
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
  public static Tensor<Integer> create(long[] shape, IntBuffer data) {
    Tensor<Integer> t = allocateForBuffer(DataType.INT32, shape, data.remaining());
    t.buffer().asIntBuffer().put(data);
    return t;
  }

  /**
   * Create a {@link Float} Tensor with data from the given buffer.
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
  public static Tensor<Float> create(long[] shape, FloatBuffer data) {
    Tensor<Float> t = allocateForBuffer(DataType.FLOAT, shape, data.remaining());
    t.buffer().asFloatBuffer().put(data);
    return t;
  }

  /**
   * Create a {@link Double} Tensor with data from the given buffer.
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
  public static Tensor<Double> create(long[] shape, DoubleBuffer data) {
    Tensor<Double> t = allocateForBuffer(DataType.DOUBLE, shape, data.remaining());
    t.buffer().asDoubleBuffer().put(data);
    return t;
  }

  /**
   * Create an {@link Long} Tensor with data from the given buffer.
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
  public static Tensor<Long> create(long[] shape, LongBuffer data) {
    Tensor<Long> t = allocateForBuffer(DataType.INT64, shape, data.remaining());
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
  public static <T> Tensor<T> create(Class<T> type, long[] shape, ByteBuffer data) {
    @SuppressWarnings("unchecked")
    Tensor<T> ret = (Tensor<T>) create(DataType.fromClass(type), shape, data);
    return ret;
  }

  private static Tensor<?> create(DataType dtype, long[] shape, ByteBuffer data) {
    int nremaining;
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
    Tensor<?> t = allocateForBuffer(dtype, shape, nremaining);
    t.buffer().put(data);
    return t;
  }

  /**
   * Returns this Tensor object with the type {@code Tensor<U>}. This method is useful when given a
   * value of type {@code Tensor<?>}.
   *
   * @param type any (non-null) array of the correct type.
   * @throws IllegalArgumentException if the actual data type of this object does not match the type
   *     {@code U}.
   */
  @SuppressWarnings("unchecked")
  public <U> Tensor<U> expect(Class<U> type) {
    DataType dt = DataType.fromClass(type);
    if (!dt.equals(dtype)) {
      throw new IllegalArgumentException(
          "Cannot cast from tensor of " + dtype + " to tensor of " + dt);
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
    long nativeHandle = allocate(t.dtype.c(), t.shapeCopy, nbytes);
    t.nativeRef = new NativeReference(nativeHandle);
    return t;
  }

  /**
   * Release resources associated with the Tensor.
   *
   * <p><b>WARNING:</b>This must be invoked for all tensors that were not been produced by an eager
   * operation or memory will be leaked.
   *
   * <p>The Tensor object is no longer usable after {@code close} returns.
   */
  @Override
  public void close() {
    nativeRef.release();
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
   * Returns the value in a scalar {@link Float} tensor.
   *
   * @throws IllegalArgumentException if the Tensor does not represent a float scalar.
   */
  public float floatValue() {
    return scalarFloat(getNativeHandle());
  }

  /**
   * Returns the value in a scalar {@link Double} tensor.
   *
   * @throws IllegalArgumentException if the Tensor does not represent a double scalar.
   */
  public double doubleValue() {
    return scalarDouble(getNativeHandle());
  }

  /**
   * Returns the value in a scalar {@link Integer} tensor.
   *
   * @throws IllegalArgumentException if the Tensor does not represent a int scalar.
   */
  public int intValue() {
    return scalarInt(getNativeHandle());
  }

  /**
   * Returns the value in a scalar {@link Long} tensor.
   *
   * @throws IllegalArgumentException if the Tensor does not represent a long scalar.
   */
  public long longValue() {
    return scalarLong(getNativeHandle());
  }

  /**
   * Returns the value in a scalar {@link Boolean} tensor.
   *
   * @throws IllegalArgumentException if the Tensor does not represent a boolean scalar.
   */
  public boolean booleanValue() {
    return scalarBoolean(getNativeHandle());
  }

  /**
   * Returns the value in a scalar {@link String} tensor.
   *
   * @throws IllegalArgumentException if the Tensor does not represent a boolean scalar.
   */
  public byte[] bytesValue() {
    return scalarBytes(getNativeHandle());
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
    readNDArray(getNativeHandle(), dst);
    return dst;
  }

  /**
   * Write the data of a {@link Integer} tensor into the given buffer.
   *
   * <p>Copies {@code numElements()} elements to the buffer.
   *
   * @param dst the destination buffer
   * @throws BufferOverflowException If there is insufficient space in the given buffer for the data
   *     in this tensor
   * @throws IllegalArgumentException If the tensor data type is not {@link Integer}
   */
  public void writeTo(IntBuffer dst) {
    if (dtype != DataType.INT32) {
      throw incompatibleBuffer(dst, dtype);
    }
    ByteBuffer src = buffer();
    dst.put(src.asIntBuffer());
  }

  /**
   * Write the data of a {@link Float} tensor into the given buffer.
   *
   * <p>Copies {@code numElements()} elements to the buffer.
   *
   * @param dst the destination buffer
   * @throws BufferOverflowException If there is insufficient space in the given buffer for the data
   *     in this tensor
   * @throws IllegalArgumentException If the tensor datatype is not {@link Float}
   */
  public void writeTo(FloatBuffer dst) {
    if (dtype != DataType.FLOAT) {
      throw incompatibleBuffer(dst, dtype);
    }
    ByteBuffer src = buffer();
    dst.put(src.asFloatBuffer());
  }

  /**
   * Write the data of a {@link Double} tensor into the given buffer.
   *
   * <p>Copies {@code numElements()} elements to the buffer.
   *
   * @param dst the destination buffer
   * @throws BufferOverflowException If there is insufficient space in the given buffer for the data
   *     in this tensor
   * @throws IllegalArgumentException If the tensor datatype is not {@link Double}
   */
  public void writeTo(DoubleBuffer dst) {
    if (dtype != DataType.DOUBLE) {
      throw incompatibleBuffer(dst, dtype);
    }
    ByteBuffer src = buffer();
    dst.put(src.asDoubleBuffer());
  }

  /**
   * Write the data of a {@link Long} tensor into the given buffer.
   *
   * <p>Copies {@code numElements()} elements to the buffer.
   *
   * @param dst the destination buffer
   * @throws BufferOverflowException If there is insufficient space in the given buffer for the data
   *     in this tensor
   * @throws IllegalArgumentException If the tensor datatype is not {@link Long}
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
   * <p>Takes ownership of the handle.
   */
  static Tensor<?> fromHandle(long handle) {
    @SuppressWarnings("rawtypes")
    Tensor<?> t = new Tensor(DataType.fromC(dtype(handle)));
    t.shapeCopy = shape(handle);
    t.nativeRef = new NativeReference(handle);
    return t;
  }

  /**
   * Create an eager Tensor object from a handle to the C TF_Tensor object.
   *
   * <p>Takes ownership of the handle.
   */
  static Tensor<?> fromHandle(long handle, EagerSession session) {
    Tensor<?> t = fromHandle(handle);
    t.nativeRef.eager(session, t);
    return t;
  }

  long getNativeHandle() {
    return nativeRef.tensorHandle;
  }

  private NativeReference nativeRef = null;
  private final DataType dtype;
  private long[] shapeCopy = null;

  private Tensor(DataType t) {
    dtype = t;
  }

  private ByteBuffer buffer() {
    return buffer(getNativeHandle()).order(ByteOrder.nativeOrder());
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
    int size = dataType.byteSize();
    if (size < 0) {
        throw new IllegalArgumentException("STRING tensors do not have a fixed element size");
    }
    return size;
  }

  private static void throwExceptionIfNotByteOfByteArrays(Object array) {
    if (!array.getClass().getName().equals("[[B")) {
      throw new IllegalArgumentException(
          "object cannot be converted to a Tensor as it includes an array with null elements");
    }
  }

  /**
   * Reference to the underlying native tensor
   *
   * <p>Tensors are commonly allocated in a `try-with-resources` statement, where they get
   * automatically released after executing the last line of the `try` block they were declared in.
   *
   * <p>They can also be attached to an eager session, where in this case their lifetime ends either
   * when this session is closed or when the Tensor instance is no longer referenced and have been
   * garbage-collected.
   *
   * <p>This helper class wraps the tensor native handle and support both situations; If an eager
   * reference to the tensor exists, it will take care of releasing the tensor at the end of its
   * life. If the tensor is being explicitly closed before this happens, it will take cake of
   * clearing its association with any eager session before cleaning up the resources.
   */
  private static class NativeReference {

    /** Attaches this reference to an eager session */
    private class EagerReference extends EagerSession.NativeReference {

      EagerReference(EagerSession session, Tensor<?> tensor) {
        super(session, tensor);
      }

      @Override
      void delete() {
        // Mark this eager reference as cleared since it has been deleted by the session
        NativeReference.this.eagerRef = null;
        NativeReference.this.release();
      }
    }

    NativeReference(long tensorHandle) {
      this.tensorHandle = tensorHandle;
    }

    void eager(EagerSession session, Tensor<?> tensor) {
      if (eagerRef != null) {
        throw new IllegalStateException("The tensor is already attached to an eager session");
      }
      eagerRef = new EagerReference(session, tensor);
    }

    synchronized void release() {
      if (tensorHandle != 0L) {
        // Clear any remaining eager reference to this tensor
        if (eagerRef != null) {
          eagerRef.clear();
          eagerRef = null;
        }
        Tensor.delete(tensorHandle);
        tensorHandle = 0L;
      }
    }

    private long tensorHandle;
    private EagerReference eagerRef;
  }

  private static HashMap<Class<?>, DataType> classDataTypes = new HashMap<>();

  static {
    classDataTypes.put(int.class, DataType.INT32);
    classDataTypes.put(Integer.class, DataType.INT32);
    classDataTypes.put(long.class, DataType.INT64);
    classDataTypes.put(Long.class, DataType.INT64);
    classDataTypes.put(float.class, DataType.FLOAT);
    classDataTypes.put(Float.class, DataType.FLOAT);
    classDataTypes.put(double.class, DataType.DOUBLE);
    classDataTypes.put(Double.class, DataType.DOUBLE);
    classDataTypes.put(byte.class, DataType.STRING);
    classDataTypes.put(Byte.class, DataType.STRING);
    classDataTypes.put(boolean.class, DataType.BOOL);
    classDataTypes.put(Boolean.class, DataType.BOOL);
  }

  /** The class for the data type to which Java object o corresponds. */
  private static Class<?> baseObjType(Object o) {
    Class<?> c = o.getClass();
    while (c.isArray()) {
      c = c.getComponentType();
    }
    return c;
  }

  /**
   * The default TensorFlow data type to which Java object o corresponds. Some Java objects
   * represent more than one TensorFlow data type; for example, 'byte' can represent both {@code
   * uint8} and {@code string}, with the latter being the default interpretation.
   */
  private static DataType dataTypeOf(Object o) {
    Class<?> c = baseObjType(o);
    return dataTypeFromClass(c);
  }

  private static DataType dataTypeFromClass(Class<?> c) {
    DataType ret = classDataTypes.get(c);
    if (ret != null) {
      return ret;
    }
    throw new IllegalArgumentException("cannot create Tensors of type " + c.getName());
  }

  /**
   * Return the number of dimensions of the tensor that object {@code o} represents as a tensor
   * whose datatype is {@code dtype}. Normally this is the same as the number of dimensions of o
   * itself, but is one smaller for tensors of strings.
   *
   * @param o The object to inspect. It must be a valid representation of the given data type.
   * @param dtype The expected data type of the tensor.
   */
  private static int numDimensions(Object o, DataType dtype) {
    int ret = numArrayDimensions(o);
    if (dtype == DataType.STRING && ret > 0) {
      return ret - 1;
    }
    return ret;
  }

  /** Returns the number of dimensions of the array object o. Returns 0 if o is not an array. */
  private static int numArrayDimensions(Object o) {
    Class<?> c = o.getClass();
    int i = 0;
    while (c.isArray()) {
      c = c.getComponentType();
      i++;
    }
    return i;
  }

  /**
   * Fills in the remaining entries in the shape array starting from position {@code dim} with the
   * dimension sizes of the multidimensional array o. Checks that all arrays reachable from o have
   * sizes consistent with the filled-in shape, throwing IllegalArgumentException otherwise.
   */
  private static void fillShape(Object o, int dim, long[] shape) {
    if (shape == null || dim == shape.length) {
      return;
    }
    final int len = Array.getLength(o);
    if (len == 0) {
      throw new IllegalArgumentException("cannot create Tensors with a 0 dimension");
    }
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

  /** Returns whether the object {@code obj} can represent a tensor with data type {@code dtype}. */
  private static boolean objectCompatWithType(Object obj, DataType dtype) {
    Class<?> c = baseObjType(obj);
    DataType dto = dataTypeFromClass(c);
    int nd = numDimensions(obj, dto);
    if (!c.isPrimitive() && c != String.class && nd != 0) {
      throw new IllegalArgumentException(
          "cannot create non-scalar Tensors from arrays of boxed values");
    }
    if (dto.equals(dtype)) {
      return true;
    }
    if (dto == DataType.STRING && dtype == DataType.UINT8) {
      return true;
    }
    return false;
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
