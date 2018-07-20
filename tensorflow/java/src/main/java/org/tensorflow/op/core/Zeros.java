package org.tensorflow.op.core;

import java.nio.ByteBuffer;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Scope;
import org.tensorflow.op.annotation.Operator;

/**
 * An operator creating a constant initialized with zeros w.r.t its type and shape.
 *
 * @param <T> constant type
 */
@Operator
public class Zeros<T> implements Op, Operand<T> {

  /**
   * Factory method for this operator
   *
   * @param scope is a scope used to add the underlying operation.
   * @param type the tensor datatype.
   * @param shape the tensor shape.
   * @return a constant initialized with zeros
   * @throws IllegalArgumentException if the tensor type or shape cannot be initialized with zeros.
   */
  public static <T> Zeros<T> create(Scope scope, Class<T> type, Shape shape) {
    int numElements = (int) shape.numElements();
    if (numElements < 0) {
      throw new IllegalArgumentException("Only shapes with known dimension sizes can be used with zeroed constants");
    }
    int sizeInBytes = DataType.fromClass(type).sizeInBytes();
    if (sizeInBytes < 0) {
      throw new IllegalArgumentException(type.getSimpleName() + " constants cannot be initialized with zeros");
    }
    return new Zeros<T>(Constant.create(scope, type, shape, ByteBuffer.allocate(numElements * sizeInBytes)));
  }

  @Override
  public Output<T> asOutput() {
    return constant.asOutput();
  }
  
  public Constant<T> constant() {
    return constant;
  }
  
  private final Constant<T> constant;
  
  private Zeros(Constant<T> constant) {
    this.constant = constant;
  }
}
