package org.tensorflow.types;

import java.util.HashMap;
import java.util.Map;
import org.tensorflow.DataType;

/**
 * Utility class for representing TF types as Java types. For each TF type (e.g., int32),
 * there is a corresponding Java type (e.g., TFInt32) that represents it at compile time
 * and a corresponding immutable object (e.g., TFInt32.T) that represents it at run time.
 */
public class Types {

  private Types() {} // not instantiable

  static final Map<Class<?>, Integer> typeCodes = new HashMap<>();
  /** Convert to the equivalent DataType. */

  public static DataType dataType(Class<?> c) {
    Integer code = typeCodes.get(c);
    if (code == null) {
      throw new IllegalArgumentException("" + c + " is not a TensorFlow type.");
    }
    return DataType.fromC(code.intValue());
  }

  static final Map<Class<?>, Object> scalars = new HashMap<>();

  public static Object defaultScalar(Class<?> c) {
    return scalars.get(c);
  }

  /**
   * A marker interface for classes representing Tensorflow types.
   */
  public interface TFType {}
}
