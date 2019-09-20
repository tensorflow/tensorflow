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

import java.nio.ByteBuffer;

import static java.lang.Character.MIN_HIGH_SURROGATE;
import static java.lang.Character.MIN_LOW_SURROGATE;
import static java.lang.Character.MIN_SUPPLEMENTARY_CODE_POINT;

public abstract class Utf8 {

  /**
   * Returns the number of bytes in the UTF-8-encoded form of {@code sequence}. For a string,
   * this method is equivalent to {@code string.getBytes(UTF_8).length}, but is more efficient in
   * both time and space.
   *
   * @throws IllegalArgumentException if {@code sequence} contains ill-formed UTF-16 (unpaired
   *     surrogates)
   */
  public abstract int encodedLength(CharSequence sequence);

  /**
   * Encodes the given characters to the target {@link ByteBuffer} using UTF-8 encoding.
   *
   * <p>Selects an optimal algorithm based on the type of {@link ByteBuffer} (i.e. heap or direct)
   * and the capabilities of the platform.
   *
   * @param in the source string to be encoded
   * @param out the target buffer to receive the encoded string.
   */
  public abstract void encodeUtf8(CharSequence in, ByteBuffer out);

  /**
   * Decodes the given UTF-8 portion of the {@link ByteBuffer} into a {@link String}.
   *
   * @throws IllegalArgumentException if the input is not valid UTF-8.
   */
  public abstract String decodeUtf8(ByteBuffer buffer, int offset, int length);

  private static Utf8 DEFAULT;

  /**
   * Get the default UTF-8 processor.
   * @return the default processor
   */
  public static Utf8 getDefault() {
    if (DEFAULT == null) {
      DEFAULT = new Utf8Safe();
    }
    return DEFAULT;
  }

  /**
   * Set the default instance of the UTF-8 processor.
   * @param instance the new instance to use
   */
  public static void setDefault(Utf8 instance) {
    DEFAULT = instance;
  }

  /**
   * Utility methods for decoding bytes into {@link String}. Callers are responsible for extracting
   * bytes (possibly using Unsafe methods), and checking remaining bytes. All other UTF-8 validity
   * checks and codepoint conversion happen in this class.
   */
  static class DecodeUtil {

    /**
     * Returns whether this is a single-byte codepoint (i.e., ASCII) with the form '0XXXXXXX'.
     */
    static boolean isOneByte(byte b) {
      return b >= 0;
    }

    /**
     * Returns whether this is a two-byte codepoint with the form '10XXXXXX'.
     */
    static boolean isTwoBytes(byte b) {
      return b < (byte) 0xE0;
    }

    /**
     * Returns whether this is a three-byte codepoint with the form '110XXXXX'.
     */
    static boolean isThreeBytes(byte b) {
      return b < (byte) 0xF0;
    }

    static void handleOneByte(byte byte1, char[] resultArr, int resultPos) {
      resultArr[resultPos] = (char) byte1;
    }

    static void handleTwoBytes(
        byte byte1, byte byte2, char[] resultArr, int resultPos)
        throws IllegalArgumentException {
      // Simultaneously checks for illegal trailing-byte in leading position (<= '11000000') and
      // overlong 2-byte, '11000001'.
      if (byte1 < (byte) 0xC2) {
        throw new IllegalArgumentException("Invalid UTF-8: Illegal leading byte in 2 bytes utf");
      }
      if (isNotTrailingByte(byte2)) {
        throw new IllegalArgumentException("Invalid UTF-8: Illegal trailing byte in 2 bytes utf");
      }
      resultArr[resultPos] = (char) (((byte1 & 0x1F) << 6) | trailingByteValue(byte2));
    }

    static void handleThreeBytes(
        byte byte1, byte byte2, byte byte3, char[] resultArr, int resultPos)
        throws IllegalArgumentException {
      if (isNotTrailingByte(byte2)
              // overlong? 5 most significant bits must not all be zero
              || (byte1 == (byte) 0xE0 && byte2 < (byte) 0xA0)
              // check for illegal surrogate codepoints
              || (byte1 == (byte) 0xED && byte2 >= (byte) 0xA0)
              || isNotTrailingByte(byte3)) {
        throw new IllegalArgumentException("Invalid UTF-8");
      }
      resultArr[resultPos] = (char)
                                 (((byte1 & 0x0F) << 12) | (trailingByteValue(byte2) << 6) | trailingByteValue(byte3));
    }

    static void handleFourBytes(
        byte byte1, byte byte2, byte byte3, byte byte4, char[] resultArr, int resultPos)
        throws IllegalArgumentException{
      if (isNotTrailingByte(byte2)
              // Check that 1 <= plane <= 16.  Tricky optimized form of:
              //   valid 4-byte leading byte?
              // if (byte1 > (byte) 0xF4 ||
              //   overlong? 4 most significant bits must not all be zero
              //     byte1 == (byte) 0xF0 && byte2 < (byte) 0x90 ||
              //   codepoint larger than the highest code point (U+10FFFF)?
              //     byte1 == (byte) 0xF4 && byte2 > (byte) 0x8F)
              || (((byte1 << 28) + (byte2 - (byte) 0x90)) >> 30) != 0
              || isNotTrailingByte(byte3)
              || isNotTrailingByte(byte4)) {
        throw new IllegalArgumentException("Invalid UTF-8");
      }
      int codepoint = ((byte1 & 0x07) << 18)
                          | (trailingByteValue(byte2) << 12)
                          | (trailingByteValue(byte3) << 6)
                          | trailingByteValue(byte4);
      resultArr[resultPos] = DecodeUtil.highSurrogate(codepoint);
      resultArr[resultPos + 1] = DecodeUtil.lowSurrogate(codepoint);
    }

    /**
     * Returns whether the byte is not a valid continuation of the form '10XXXXXX'.
     */
    private static boolean isNotTrailingByte(byte b) {
      return b > (byte) 0xBF;
    }

    /**
     * Returns the actual value of the trailing byte (removes the prefix '10') for composition.
     */
    private static int trailingByteValue(byte b) {
      return b & 0x3F;
    }

    private static char highSurrogate(int codePoint) {
      return (char) ((MIN_HIGH_SURROGATE - (MIN_SUPPLEMENTARY_CODE_POINT >>> 10))
                         + (codePoint >>> 10));
    }

    private static char lowSurrogate(int codePoint) {
      return (char) (MIN_LOW_SURROGATE + (codePoint & 0x3ff));
    }
  }

  // These UTF-8 handling methods are copied from Guava's Utf8Unsafe class with a modification to throw
  // a protocol buffer local exception. This exception is then caught in CodedOutputStream so it can
  // fallback to more lenient behavior.
  static class UnpairedSurrogateException extends IllegalArgumentException {
    UnpairedSurrogateException(int index, int length) {
      super("Unpaired surrogate at index " + index + " of " + length);
    }
  }
}
