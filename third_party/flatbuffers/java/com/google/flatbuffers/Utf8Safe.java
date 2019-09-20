// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package com.google.flatbuffers;

import java.nio.ByteBuffer;
import static java.lang.Character.MAX_SURROGATE;
import static java.lang.Character.MIN_SUPPLEMENTARY_CODE_POINT;
import static java.lang.Character.MIN_SURROGATE;
import static java.lang.Character.isSurrogatePair;
import static java.lang.Character.toCodePoint;

/**
 * A set of low-level, high-performance static utility methods related
 * to the UTF-8 character encoding.  This class has no dependencies
 * outside of the core JDK libraries.
 *
 * <p>There are several variants of UTF-8.  The one implemented by
 * this class is the restricted definition of UTF-8 introduced in
 * Unicode 3.1, which mandates the rejection of "overlong" byte
 * sequences as well as rejection of 3-byte surrogate codepoint byte
 * sequences.  Note that the UTF-8 decoder included in Oracle's JDK
 * has been modified to also reject "overlong" byte sequences, but (as
 * of 2011) still accepts 3-byte surrogate codepoint byte sequences.
 *
 * <p>The byte sequences considered valid by this class are exactly
 * those that can be roundtrip converted to Strings and back to bytes
 * using the UTF-8 charset, without loss: <pre> {@code
 * Arrays.equals(bytes, new String(bytes, Internal.UTF_8).getBytes(Internal.UTF_8))
 * }</pre>
 *
 * <p>See the Unicode Standard,</br>
 * Table 3-6. <em>UTF-8 Bit Distribution</em>,</br>
 * Table 3-7. <em>Well Formed UTF-8 Byte Sequences</em>.
 */
final public class Utf8Safe extends Utf8 {

  /**
   * Returns the number of bytes in the UTF-8-encoded form of {@code sequence}. For a string,
   * this method is equivalent to {@code string.getBytes(UTF_8).length}, but is more efficient in
   * both time and space.
   *
   * @throws IllegalArgumentException if {@code sequence} contains ill-formed UTF-16 (unpaired
   *     surrogates)
   */
  private static int computeEncodedLength(CharSequence sequence) {
    // Warning to maintainers: this implementation is highly optimized.
    int utf16Length = sequence.length();
    int utf8Length = utf16Length;
    int i = 0;

    // This loop optimizes for pure ASCII.
    while (i < utf16Length && sequence.charAt(i) < 0x80) {
      i++;
    }

    // This loop optimizes for chars less than 0x800.
    for (; i < utf16Length; i++) {
      char c = sequence.charAt(i);
      if (c < 0x800) {
        utf8Length += ((0x7f - c) >>> 31);  // branch free!
      } else {
        utf8Length += encodedLengthGeneral(sequence, i);
        break;
      }
    }

    if (utf8Length < utf16Length) {
      // Necessary and sufficient condition for overflow because of maximum 3x expansion
      throw new IllegalArgumentException("UTF-8 length does not fit in int: "
                                             + (utf8Length + (1L << 32)));
    }
    return utf8Length;
  }

  private static int encodedLengthGeneral(CharSequence sequence, int start) {
    int utf16Length = sequence.length();
    int utf8Length = 0;
    for (int i = start; i < utf16Length; i++) {
      char c = sequence.charAt(i);
      if (c < 0x800) {
        utf8Length += (0x7f - c) >>> 31; // branch free!
      } else {
        utf8Length += 2;
        // jdk7+: if (Character.isSurrogate(c)) {
        if (Character.MIN_SURROGATE <= c && c <= Character.MAX_SURROGATE) {
          // Check that we have a well-formed surrogate pair.
          int cp = Character.codePointAt(sequence, i);
          if (cp < MIN_SUPPLEMENTARY_CODE_POINT) {
            throw new Utf8Safe.UnpairedSurrogateException(i, utf16Length);
          }
          i++;
        }
      }
    }
    return utf8Length;
  }

  private static String decodeUtf8Array(byte[] bytes, int index, int size) {
    // Bitwise OR combines the sign bits so any negative value fails the check.
    if ((index | size | bytes.length - index - size) < 0) {
      throw new ArrayIndexOutOfBoundsException(
          String.format("buffer length=%d, index=%d, size=%d", bytes.length, index, size));
    }

    int offset = index;
    final int limit = offset + size;

    // The longest possible resulting String is the same as the number of input bytes, when it is
    // all ASCII. For other cases, this over-allocates and we will truncate in the end.
    char[] resultArr = new char[size];
    int resultPos = 0;

    // Optimize for 100% ASCII (Hotspot loves small simple top-level loops like this).
    // This simple loop stops when we encounter a byte >= 0x80 (i.e. non-ASCII).
    while (offset < limit) {
      byte b = bytes[offset];
      if (!DecodeUtil.isOneByte(b)) {
        break;
      }
      offset++;
      DecodeUtil.handleOneByte(b, resultArr, resultPos++);
    }

    while (offset < limit) {
      byte byte1 = bytes[offset++];
      if (DecodeUtil.isOneByte(byte1)) {
        DecodeUtil.handleOneByte(byte1, resultArr, resultPos++);
        // It's common for there to be multiple ASCII characters in a run mixed in, so add an
        // extra optimized loop to take care of these runs.
        while (offset < limit) {
          byte b = bytes[offset];
          if (!DecodeUtil.isOneByte(b)) {
            break;
          }
          offset++;
          DecodeUtil.handleOneByte(b, resultArr, resultPos++);
        }
      } else if (DecodeUtil.isTwoBytes(byte1)) {
        if (offset >= limit) {
          throw new IllegalArgumentException("Invalid UTF-8");
        }
        DecodeUtil.handleTwoBytes(byte1, /* byte2 */ bytes[offset++], resultArr, resultPos++);
      } else if (DecodeUtil.isThreeBytes(byte1)) {
        if (offset >= limit - 1) {
          throw new IllegalArgumentException("Invalid UTF-8");
        }
        DecodeUtil.handleThreeBytes(
            byte1,
            /* byte2 */ bytes[offset++],
            /* byte3 */ bytes[offset++],
            resultArr,
            resultPos++);
      } else {
        if (offset >= limit - 2) {
          throw new IllegalArgumentException("Invalid UTF-8");
        }
        DecodeUtil.handleFourBytes(
            byte1,
            /* byte2 */ bytes[offset++],
            /* byte3 */ bytes[offset++],
            /* byte4 */ bytes[offset++],
            resultArr,
            resultPos++);
        // 4-byte case requires two chars.
        resultPos++;
      }
    }

    return new String(resultArr, 0, resultPos);
  }

  private static String decodeUtf8Buffer(ByteBuffer buffer, int offset,
                                         int length) {
    // Bitwise OR combines the sign bits so any negative value fails the check.
    if ((offset | length | buffer.limit() - offset - length) < 0) {
      throw new ArrayIndexOutOfBoundsException(
          String.format("buffer limit=%d, index=%d, limit=%d", buffer.limit(),
              offset, length));
    }

    final int limit = offset + length;

    // The longest possible resulting String is the same as the number of input bytes, when it is
    // all ASCII. For other cases, this over-allocates and we will truncate in the end.
    char[] resultArr = new char[length];
    int resultPos = 0;

    // Optimize for 100% ASCII (Hotspot loves small simple top-level loops like this).
    // This simple loop stops when we encounter a byte >= 0x80 (i.e. non-ASCII).
    while (offset < limit) {
      byte b = buffer.get(offset);
      if (!DecodeUtil.isOneByte(b)) {
        break;
      }
      offset++;
      DecodeUtil.handleOneByte(b, resultArr, resultPos++);
    }

    while (offset < limit) {
      byte byte1 = buffer.get(offset++);
      if (DecodeUtil.isOneByte(byte1)) {
        DecodeUtil.handleOneByte(byte1, resultArr, resultPos++);
        // It's common for there to be multiple ASCII characters in a run mixed in, so add an
        // extra optimized loop to take care of these runs.
        while (offset < limit) {
          byte b = buffer.get(offset);
          if (!DecodeUtil.isOneByte(b)) {
            break;
          }
          offset++;
          DecodeUtil.handleOneByte(b, resultArr, resultPos++);
        }
      } else if (DecodeUtil.isTwoBytes(byte1)) {
        if (offset >= limit) {
          throw new IllegalArgumentException("Invalid UTF-8");
        }
        DecodeUtil.handleTwoBytes(
            byte1, /* byte2 */ buffer.get(offset++), resultArr, resultPos++);
      } else if (DecodeUtil.isThreeBytes(byte1)) {
        if (offset >= limit - 1) {
          throw new IllegalArgumentException("Invalid UTF-8");
        }
        DecodeUtil.handleThreeBytes(
            byte1,
            /* byte2 */ buffer.get(offset++),
            /* byte3 */ buffer.get(offset++),
            resultArr,
            resultPos++);
      } else {
        if (offset >= limit - 2) {
          throw new IllegalArgumentException("Invalid UTF-8");
        }
        DecodeUtil.handleFourBytes(
            byte1,
            /* byte2 */ buffer.get(offset++),
            /* byte3 */ buffer.get(offset++),
            /* byte4 */ buffer.get(offset++),
            resultArr,
            resultPos++);
        // 4-byte case requires two chars.
        resultPos++;
      }
    }

    return new String(resultArr, 0, resultPos);
  }

  @Override
  public int encodedLength(CharSequence in) {
    return computeEncodedLength(in);
  }

  /**
   * Decodes the given UTF-8 portion of the {@link ByteBuffer} into a {@link String}.
   *
   * @throws IllegalArgumentException if the input is not valid UTF-8.
   */
  @Override
  public String decodeUtf8(ByteBuffer buffer, int offset, int length)
      throws IllegalArgumentException {
    if (buffer.hasArray()) {
      return decodeUtf8Array(buffer.array(), buffer.arrayOffset() + offset, length);
    } else {
      return decodeUtf8Buffer(buffer, offset, length);
    }
  }


  private static void encodeUtf8Buffer(CharSequence in, ByteBuffer out) {
    final int inLength = in.length();
    int outIx = out.position();
    int inIx = 0;

    // Since ByteBuffer.putXXX() already checks boundaries for us, no need to explicitly check
    // access. Assume the buffer is big enough and let it handle the out of bounds exception
    // if it occurs.
    try {
      // Designed to take advantage of
      // https://wikis.oracle.com/display/HotSpotInternals/RangeCheckElimination
      for (char c; inIx < inLength && (c = in.charAt(inIx)) < 0x80; ++inIx) {
        out.put(outIx + inIx, (byte) c);
      }
      if (inIx == inLength) {
        // Successfully encoded the entire string.
        out.position(outIx + inIx);
        return;
      }

      outIx += inIx;
      for (char c; inIx < inLength; ++inIx, ++outIx) {
        c = in.charAt(inIx);
        if (c < 0x80) {
          // One byte (0xxx xxxx)
          out.put(outIx, (byte) c);
        } else if (c < 0x800) {
          // Two bytes (110x xxxx 10xx xxxx)

          // Benchmarks show put performs better than putShort here (for HotSpot).
          out.put(outIx++, (byte) (0xC0 | (c >>> 6)));
          out.put(outIx, (byte) (0x80 | (0x3F & c)));
        } else if (c < MIN_SURROGATE || MAX_SURROGATE < c) {
          // Three bytes (1110 xxxx 10xx xxxx 10xx xxxx)
          // Maximum single-char code point is 0xFFFF, 16 bits.

          // Benchmarks show put performs better than putShort here (for HotSpot).
          out.put(outIx++, (byte) (0xE0 | (c >>> 12)));
          out.put(outIx++, (byte) (0x80 | (0x3F & (c >>> 6))));
          out.put(outIx, (byte) (0x80 | (0x3F & c)));
        } else {
          // Four bytes (1111 xxxx 10xx xxxx 10xx xxxx 10xx xxxx)

          // Minimum code point represented by a surrogate pair is 0x10000, 17 bits, four UTF-8
          // bytes
          final char low;
          if (inIx + 1 == inLength || !isSurrogatePair(c, (low = in.charAt(++inIx)))) {
            throw new UnpairedSurrogateException(inIx, inLength);
          }
          // TODO(nathanmittler): Consider using putInt() to improve performance.
          int codePoint = toCodePoint(c, low);
          out.put(outIx++, (byte) ((0xF << 4) | (codePoint >>> 18)));
          out.put(outIx++, (byte) (0x80 | (0x3F & (codePoint >>> 12))));
          out.put(outIx++, (byte) (0x80 | (0x3F & (codePoint >>> 6))));
          out.put(outIx, (byte) (0x80 | (0x3F & codePoint)));
        }
      }

      // Successfully encoded the entire string.
      out.position(outIx);
    } catch (IndexOutOfBoundsException e) {
      // TODO(nathanmittler): Consider making the API throw IndexOutOfBoundsException instead.

      // If we failed in the outer ASCII loop, outIx will not have been updated. In this case,
      // use inIx to determine the bad write index.
      int badWriteIndex = out.position() + Math.max(inIx, outIx - out.position() + 1);
      throw new ArrayIndexOutOfBoundsException(
          "Failed writing " + in.charAt(inIx) + " at index " + badWriteIndex);
    }
  }

  private static int encodeUtf8Array(CharSequence in, byte[] out,
                                     int offset, int length) {
    int utf16Length = in.length();
    int j = offset;
    int i = 0;
    int limit = offset + length;
    // Designed to take advantage of
    // https://wikis.oracle.com/display/HotSpotInternals/RangeCheckElimination
    for (char c; i < utf16Length && i + j < limit && (c = in.charAt(i)) < 0x80; i++) {
      out[j + i] = (byte) c;
    }
    if (i == utf16Length) {
      return j + utf16Length;
    }
    j += i;
    for (char c; i < utf16Length; i++) {
      c = in.charAt(i);
      if (c < 0x80 && j < limit) {
        out[j++] = (byte) c;
      } else if (c < 0x800 && j <= limit - 2) { // 11 bits, two UTF-8 bytes
        out[j++] = (byte) ((0xF << 6) | (c >>> 6));
        out[j++] = (byte) (0x80 | (0x3F & c));
      } else if ((c < Character.MIN_SURROGATE || Character.MAX_SURROGATE < c) && j <= limit - 3) {
        // Maximum single-char code point is 0xFFFF, 16 bits, three UTF-8 bytes
        out[j++] = (byte) ((0xF << 5) | (c >>> 12));
        out[j++] = (byte) (0x80 | (0x3F & (c >>> 6)));
        out[j++] = (byte) (0x80 | (0x3F & c));
      } else if (j <= limit - 4) {
        // Minimum code point represented by a surrogate pair is 0x10000, 17 bits,
        // four UTF-8 bytes
        final char low;
        if (i + 1 == in.length()
                || !Character.isSurrogatePair(c, (low = in.charAt(++i)))) {
          throw new UnpairedSurrogateException((i - 1), utf16Length);
        }
        int codePoint = Character.toCodePoint(c, low);
        out[j++] = (byte) ((0xF << 4) | (codePoint >>> 18));
        out[j++] = (byte) (0x80 | (0x3F & (codePoint >>> 12)));
        out[j++] = (byte) (0x80 | (0x3F & (codePoint >>> 6)));
        out[j++] = (byte) (0x80 | (0x3F & codePoint));
      } else {
        // If we are surrogates and we're not a surrogate pair, always throw an
        // UnpairedSurrogateException instead of an ArrayOutOfBoundsException.
        if ((Character.MIN_SURROGATE <= c && c <= Character.MAX_SURROGATE)
                && (i + 1 == in.length()
                        || !Character.isSurrogatePair(c, in.charAt(i + 1)))) {
          throw new UnpairedSurrogateException(i, utf16Length);
        }
        throw new ArrayIndexOutOfBoundsException("Failed writing " + c + " at index " + j);
      }
    }
    return j;
  }

  /**
   * Encodes the given characters to the target {@link ByteBuffer} using UTF-8 encoding.
   *
   * <p>Selects an optimal algorithm based on the type of {@link ByteBuffer} (i.e. heap or direct)
   * and the capabilities of the platform.
   *
   * @param in the source string to be encoded
   * @param out the target buffer to receive the encoded string.
   */
  @Override
  public void encodeUtf8(CharSequence in, ByteBuffer out) {
    if (out.hasArray()) {
      int start = out.arrayOffset();
      int end = encodeUtf8Array(in, out.array(), start + out.position(),
          out.remaining());
      out.position(end - start);
    } else {
      encodeUtf8Buffer(in, out);
    }
  }

  // These UTF-8 handling methods are copied from Guava's Utf8Unsafe class with
  // a modification to throw a local exception. This exception can be caught
  // to fallback to more lenient behavior.
  static class UnpairedSurrogateException extends IllegalArgumentException {
    UnpairedSurrogateException(int index, int length) {
      super("Unpaired surrogate at index " + index + " of " + length);
    }
  }
}
