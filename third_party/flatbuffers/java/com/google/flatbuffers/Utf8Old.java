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
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.CoderResult;
import java.nio.charset.StandardCharsets;

/**
 * This class implements the Utf8 API using the Java Utf8 encoder. Use
 * Utf8.setDefault(new Utf8Old()); to use it.
 */
public class Utf8Old extends Utf8 {

  private static class Cache {
    final CharsetEncoder encoder;
    final CharsetDecoder decoder;
    CharSequence lastInput = null;
    ByteBuffer lastOutput = null;

    Cache() {
      encoder = StandardCharsets.UTF_8.newEncoder();
      decoder = StandardCharsets.UTF_8.newDecoder();
    }
  }

  private static final ThreadLocal<Cache> CACHE =
      ThreadLocal.withInitial(() -> new Cache());

  // Play some games so that the old encoder doesn't pay twice for computing
  // the length of the encoded string.

  @Override
  public int encodedLength(CharSequence in) {
    final Cache cache = CACHE.get();
    int estimated = (int) (in.length() * cache.encoder.maxBytesPerChar());
    if (cache.lastOutput == null || cache.lastOutput.capacity() < estimated) {
      cache.lastOutput = ByteBuffer.allocate(Math.max(128, estimated));
    }
    cache.lastOutput.clear();
    cache.lastInput = in;
    CharBuffer wrap = (in instanceof CharBuffer) ?
                          (CharBuffer) in : CharBuffer.wrap(in);
    CoderResult result = cache.encoder.encode(wrap, cache.lastOutput, true);
    if (result.isError()) {
      try {
        result.throwException();
      } catch (CharacterCodingException e) {
        throw new IllegalArgumentException("bad character encoding", e);
      }
    }
    cache.lastOutput.flip();
    return cache.lastOutput.remaining();
  }

  @Override
  public void encodeUtf8(CharSequence in, ByteBuffer out) {
    final Cache cache = CACHE.get();
    if (cache.lastInput != in) {
      // Update the lastOutput to match our input, although flatbuffer should
      // never take this branch.
      encodedLength(in);
    }
    out.put(cache.lastOutput);
  }

  @Override
  public String decodeUtf8(ByteBuffer buffer, int offset, int length) {
    CharsetDecoder decoder = CACHE.get().decoder;
    decoder.reset();
    buffer = buffer.duplicate();
    buffer.position(offset);
    buffer.limit(offset + length);
    try {
      CharBuffer result = decoder.decode(buffer);
      return result.toString();
    } catch (CharacterCodingException e) {
      throw new IllegalArgumentException("Bad encoding", e);
    }
  }
}
