/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.support.metadata;

import static java.lang.Math.min;
import static org.tensorflow.lite.support.metadata.Preconditions.checkArgument;
import static org.tensorflow.lite.support.metadata.Preconditions.checkNotNull;

import java.nio.ByteBuffer;
import java.nio.channels.NonWritableChannelException;

/** Implements the {@link SeekableByteChannelCompat} on top of {@link ByteBuffer}. */
final class ByteBufferChannel implements SeekableByteChannelCompat {

  /** The ByteBuffer that holds the data. */
  private final ByteBuffer buffer;

  /**
   * Creates a {@link ByteBufferChannel} that wraps a {@link ByteBuffer}.
   *
   * @param buffer the {@link ByteBuffer} that backs this {@link ByteBufferChannel}
   * @throws NullPointerException if {@code buffer} is null
   */
  public ByteBufferChannel(ByteBuffer buffer) {
    checkNotNull(buffer, "The ByteBuffer cannot be null.");
    this.buffer = buffer;
  }

  @Override
  public void close() {}

  @Override
  public boolean isOpen() {
    return true;
  }

  @Override
  public long position() {
    return buffer.position();
  }

  /**
   * Sets this channel's position.
   *
   * @param newPosition the new position, a non-negative integer counting the number of bytes from
   *     the beginning of the entity
   * @return this channel
   * @throws IllegalArgumentException if the new position is negative, or greater than the size of
   *     the underlying {@link ByteBuffer}, or greater than Integer.MAX_VALUE
   */
  @Override
  public synchronized ByteBufferChannel position(long newPosition) {
    checkArgument(
        (newPosition >= 0 && newPosition <= Integer.MAX_VALUE),
        "The new position should be non-negative and be less than Integer.MAX_VALUE.");
    buffer.position((int) newPosition);
    return this;
  }

  /**
   * {@inheritDoc}
   *
   * <p>Bytes are read starting at this channel's current position, and then the position is updated
   * with the number of bytes actually read. Otherwise this method behaves exactly as specified in
   * the {@link ReadableByteChannel} interface.
   */
  @Override
  public synchronized int read(ByteBuffer dst) {
    if (buffer.remaining() == 0) {
      return -1;
    }

    int count = min(dst.remaining(), buffer.remaining());
    if (count > 0) {
      ByteBuffer tempBuffer = buffer.slice();
      tempBuffer.order(buffer.order()).limit(count);
      dst.put(tempBuffer);
      buffer.position(buffer.position() + count);
    }
    return count;
  }

  @Override
  public long size() {
    return buffer.limit();
  }

  @Override
  public synchronized ByteBufferChannel truncate(long size) {
    checkArgument(
        (size >= 0 && size <= Integer.MAX_VALUE),
        "The new size should be non-negative and be less than Integer.MAX_VALUE.");

    if (size < buffer.limit()) {
      buffer.limit((int) size);
      if (buffer.position() > size) {
        buffer.position((int) size);
      }
    }
    return this;
  }

  @Override
  public synchronized int write(ByteBuffer src) {
    if (buffer.isReadOnly()) {
      throw new NonWritableChannelException();
    }

    int count = min(src.remaining(), buffer.remaining());
    if (count > 0) {
      ByteBuffer tempBuffer = src.slice();
      tempBuffer.order(buffer.order()).limit(count);
      buffer.put(tempBuffer);
    }
    return count;
  }
}
