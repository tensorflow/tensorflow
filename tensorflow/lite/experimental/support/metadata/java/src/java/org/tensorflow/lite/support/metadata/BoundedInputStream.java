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

import static org.tensorflow.lite.support.metadata.Preconditions.checkArgument;
import static org.tensorflow.lite.support.metadata.Preconditions.checkElementIndex;
import static org.tensorflow.lite.support.metadata.Preconditions.checkNotNull;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

/**
 * An {@link InputStream} that wraps a section of a {@link SeekableByteChannelCompat}.
 *
 * <p><b>WARNING:</b> Similar as {@link InputStream}, instances of an {@link BoundedInputStream} are
 * <b>not</b> thread-safe. If multiple threads concurrently reading from the same {@link
 * BoundedInputStream}, it must be synchronized externally. Also, if multiple instances of {@link
 * BoundedInputStream} are created on the same {@link SeekableByteChannelCompat}, it must be
 * synchronized as well.
 */
final class BoundedInputStream extends InputStream {
  private final ByteBuffer singleByteBuffer = ByteBuffer.allocate(1);
  private final long end; // The valid data for the stream is between [start, end).
  private long position;
  private final SeekableByteChannelCompat channel;

  /**
   * Creates a {@link BoundedInputStream} with a {@link SeekableByteChannelCompat}.
   *
   * @param channel the {@link SeekableByteChannelCompat} that backs up this {@link
   *     BoundedInputStream}
   * @param start the starting position of this {@link BoundedInputStream} in the given {@link
   *     SeekableByteChannelCompat}
   * @param remaining the length of this {@link BoundedInputStream}
   * @throws IllegalArgumentException if {@code start} or {@code remaining} is negative
   */
  BoundedInputStream(SeekableByteChannelCompat channel, long start, long remaining) {
    checkArgument(
        remaining >= 0 && start >= 0,
        String.format("Invalid length of stream at offset=%d, length=%d", start, remaining));

    end = start + remaining;
    this.channel = channel;
    position = start;
  }

  @Override
  public int available() throws IOException {
    return (int) (Math.min(end, channel.size()) - position);
  }

  @Override
  public int read() throws IOException {
    if (position >= end) {
      return -1;
    }

    singleByteBuffer.rewind();
    int count = read(position, singleByteBuffer);
    if (count < 0) {
      return count;
    }

    position++;
    return singleByteBuffer.get() & 0xff;
  }

  @Override
  public int read(byte[] b, int off, int len) throws IOException {
    checkNotNull(b);
    checkElementIndex(off, b.length, "The start offset");
    checkElementIndex(len, b.length - off + 1, "The maximumn number of bytes to read");

    if (len == 0) {
      return 0;
    }

    if (len > end - position) {
      if (position >= end) {
        return -1;
      }
      len = (int) (end - position);
    }

    ByteBuffer buf = ByteBuffer.wrap(b, off, len);
    int count = read(position, buf);
    if (count > 0) {
      position += count;
    }
    return count;
  }

  private int read(long position, ByteBuffer buf) throws IOException {
    int count;
    synchronized (channel) {
      channel.position(position);
      count = channel.read(buf);
    }
    buf.flip();
    return count;
  }
}
