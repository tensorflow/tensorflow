/*
 * Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

package org.tensorflow.lite.support.image;

import java.nio.ByteBuffer;
import org.tensorflow.lite.DataType;

/**
 * Minimal TensorImage implementation for testing FastImageProcessor.
 * This is a simplified version for testing purposes.
 */
public class TensorImage {
  private ByteBuffer buffer;
  private DataType dataType;
  private int width;
  private int height;
  private int channels;
  
  private TensorImage(DataType dataType) {
    this.dataType = dataType;
  }
  
  public static TensorImage createFrom(ByteBuffer buffer, DataType dataType) {
    TensorImage image = new TensorImage(dataType);
    image.buffer = buffer;
    return image;
  }
  
  public void load(ByteBuffer buffer) {
    this.buffer = buffer;
  }
  
  public void load(ByteBuffer buffer, int width, int height, int channels) {
    this.buffer = buffer;
    this.width = width;
    this.height = height;
    this.channels = channels;
  }
  
  public ByteBuffer getBuffer() {
    return buffer;
  }
  
  public int getWidth() {
    return width;
  }
  
  public int getHeight() {
    return height;
  }
  
  public int getChannels() {
    return channels;
  }
  
  public DataType getDataType() {
    return dataType;
  }
}

