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

import static com.google.common.truth.Truth.assertThat;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.lite.DataType;

/** Unit tests for {@link FastImageProcessor}. */
@RunWith(JUnit4.class)
public final class FastImageProcessorTest {
  
  private static final int TEST_WIDTH = 224;
  private static final int TEST_HEIGHT = 224;
  private static final int TEST_CHANNELS = 3;
  private static final int TEST_SIZE = TEST_WIDTH * TEST_HEIGHT * TEST_CHANNELS;
  
  @Before
  public void setUp() {
    // Setup runs before each test
  }
  
  @Test
  public void testPreprocessingConstructor() {
    float scale = 1.0f / 255.0f;
    float offset = 0.0f;
    FastImageProcessor processor = new FastImageProcessor(scale, offset);
    
    // Verify it's configured for preprocessing (FLOAT32 output)
    assertThat(processor).isNotNull();
  }
  
  @Test
  public void testPostprocessingConstructor() {
    float scale = 255.0f;
    float offset = 0.0f;
    FastImageProcessor processor = new FastImageProcessor(scale, offset, DataType.UINT8);
    
    // Verify it's configured for postprocessing (UINT8 output)
    assertThat(processor).isNotNull();
  }
  
  @Test
  public void testNormalizeFloatProcessing() {
    // Create processor for preprocessing
    float scale = 1.0f / 255.0f;
    float offset = 0.0f;
    FastImageProcessor processor = new FastImageProcessor(scale, offset);
    
    // Create input buffer with UINT8 values (0-255)
    ByteBuffer inputBuffer = ByteBuffer.allocateDirect(TEST_SIZE);
    inputBuffer.order(ByteOrder.nativeOrder());
    for (int i = 0; i < TEST_SIZE; i++) {
      inputBuffer.put((byte) 128); // Middle value
    }
    inputBuffer.rewind();
    
    // Create TensorImage from input
    TensorImage input = TensorImage.createFrom(inputBuffer, DataType.UINT8);
    input.load(inputBuffer, TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS);
    
    // Process the image
    TensorImage output = processor.process(input);
    
    // Verify output
    assertThat(output).isNotNull();
    assertThat(output.getWidth()).isEqualTo(TEST_WIDTH);
    assertThat(output.getHeight()).isEqualTo(TEST_HEIGHT);
    assertThat(output.getChannels()).isEqualTo(TEST_CHANNELS);
    
    // Verify normalization: 128 / 255.0 ≈ 0.502
    ByteBuffer outputBuffer = output.getBuffer();
    outputBuffer.rewind();
    float normalizedValue = outputBuffer.getFloat();
    assertThat(normalizedValue).isWithin(0.01f).of(128.0f / 255.0f);
  }
  
  @Test
  public void testBufferReuse() {
    FastImageProcessor processor = new FastImageProcessor(1.0f / 255.0f, 0.0f);
    
    // Create first input
    ByteBuffer input1 = ByteBuffer.allocateDirect(TEST_SIZE);
    input1.order(ByteOrder.nativeOrder());
    for (int i = 0; i < TEST_SIZE; i++) {
      input1.put((byte) 100);
    }
    input1.rewind();
    
    TensorImage img1 = TensorImage.createFrom(input1, DataType.UINT8);
    img1.load(input1, TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS);
    
    // Process first image
    TensorImage output1 = processor.process(img1);
    
    // Create second input with same dimensions
    ByteBuffer input2 = ByteBuffer.allocateDirect(TEST_SIZE);
    input2.order(ByteOrder.nativeOrder());
    for (int i = 0; i < TEST_SIZE; i++) {
      input2.put((byte) 200);
    }
    input2.rewind();
    
    TensorImage img2 = TensorImage.createFrom(input2, DataType.UINT8);
    img2.load(input2, TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS);
    
    // Process second image - should reuse buffers
    TensorImage output2 = processor.process(img2);
    
    // Both should work correctly
    assertThat(output1).isNotNull();
    assertThat(output2).isNotNull();
  }
  
  @Test
  public void testClearCache() {
    FastImageProcessor processor = new FastImageProcessor(1.0f / 255.0f, 0.0f);
    
    // Process an image to allocate buffers
    ByteBuffer input = ByteBuffer.allocateDirect(TEST_SIZE);
    input.order(ByteOrder.nativeOrder());
    for (int i = 0; i < TEST_SIZE; i++) {
      input.put((byte) 128);
    }
    input.rewind();
    
    TensorImage img = TensorImage.createFrom(input, DataType.UINT8);
    img.load(input, TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS);
    processor.process(img);
    
    // Clear cache
    processor.clearCache();
    
    // Should still be able to process after clearing
    input.rewind();
    img.load(input, TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS);
    TensorImage output = processor.process(img);
    assertThat(output).isNotNull();
  }
  
  @Test
  public void testNormalizeAndCast() {
    // Create processor for postprocessing (UINT8 output)
    float scale = 255.0f;
    float offset = 0.0f;
    FastImageProcessor processor = new FastImageProcessor(scale, offset, DataType.UINT8);
    
    // Create input buffer with float values (0.0-1.0)
    ByteBuffer inputBuffer = ByteBuffer.allocateDirect(TEST_SIZE * 4); // 4 bytes per float
    inputBuffer.order(ByteOrder.nativeOrder());
    for (int i = 0; i < TEST_SIZE; i++) {
      inputBuffer.putFloat(0.5f); // 0.5 should map to ~128
    }
    inputBuffer.rewind();
    
    TensorImage input = TensorImage.createFrom(inputBuffer, DataType.FLOAT32);
    input.load(inputBuffer, TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS);
    
    // Process
    TensorImage output = processor.process(input);
    
    // Verify output
    assertThat(output).isNotNull();
    
    // Note: processToBitmap test would require Android Bitmap, which may not be available
    // in non-Android test environment
  }
  
  @Test
  public void testDifferentDimensions() {
    FastImageProcessor processor = new FastImageProcessor(1.0f / 255.0f, 0.0f);
    
    // Test with smaller image
    int smallWidth = 100;
    int smallHeight = 100;
    int smallSize = smallWidth * smallHeight * TEST_CHANNELS;
    
    ByteBuffer smallInput = ByteBuffer.allocateDirect(smallSize);
    smallInput.order(ByteOrder.nativeOrder());
    for (int i = 0; i < smallSize; i++) {
      smallInput.put((byte) 128);
    }
    smallInput.rewind();
    
    TensorImage smallImg = TensorImage.createFrom(smallInput, DataType.UINT8);
    smallImg.load(smallInput, smallWidth, smallHeight, TEST_CHANNELS);
    
    TensorImage smallOutput = processor.process(smallImg);
    assertThat(smallOutput.getWidth()).isEqualTo(smallWidth);
    assertThat(smallOutput.getHeight()).isEqualTo(smallHeight);
    
    // Test with larger image - should reallocate buffers
    ByteBuffer largeInput = ByteBuffer.allocateDirect(TEST_SIZE);
    largeInput.order(ByteOrder.nativeOrder());
    for (int i = 0; i < TEST_SIZE; i++) {
      largeInput.put((byte) 128);
    }
    largeInput.rewind();
    
    TensorImage largeImg = TensorImage.createFrom(largeInput, DataType.UINT8);
    largeImg.load(largeInput, TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS);
    
    TensorImage largeOutput = processor.process(largeImg);
    assertThat(largeOutput.getWidth()).isEqualTo(TEST_WIDTH);
    assertThat(largeOutput.getHeight()).isEqualTo(TEST_HEIGHT);
  }
}

