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

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.zip.ZipException;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Tensor.QuantizationParams;
import org.tensorflow.lite.schema.Tensor;
import org.tensorflow.lite.support.metadata.schema.ModelMetadata;
import org.tensorflow.lite.support.metadata.schema.TensorMetadata;

/**
 * Loads metadata from TFLite Model FlatBuffer.
 *
 * <p>TFLite Model FlatBuffer can be generated using the <a
 * href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs">TFLite
 * Model schema file.</a>
 *
 * <p>Some models contain a TFLite Metadata Flatbuffer, which records more information about what
 * the model does and how to interprete the model. TFLite Metadata Flatbuffer can be generated using
 * the <a
 * href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/metadata_schema.fbs">TFLite
 * Metadata schema file.</a>
 *
 * <p>It is allowed to pass in a model FlatBuffer without TFLite metadata. However, invoking methods
 * that read from TFLite metadata will cause runtime errors.
 *
 * <p>Similarly, it is allowed to pass in a model FlatBuffer without associated files. However,
 * invoking methods that read the associated files will cause runtime errors.
 *
 * <p>Though TFLite model FlatBuffer supports multiple subgraphs, TFLite Interpreter only supports a
 * single subgraph so far. See the <a
 * href="https://www.tensorflow.org/lite/convert/cmdline_examples#specifying_subgraphs">instruction
 * of how to specify subgraph during convertion for more information.</a> Therefore, {@link
 * MetadataExtractor} omits subgraph index as an input in its methods.
 */
public class MetadataExtractor {
  /** The helper class to load metadata from TFLite model FlatBuffer. */
  private final ModelInfo modelInfo;

  /** The helper class to load metadata from TFLite metadata FlatBuffer. */
  @Nullable private final ModelMetadataInfo metadataInfo;

  /** The handler to load associated files through zip. */
  @Nullable private final ZipFile zipFile;

  /**
   * Creates a {@link MetadataExtractor} with TFLite model FlatBuffer.
   *
   * @param buffer the TFLite model FlatBuffer
   * @throws IllegalArgumentException if the number of input or output tensors in the model does not
   *     match that in the metadata
   * @throws IOException if an error occurs while reading the model as a Zip file
   */
  public MetadataExtractor(ByteBuffer buffer) throws IOException {
    modelInfo = new ModelInfo(buffer);
    ByteBuffer metadataBuffer = modelInfo.getMetadataBuffer();
    if (metadataBuffer != null) {
      metadataInfo = new ModelMetadataInfo(metadataBuffer);
      checkArgument(
          modelInfo.getInputTensorCount() == metadataInfo.getInputTensorCount(),
          String.format(
              "The number of input tensors in the model is %d. The number of input tensors that"
                  + " recorded in the metadata is %d. These two values does not match.",
              modelInfo.getInputTensorCount(), metadataInfo.getInputTensorCount()));
      checkArgument(
          modelInfo.getOutputTensorCount() == metadataInfo.getOutputTensorCount(),
          String.format(
              "The number of output tensors in the model is %d. The number of output tensors that"
                  + " recorded in the metadata is %d. These two values does not match.",
              modelInfo.getOutputTensorCount(), metadataInfo.getOutputTensorCount()));
    } else {
      // It is allowed to pass in a model FlatBuffer without TFLite metadata. However, invoking
      // methods that read from TFLite metadata will cause runtime errors.
      metadataInfo = null;
    }

    zipFile = createZipFile(buffer);
  }

  /** Returns {@code true} if the model has metadata. Otherwise, returns {@code false}. */
  public Boolean hasMetadata() {
    return metadataInfo != null;
  }

  /**
   * Gets the packed associated file with the specified {@code fileName}.
   *
   * @param fileName the name of the associated file
   * @return the raw input stream containing specified file
   * @throws IllegalStateException if the model is not a zip file
   * @throws IllegalArgumentException if the specified file does not exist in the model
   */
  public InputStream getAssociatedFile(String fileName) {
    assertZipFile();
    return zipFile.getRawInputStream(fileName);
  }

  /** Gets the count of input tensors in the model. */
  public int getInputTensorCount() {
    return modelInfo.getInputTensorCount();
  }

  /**
   * Gets the metadata for the input tensor specified by {@code inputIndex}.
   *
   * @param inputIndex the index of the desired input tensor
   * @throws IllegalStateException if this model does not contain model metadata
   */
  @Nullable
  public TensorMetadata getInputTensorMetadata(int inputIndex) {
    assertMetadataInfo();
    return metadataInfo.getInputTensorMetadata(inputIndex);
  }

  /**
   * Gets the quantization parameters for the input tensor specified by {@code inputIndex}.
   *
   * @param inputIndex the index of the desired input tensor
   */
  public QuantizationParams getInputTensorQuantizationParams(int inputIndex) {
    Tensor tensor = modelInfo.getInputTensor(inputIndex);
    return modelInfo.getQuantizationParams(tensor);
  }

  /**
   * Gets the shape of the input tensor with {@code inputIndex}.
   *
   * @param inputIndex the index of the desired input tensor
   */
  public int[] getInputTensorShape(int inputIndex) {
    return modelInfo.getInputTensorShape(inputIndex);
  }

  /**
   * Gets the {@link DataType} of the input tensor with {@code inputIndex}.
   *
   * @param inputIndex the index of the desired input tensor
   */
  public DataType getInputTensorType(int inputIndex) {
    return modelInfo.getInputTensorType(inputIndex);
  }

  /**
   * Gets the root handler for the model metadata.
   *
   * @throws IllegalStateException if this model does not contain model metadata
   */
  public ModelMetadata getModelMetadata() {
    assertMetadataInfo();
    return metadataInfo.getModelMetadata();
  }

  /** Gets the count of output tensors in the model. */
  public int getOutputTensorCount() {
    return modelInfo.getOutputTensorCount();
  }

  /**
   * Gets the metadata for the output tensor specified by {@code outputIndex}.
   *
   * @param outputIndex the index of the desired output tensor
   * @throws IllegalStateException if this model does not contain model metadata
   */
  @Nullable
  public TensorMetadata getOutputTensorMetadata(int outputIndex) {
    assertMetadataInfo();
    return metadataInfo.getOutputTensorMetadata(outputIndex);
  }

  /**
   * Gets the quantization parameters for the output tensor specified by {@code outputIndex}.
   *
   * @param outputIndex the index of the desired output tensor
   */
  public QuantizationParams getOutputTensorQuantizationParams(int outputIndex) {
    Tensor tensor = modelInfo.getOutputTensor(outputIndex);
    return modelInfo.getQuantizationParams(tensor);
  }

  /**
   * Gets the shape of the output tensor with {@code outputIndex}.
   *
   * @param outputIndex the index of the desired output tensor
   */
  public int[] getOutputTensorShape(int outputIndex) {
    return modelInfo.getOutputTensorShape(outputIndex);
  }

  /**
   * Gets the {@link DataType} of the output tensor with {@code outputIndex}.
   *
   * @param outputIndex the index of the desired output tensor
   */
  public DataType getOutputTensorType(int outputIndex) {
    return modelInfo.getOutputTensorType(outputIndex);
  }

  /**
   * Asserts if {@link metdadataInfo} is not initialized. Some models may not have metadata and this
   * is allowed. However, invoking methods that reads the metadata is not allowed.
   *
   * @throws IllegalStateException if this model does not contain model metadata
   */
  private void assertMetadataInfo() {
    if (metadataInfo == null) {
      throw new IllegalStateException("This model does not contain model metadata.");
    }
  }

  /**
   * Asserts if {@link #zipFile} is not initialized. Some models may not have associated files, thus
   * are not Zip files. This is allowed. However, invoking methods that reads those associated files
   * is not allowed.
   *
   * @throws IllegalStateException if this model is not a Zip file
   */
  private void assertZipFile() {
    if (zipFile == null) {
      throw new IllegalStateException(
          "This model does not contain associated files, and is not a Zip file.");
    }
  }

  /**
   * Creates a Zip file handler to read the associated files. If the model is not a zip file, i.e.
   * it does not have associated files, return a null handler.
   *
   * @param buffer the TFLite model FlatBuffer
   * @throws IOException if an error occurs while reading the model as a Zip file
   */
  @Nullable
  private static ZipFile createZipFile(ByteBuffer buffer) throws IOException {
    try {
      // Creates the handler to hold the associated files through the Zip.
      ByteBufferChannel byteBufferChannel = new ByteBufferChannel(buffer);
      return ZipFile.createFrom(byteBufferChannel);
    } catch (ZipException e) {
      // Some models may not have associate files. Therefore, Those models are not zip files.
      // However, invoking methods that read associated files later will lead into errors.
      return null;
    }
  }
}
