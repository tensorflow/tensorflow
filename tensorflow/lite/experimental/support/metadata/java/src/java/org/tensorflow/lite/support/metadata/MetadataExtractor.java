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
  // TODO(b/156539454): remove the hardcode versioning number and populate the version through
  // genrule.
  /** The version of the metadata parser that this {@link MetadataExtractor} library depends on. */
  public static final String METADATA_PARSER_VERSION = "1.0.1";

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

      // Prints warning message if the minimum parser version is not satisfied.
      if (!isMinimumParserVersionSatisfied()) {
        System.err.printf(
            "<Warning> Some fields in the metadata belong to a future schema. The minimum parser"
                + " version required is %s, but the version of the current metadata parser is %s",
            metadataInfo.getMininumParserVersion(), METADATA_PARSER_VERSION);
      }

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

  /**
   * Quantization parameters that corresponds to the table, {@code QuantizationParameters}, in the
   * <a
   * href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs">TFLite
   * Model schema file.</a>
   *
   * <p>Since per-channel quantization does not apply to input and output tensors, {@code scale} and
   * {@code zero_point} are both single values instead of arrays.
   *
   * <p>For tensor that are not quantized, the values of scale and zero_point are both 0.
   *
   * <p>Given a quantized value q, the corresponding float value f should be: <br>
   * f = scale * (q - zero_point) <br>
   */
  public static class QuantizationParams {
    /** The scale value used in quantization. */
    private final float scale;
    /** The zero point value used in quantization. */
    private final int zeroPoint;

    /**
     * Creates a {@link QuantizationParams} with {@code scale} and {@code zero_point}.
     *
     * @param scale The scale value used in quantization.
     * @param zeroPoint The zero point value used in quantization.
     */
    public QuantizationParams(final float scale, final int zeroPoint) {
      this.scale = scale;
      this.zeroPoint = zeroPoint;
    }

    /** Returns the scale value. */
    public float getScale() {
      return scale;
    }

    /** Returns the zero point value. */
    public int getZeroPoint() {
      return zeroPoint;
    }
  }

  /** Returns {@code true} if the model has metadata. Otherwise, returns {@code false}. */
  public boolean hasMetadata() {
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
   * Gets the {@link TensorType} of the input tensor with {@code inputIndex}.
   *
   * @param inputIndex the index of the desired input tensor
   */
  public byte getInputTensorType(int inputIndex) {
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
   * Gets the {@link TensorType} of the output tensor with {@code outputIndex}.
   *
   * @param outputIndex the index of the desired output tensor
   */
  public byte getOutputTensorType(int outputIndex) {
    return modelInfo.getOutputTensorType(outputIndex);
  }

  /**
   * Returns {@code true} if the minimum parser version required by the given metadata flatbuffer
   * precedes or equals to the version of the metadata parser that this MetadataExtractor library is
   * relying on. All fields in the metadata can be parsed correctly with this metadata extractor
   * library in this case. Otherwise, it returns {@code false}.
   *
   * <p>For example, assume the underlying metadata parser version is {@code 1.14.1},
   *
   * <ul>
   *   <li>it returns {@code true}, if the required minimum parser version is the same or older,
   *       such as {@code 1.14.1} or {@code 1.14.0}. Null version precedes all numeric versions,
   *       because some metadata flatbuffers are generated before the first versioned release; <br>
   *   <li>it returns {@code false}, if the required minimum parser version is newer, such as {@code
   *       1.14.2}.
   * </ul>
   */
  public final boolean isMinimumParserVersionSatisfied() {
    String minVersion = metadataInfo.getMininumParserVersion();
    if (minVersion == null) {
      return true;
    }
    return compareVersions(minVersion, METADATA_PARSER_VERSION) <= 0;
  }

  /**
   * Asserts if {@link #metadataInfo} is not initialized. Some models may not have metadata and this
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

  /**
   * Compares two semantic version numbers.
   *
   * <p>Examples of comparing two versions: <br>
   * {@code 1.9} precedes {@code 1.14}; <br>
   * {@code 1.14} precedes {@code 1.14.1}; <br>
   * {@code 1.14} and {@code 1.14.0} are euqal;
   *
   * @return the value {@code 0} if the two versions are equal; a value less than {@code 0} if
   *     {@code version1} precedes {@code version2}; a value greater than {@code 0} if {@code
   *     version2} precedes {@code version1}.
   */
  private static int compareVersions(String version1, String version2) {
    // Using String.split instead of the recommanded Guava Splitter because we've been avoiding
    // depending on other third party libraries in this project.
    String[] levels1 = version1.split("\\.", 0);
    String[] levels2 = version2.split("\\.", 0);

    int length = Math.max(levels1.length, levels2.length);
    for (int i = 0; i < length; i++) {
      Integer v1 = i < levels1.length ? Integer.parseInt(levels1[i]) : 0;
      Integer v2 = i < levels2.length ? Integer.parseInt(levels2[i]) : 0;
      int compare = v1.compareTo(v2);
      if (compare != 0) {
        return compare;
      }
    }

    return 0;
  }
}
