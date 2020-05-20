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
import static org.tensorflow.lite.support.metadata.Preconditions.checkNotNull;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Tensor.QuantizationParams;
import org.tensorflow.lite.schema.Buffer;
import org.tensorflow.lite.schema.Metadata;
import org.tensorflow.lite.schema.Model;
import org.tensorflow.lite.schema.QuantizationParameters;
import org.tensorflow.lite.schema.SubGraph;
import org.tensorflow.lite.schema.Tensor;
import org.tensorflow.lite.schema.TensorType;

/** Extracts model information out of TFLite model FLatBuffer. */
final class ModelInfo {
  /** The model that is loaded from TFLite model FlatBuffer. */
  private final Model model;

  /** A list of input tensors. */
  private final List</* @Nullable */ Tensor> inputTensors;

  /** A list of output tensors. */
  private final List</* @Nullable */ Tensor> outputTensors;

  /** Identifier of the TFLite model metadata in the Metadata array. */
  static final String METADATA_FIELD_NAME = "TFLITE_METADATA";

  /** Maps from TensorType in TFlite FlatBuffer to {@link DataType} in Java. */
  private final Map<Byte, DataType> tensorTypeToDataTypeMap;

  /**
   * Creates a {@link ModelInfo} with the model FlatBuffer, {@code buffer}.
   *
   * <p>Though TFLite model FlatBuffer supports multiple subgraphs, TFLite Interpreter only supports
   * single subgraph so far. See the <a
   * href="https://www.tensorflow.org/lite/convert/cmdline_examples#specifying_subgraphs">instruction
   * of how to specify subgraph during convertion for more information.</a> Therefore, all methods
   * in {@link ModelInfo} retrieves metadata of the first subgrpah as default.
   *
   * @param buffer the TFLite model FlatBuffer
   * @throws NullPointerException if {@code buffer} is null
   * @throws IllegalArgumentException if the model does not contain any subgraph, or the model does
   *     not contain the expected identifier
   */
  ModelInfo(ByteBuffer buffer) {
    assertTFLiteModel(buffer);

    model = Model.getRootAsModel(buffer);
    checkArgument(model.subgraphsLength() > 0, "The model does not contain any subgraph.");

    inputTensors = getInputTensors(model);
    outputTensors = getOutputTensors(model);
    tensorTypeToDataTypeMap = createTensorTypeToDataTypeMap();
  }

  /**
   * Gets the input tensor with {@code inputIndex}.
   *
   * @param inputIndex The index of the desired input tensor.
   * @throws IllegalArgumentException if the inputIndex specified is invalid.
   */
  @Nullable
  Tensor getInputTensor(int inputIndex) {
    checkArgument(
        inputIndex >= 0 && inputIndex < inputTensors.size(),
        "The inputIndex specified is invalid.");
    return inputTensors.get(inputIndex);
  }

  int getInputTensorCount() {
    return inputTensors.size();
  }

  /**
   * Gets shape of the input tensor with {@code inputIndex}.
   *
   * @param inputIndex The index of the desired intput tensor.
   */
  int[] getInputTensorShape(int inputIndex) {
    Tensor tensor = getInputTensor(inputIndex);
    return getShape(tensor);
  }

  /**
   * Gets {@link DataType} of the input tensor with {@code inputIndex}.
   *
   * @param inputIndex The index of the desired intput tensor.
   */
  DataType getInputTensorType(int inputIndex) {
    Tensor tensor = getInputTensor(inputIndex);
    return getDataType(tensor.type());
  }

  /** Gets the metadata FlatBuffer from the model FlatBuffer. */
  @Nullable
  ByteBuffer getMetadataBuffer() {
    // Some models may not have metadata, and this is allowed.
    if (model.metadataLength() == 0) {
      return null;
    }

    for (int i = 0; i < model.metadataLength(); i++) {
      Metadata meta = model.metadata(i);
      if (METADATA_FIELD_NAME.equals(meta.name())) {
        long bufferIndex = meta.buffer();
        Buffer metadataBuf = model.buffers((int) bufferIndex);
        return metadataBuf.dataAsByteBuffer();
      }
    }
    return null;
  }

  /**
   * Gets the output tensor with {@code outputIndex}.
   *
   * @param outputIndex The index of the desired outtput tensor.
   * @throws IllegalArgumentException if the outputIndex specified is invalid.
   */
  @Nullable
  Tensor getOutputTensor(int outputIndex) {
    checkArgument(
        outputIndex >= 0 && outputIndex < outputTensors.size(),
        "The outputIndex specified is invalid.");
    return outputTensors.get(outputIndex);
  }

  int getOutputTensorCount() {
    return outputTensors.size();
  }

  /**
   * Gets shape of the output tensor with {@code outputIndex}.
   *
   * @param outputIndex The index of the desired outtput tensor.
   */
  int[] getOutputTensorShape(int outputIndex) {
    Tensor tensor = getOutputTensor(outputIndex);
    return getShape(tensor);
  }

  /**
   * Gets {@link DataType} of the output tensor {@code outputIndex}.
   *
   * @param outputIndex The index of the desired outtput tensor.
   */
  DataType getOutputTensorType(int outputIndex) {
    Tensor tensor = getOutputTensor(outputIndex);
    return getDataType(tensor.type());
  }

  /**
   * Gets the quantization parameters of a tensor.
   *
   * <p>Only quantized tensors have valid {@code QuantizationParameters}. For tensor that are not
   * quantized, the values of scale and zero_point are both 0.
   *
   * @param tensor The tensor whoes quantization parameters is desired.
   * @throws NullPointerException if the tensor is null.
   * @throws IllegalArgumentException if {@code scale} and {@code zeroPoint} of the tensor's {@link
   *     QuantizationParameters} are not single values.
   */
  QuantizationParams getQuantizationParams(Tensor tensor) {
    checkNotNull(tensor, "Tensor cannot be null.");

    float scale;
    int zeroPoint;
    QuantizationParameters quantization = tensor.quantization();

    // Tensors that are not quantized do not have quantization parameters, which can be null when
    // being extracted from the flatbuffer.
    if (quantization == null) {
      scale = 0.0f;
      zeroPoint = 0;
      return new QuantizationParams(scale, zeroPoint);
    }

    // Tensors that are not quantized do not have quantization parameters.
    // quantization.scaleLength() and quantization.zeroPointLength() may both return 0.
    checkArgument(
        quantization.scaleLength() <= 1,
        "Input and output tensors do not support per-channel quantization.");
    checkArgument(
        quantization.zeroPointLength() <= 1,
        "Input and output tensors do not support per-channel quantization.");

    // For tensors that are not quantized, quantization.scale(0) and quantization.zeroPoint(0) will
    // both be the default value in flatbuffer, 0. This behavior is consistent with the TFlite C++
    // runtime.
    scale = quantization.scale(0);
    // zeroPoint is a long value in the schema, but an integer in the C++ runtime. Here we keep it
    // consistent with the C++ runtime.
    zeroPoint = (int) quantization.zeroPoint(0);

    return new QuantizationParams(scale, zeroPoint);
  }

  /**
   * Verifies if the buffer is a valid TFLite model.
   *
   * @param buffer the TFLite model flatbuffer
   * @throws NullPointerException if {@code buffer} is null.
   * @throws IllegalArgumentException if {@code buffer} does not contain the expected identifier
   */
  private static void assertTFLiteModel(ByteBuffer buffer) {
    checkNotNull(buffer, "Model flatbuffer cannot be null.");
    checkArgument(
        Model.ModelBufferHasIdentifier(buffer),
        "The identifier of the model is invalid. The buffer may not be a valid TFLite model"
            + " flatbuffer.");
  }

  private static Map<Byte, DataType> createTensorTypeToDataTypeMap() {
    Map<Byte, DataType> map = new HashMap<>();
    map.put(TensorType.FLOAT32, DataType.FLOAT32);
    map.put(TensorType.INT32, DataType.INT32);
    map.put(TensorType.UINT8, DataType.UINT8);
    map.put(TensorType.INT64, DataType.INT64);
    map.put(TensorType.STRING, DataType.STRING);
    return Collections.unmodifiableMap(map);
  }

  /**
   * Transforms from TensorType in TFlite FlatBuffer to {@link DataType} in Java.
   *
   * @param tensorType The tensor type to be converted.
   * @throws IllegalArgumentException if the tensor type is not supported.
   */
  private DataType getDataType(byte tensorType) {
    checkArgument(
        tensorTypeToDataTypeMap.containsKey(tensorType),
        String.format("Tensor type %d is not supported.", tensorType));
    return tensorTypeToDataTypeMap.get(tensorType);
  }

  /**
   * Gets the shape of a tensor.
   *
   * @param tensor The tensor whoes shape is desired.
   * @throws NullPointerException if the tensor is null.
   */
  private static int[] getShape(Tensor tensor) {
    checkNotNull(tensor, "Tensor cannot be null.");
    int shapeDim = tensor.shapeLength();
    int[] tensorShape = new int[shapeDim];
    for (int i = 0; i < shapeDim; i++) {
      tensorShape[i] = tensor.shape(i);
    }
    return tensorShape;
  }

  /** Gets input tensors from a model. */
  private static List<Tensor> getInputTensors(Model model) {
    // TFLite only support one subgraph currently.
    SubGraph subgraph = model.subgraphs(0);
    int tensorNum = subgraph.inputsLength();
    ArrayList<Tensor> inputTensors = new ArrayList<>(tensorNum);
    for (int i = 0; i < tensorNum; i++) {
      inputTensors.add(subgraph.tensors(subgraph.inputs(i)));
    }
    return Collections.unmodifiableList(inputTensors);
  }

  /** Gets output tensors from a model. */
  private static List<Tensor> getOutputTensors(Model model) {
    // TFLite only support one subgraph currently.
    SubGraph subgraph = model.subgraphs(0);
    int tensorNum = subgraph.outputsLength();
    ArrayList<Tensor> outputTensors = new ArrayList<>(tensorNum);
    for (int i = 0; i < tensorNum; i++) {
      outputTensors.add(subgraph.tensors(subgraph.outputs(i)));
    }
    return Collections.unmodifiableList(outputTensors);
  }
}
