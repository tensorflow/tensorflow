/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
using System;
using System.Runtime.InteropServices;

using TfLiteInterpreter = System.IntPtr;
using TfLiteInterpreterOptions = System.IntPtr;
using TfLiteModel = System.IntPtr;
using TfLiteTensor = System.IntPtr;

namespace TensorFlowLite
{
  /// <summary>
  /// Simple C# bindings for the experimental TensorFlowLite C API.
  /// </summary>
  public class Interpreter : IDisposable
  {
    private const string TensorFlowLibrary = "tensorflowlite_c";

    private TfLiteModel model;
    private TfLiteInterpreter interpreter;

    public Interpreter(byte[] modelData) {
      GCHandle modelDataHandle = GCHandle.Alloc(modelData, GCHandleType.Pinned);
      IntPtr modelDataPtr = modelDataHandle.AddrOfPinnedObject();
      model = TfLiteModelCreate(modelDataPtr, modelData.Length);
      if (model == IntPtr.Zero) throw new Exception("Failed to create TensorFlowLite Model");
      interpreter = TfLiteInterpreterCreate(model, /*options=*/IntPtr.Zero);
      if (interpreter == IntPtr.Zero) throw new Exception("Failed to create TensorFlowLite Interpreter");
    }

    ~Interpreter() {
      Dispose();
    }

    public void Dispose() {
      if (interpreter != IntPtr.Zero) TfLiteInterpreterDelete(interpreter);
      interpreter = IntPtr.Zero;
      if (model != IntPtr.Zero) TfLiteModelDelete(model);
      model = IntPtr.Zero;
    }

    public void Invoke() {
      ThrowIfError(TfLiteInterpreterInvoke(interpreter));
    }

    public int GetInputTensorCount() {
      return TfLiteInterpreterGetInputTensorCount(interpreter);
    }

    public void SetInputTensorData(int inputTensorIndex, Array inputTensorData) {
      GCHandle tensorDataHandle = GCHandle.Alloc(inputTensorData, GCHandleType.Pinned);
      IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
      TfLiteTensor tensor = TfLiteInterpreterGetInputTensor(interpreter, inputTensorIndex);
      ThrowIfError(TfLiteTensorCopyFromBuffer(
          tensor, tensorDataPtr, Buffer.ByteLength(inputTensorData)));
    }

    public void ResizeInputTensor(int inputTensorIndex, int[] inputTensorShape) {
      ThrowIfError(TfLiteInterpreterResizeInputTensor(
          interpreter, inputTensorIndex, inputTensorShape, inputTensorShape.Length));
    }

    public void AllocateTensors() {
      ThrowIfError(TfLiteInterpreterAllocateTensors(interpreter));
    }

    public int GetOutputTensorCount() {
      return TfLiteInterpreterGetOutputTensorCount(interpreter);
    }

    public void GetOutputTensorData(int outputTensorIndex, Array outputTensorData) {
      GCHandle tensorDataHandle = GCHandle.Alloc(outputTensorData, GCHandleType.Pinned);
      IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
      TfLiteTensor tensor = TfLiteInterpreterGetOutputTensor(interpreter, outputTensorIndex);
      ThrowIfError(TfLiteTensorCopyToBuffer(
          tensor, tensorDataPtr, Buffer.ByteLength(outputTensorData)));
    }

    private static void ThrowIfError(int resultCode) {
      if (resultCode != 0) throw new Exception("TensorFlowLite operation failed.");
    }

    #region Externs

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TfLiteInterpreter TfLiteModelCreate(IntPtr model_data, int model_size);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TfLiteInterpreter TfLiteModelDelete(TfLiteModel model);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TfLiteInterpreter TfLiteInterpreterCreate(
        TfLiteModel model,
        TfLiteInterpreterOptions optional_options);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe void TfLiteInterpreterDelete(TfLiteInterpreter interpreter);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TfLiteInterpreterGetInputTensorCount(
        TfLiteInterpreter interpreter);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TfLiteTensor TfLiteInterpreterGetInputTensor(
        TfLiteInterpreter interpreter,
        int input_index);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TfLiteInterpreterResizeInputTensor(
        TfLiteInterpreter interpreter,
        int input_index,
        int[] input_dims,
        int input_dims_size);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TfLiteInterpreterAllocateTensors(
        TfLiteInterpreter interpreter);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TfLiteInterpreterInvoke(TfLiteInterpreter interpreter);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TfLiteInterpreterGetOutputTensorCount(
        TfLiteInterpreter interpreter);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TfLiteTensor TfLiteInterpreterGetOutputTensor(
        TfLiteInterpreter interpreter,
        int output_index);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TfLiteTensorCopyFromBuffer(
        TfLiteTensor tensor,
        IntPtr input_data,
        int input_data_size);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TfLiteTensorCopyToBuffer(
        TfLiteTensor tensor,
        IntPtr output_data,
        int output_data_size);

    #endregion
  }
}
