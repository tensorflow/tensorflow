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

using TFL_Interpreter = System.IntPtr;
using TFL_Tensor = System.IntPtr;

namespace TensorFlowLite
{
  /// <summary>
  /// Simple C# bindings for the experimental TensorFlowLite C API.
  /// </summary>
  public class Interpreter : IDisposable
  {
    private const string TensorFlowLibrary = "tensorflowlite_c";

    private TFL_Interpreter handle;

    public Interpreter(byte[] modelData) {
      GCHandle modelDataHandle = GCHandle.Alloc(modelData, GCHandleType.Pinned);
      IntPtr modelDataPtr = modelDataHandle.AddrOfPinnedObject();
      handle = TFL_NewInterpreter(modelDataPtr, modelData.Length);
      if (handle == IntPtr.Zero) throw new Exception("Failed to create TensorFlowLite Interpreter");
    }

    ~Interpreter() {
      Dispose();
    }

    public void Dispose() {
      if (handle != IntPtr.Zero) TFL_DeleteInterpreter(handle);
      handle = IntPtr.Zero;
    }

    public void Invoke() {
      ThrowIfError(TFL_InterpreterInvoke(handle));
    }

    public int GetInputTensorCount() {
      return TFL_InterpreterGetInputTensorCount(handle);
    }

    public void SetInputTensorData(int inputTensorIndex, Array inputTensorData) {
      GCHandle tensorDataHandle = GCHandle.Alloc(inputTensorData, GCHandleType.Pinned);
      IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
      TFL_Tensor tensor = TFL_InterpreterGetInputTensor(handle, inputTensorIndex);
      ThrowIfError(TFL_TensorCopyFromBuffer(
          tensor, tensorDataPtr, Buffer.ByteLength(inputTensorData)));
    }

    public void ResizeInputTensor(int inputTensorIndex, int[] inputTensorShape) {
      ThrowIfError(TFL_InterpreterResizeInputTensor(
          handle, inputTensorIndex, inputTensorShape, inputTensorShape.Length));
    }

    public void AllocateTensors() {
      ThrowIfError(TFL_InterpreterAllocateTensors(handle));
    }

    public int GetOutputTensorCount() {
      return TFL_InterpreterGetOutputTensorCount(handle);
    }

    public void GetOutputTensorData(int outputTensorIndex, Array outputTensorData) {
      GCHandle tensorDataHandle = GCHandle.Alloc(outputTensorData, GCHandleType.Pinned);
      IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
      TFL_Tensor tensor = TFL_InterpreterGetOutputTensor(handle, outputTensorIndex);
      ThrowIfError(TFL_TensorCopyToBuffer(
          tensor, tensorDataPtr, Buffer.ByteLength(outputTensorData)));
    }

    private static void ThrowIfError(int resultCode) {
      if (resultCode != 0) throw new Exception("TensorFlowLite operation failed.");
    }

    #region Externs

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TFL_Interpreter TFL_NewInterpreter(
        IntPtr model_data,
        int model_size);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe void TFL_DeleteInterpreter(TFL_Interpreter interpreter);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TFL_InterpreterGetInputTensorCount(
        TFL_Interpreter interpreter);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TFL_Tensor TFL_InterpreterGetInputTensor(
        TFL_Interpreter interpreter,
        int input_index);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TFL_InterpreterResizeInputTensor(
        TFL_Interpreter interpreter,
        int input_index,
        int[] input_dims,
        int input_dims_size);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TFL_InterpreterAllocateTensors(
        TFL_Interpreter interpreter);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TFL_InterpreterInvoke(TFL_Interpreter interpreter);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TFL_InterpreterGetOutputTensorCount(
        TFL_Interpreter interpreter);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TFL_Tensor TFL_InterpreterGetOutputTensor(
        TFL_Interpreter interpreter,
        int output_index);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TFL_TensorCopyFromBuffer(
        TFL_Tensor tensor,
        IntPtr input_data,
        int input_data_size);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TFL_TensorCopyToBuffer(
        TFL_Tensor tensor,
        IntPtr output_data,
        int output_data_size);

    #endregion
  }
}
