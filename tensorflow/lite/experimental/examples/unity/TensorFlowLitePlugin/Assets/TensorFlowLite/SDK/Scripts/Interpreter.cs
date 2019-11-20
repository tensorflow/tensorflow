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
using TfLiteDelegate = System.IntPtr;

namespace TensorFlowLite
{
  /// <summary>
  /// Simple C# bindings for the experimental TensorFlowLite C API.
  /// </summary>
  public class Interpreter : IDisposable
  {
    private TfLiteModel model;
    private TfLiteInterpreter interpreter;
    private TfLiteInterpreterOptions options;

    public Interpreter(byte[] modelData, int threads) {
      GCHandle modelDataHandle = GCHandle.Alloc(modelData, GCHandleType.Pinned);
      IntPtr modelDataPtr = modelDataHandle.AddrOfPinnedObject();
      model = TfLiteModelCreate(modelDataPtr, modelData.Length);
      if (model == IntPtr.Zero) throw new Exception("Failed to create TensorFlowLite Model");
      
      options = TfLiteInterpreterOptionsCreate();
      
      if (threads > 1) {
        TfLiteInterpreterOptionsSetNumThreads(options, threads);
      }

      interpreter = TfLiteInterpreterCreate(model, options);
      if (interpreter == IntPtr.Zero) throw new Exception("Failed to create TensorFlowLite Interpreter");
    }

    public void Dispose() {
      if (model != IntPtr.Zero) TfLiteModelDelete(model);
      model = IntPtr.Zero;
      if (interpreter != IntPtr.Zero) TfLiteInterpreterDelete(interpreter);
      interpreter = IntPtr.Zero;
      if (options != IntPtr.Zero) TfLiteInterpreterOptionsDelete(options);
      options = IntPtr.Zero;
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

    public string GetInputTensorInfo(int index) {
      TfLiteTensor tensor = TfLiteInterpreterGetInputTensor(interpreter, index);
      return GetTensorInfo(tensor);
    }

    public string GetOutputTensorInfo(int index) {
      TfLiteTensor tensor = TfLiteInterpreterGetOutputTensor(interpreter, index);
      return GetTensorInfo(tensor);
    }

    public static string GetVersion() {
      return Marshal.PtrToStringAnsi(TfLiteVersion());
    }

    private static string GetTensorName(TfLiteTensor tensor) {
      return Marshal.PtrToStringAnsi(TfLiteTensorName(tensor));
    }

    private static string GetTensorInfo(TfLiteTensor tensor) {
      var sb = new System.Text.StringBuilder();
      sb.AppendFormat("{0} type:{1}, dims:[",
          GetTensorName(tensor),
          TfLiteTensorType(tensor));
      
      int dims = TfLiteTensorNumDims(tensor);
      for (int i = 0; i < dims; i++) {
        sb.Append(TfLiteTensorDim(tensor, i));
        sb.Append(i == dims - 1 ? "]" : ", ");
      }
      return sb.ToString();
    }

    private static void ThrowIfError(int resultCode) {
      if (resultCode != 0) throw new Exception("TensorFlowLite operation failed.");
    }

    #region Externs

    #if UNITY_IPHONE && !UNITY_EDITOR
    private const string TensorFlowLibrary = "__Internal";
#else
    private const string TensorFlowLibrary = "tensorflowlite_c";
#endif

    public enum TfLiteType {
      NoType = 0,
      Float32 = 1,
      Int32 = 2,
      UInt8 = 3,
      Int64 = 4,
      String = 5,
      Bool = 6,
      Int16 = 7,
      Complex64 = 8,
      Int8 = 9,
      Float16 = 10,
    }

    public struct TfLiteQuantizationParams {
      public float scale;
      public int zero_point;
    }

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe IntPtr TfLiteVersion();

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TfLiteInterpreter TfLiteModelCreate(IntPtr model_data, int model_size);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TfLiteInterpreter TfLiteModelDelete(TfLiteModel model);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TfLiteInterpreterOptions TfLiteInterpreterOptionsCreate();

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions options);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TfLiteInterpreterOptions TfLiteInterpreterOptionsSetNumThreads(
        TfLiteInterpreterOptions options,
        int num_threads
    );

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TfLiteInterpreterOptions TfLiteInterpreterOptionsAddDelegate(
        TfLiteInterpreterOptions options,
        TfLiteDelegate _delegate);

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
    private static extern unsafe TfLiteType TfLiteTensorType(TfLiteTensor tensor);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe int TfLiteTensorNumDims(TfLiteTensor tensor);

    [DllImport (TensorFlowLibrary)]
    private static extern int TfLiteTensorDim(TfLiteTensor tensor, int dim_index);

    [DllImport (TensorFlowLibrary)]
    private static extern uint TfLiteTensorByteSize(TfLiteTensor tensor);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe IntPtr TfLiteTensorData(TfLiteTensor tensor);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe IntPtr TfLiteTensorName(TfLiteTensor tensor);

    [DllImport (TensorFlowLibrary)]
    private static extern unsafe TfLiteQuantizationParams TfLiteTensorQuantizationParams(TfLiteTensor tensor);

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
