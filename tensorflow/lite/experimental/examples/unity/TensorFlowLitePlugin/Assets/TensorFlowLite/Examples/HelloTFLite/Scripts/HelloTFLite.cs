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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Simple example demonstrating use of the experimental C# bindings for TensorFlowLite.
/// </summary>
public class HelloTFLite : MonoBehaviour {

  [Tooltip("Configurable TFLite model.")]
  public TextAsset model;

  [Tooltip("Configurable TFLite input tensor data.")]
  public float[] inputs;

  [Tooltip("Target Text widget for display of inference execution.")]
  public Text inferenceText;

  private Interpreter interpreter;
  private float[] outputs;

  void Awake() {
    // As the demo is extremely simple, there's no need to run at full frame-rate.
    QualitySettings.vSyncCount = 0;
    Application.targetFrameRate = 5;
  }

  void Start () {
    interpreter = new Interpreter(model.bytes);
    Debug.LogFormat(
        "InputCount: {0}, OutputCount: {1}",
        interpreter.GetInputTensorCount(),
        interpreter.GetOutputTensorCount());
  }

  void Update () {
    if (inputs == null) {
      return;
    }

    if (outputs == null || outputs.Length != inputs.Length) {
      interpreter.ResizeInputTensor(0, new int[]{inputs.Length});
      interpreter.AllocateTensors();
      outputs = new float[inputs.Length];
    }

    float startTimeSeconds = Time.realtimeSinceStartup;
    interpreter.SetInputTensorData(0, inputs);
    interpreter.Invoke();
    interpreter.GetOutputTensorData(0, outputs);
    float inferenceTimeSeconds = Time.realtimeSinceStartup - startTimeSeconds;

    inferenceText.text = string.Format(
        "Inference took {0:0.0000} ms\nInput(s): {1}\nOutput(s): {2}",
        inferenceTimeSeconds * 1000.0,
        ArrayToString(inputs),
        ArrayToString(outputs));
  }

  void OnDestroy() {
    interpreter.Dispose();
  }

   private static string ArrayToString(float[] values) {
    return string.Join(",", values.Select(x => x.ToString()).ToArray());
  }
}
