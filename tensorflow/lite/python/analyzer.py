# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This tool analyzes a TensorFlow Lite graph."""

import http.server
import os

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join("tflite_runtime", "analyzer")):
  # This file is part of tensorflow package.
  from tensorflow.lite.python.analyzer_wrapper import _pywrap_analyzer_wrapper as _analyzer_wrapper
else:
  # This file is part of tflite_runtime package.
  from tflite_runtime import _pywrap_analyzer_wrapper as _analyzer_wrapper


def _handle_webserver(host_name, server_port, html_body):
  """Start a HTTP server for the given html_body."""

  class MyServer(http.server.BaseHTTPRequestHandler):

    def do_GET(self):  # pylint: disable=invalid-name
      self.send_response(200)
      self.send_header("Content-type", "text/html")
      self.end_headers()
      self.wfile.write(bytes(html_body, "utf-8"))

  web_server = http.server.HTTPServer((host_name, server_port), MyServer)
  print("Server started http://%s:%s" % (host_name, server_port))
  try:
    web_server.serve_forever()
  except KeyboardInterrupt:
    pass
  web_server.server_close()


class ModelAnalyzer:
  """Provides a collection of TFLite model analyzer tools."""

  @staticmethod
  def analyze(model_path=None,
              model_content=None,
              experimental_use_mlir=False,
              gpu_compatibility=False):
    """Analyzes the given tflite_model.

    Args:
      model_path: TFLite flatbuffer model path.
      model_content: TFLite flatbuffer model object.
      experimental_use_mlir: Use MLIR format for model dump.
      gpu_compatibility: Whether to check GPU delegate compatibility.

    Returns:
      Print analyzed report via console output.
    """
    if not model_path and not model_content:
      raise ValueError("neither `model_path` nor `model_content` is provided")
    if model_path:
      print(f"=== {model_path} ===\n")
      tflite_model = model_path
      input_is_filepath = True
    else:
      print("=== TFLite ModelAnalyzer ===\n")
      tflite_model = model_content
      input_is_filepath = False

    if experimental_use_mlir:
      print(_analyzer_wrapper.FlatBufferToMlir(tflite_model, input_is_filepath))
    else:
      print(
          _analyzer_wrapper.ModelAnalyzer(tflite_model, input_is_filepath,
                                          gpu_compatibility))
