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
  from tensorflow.lite.tools import visualize
  from tensorflow.lite.python.analyzer_wrapper import _pywrap_analyzer_wrapper as _analyzer_wrapper
else:
  # This file is part of tflite_runtime package.
  from tflite_runtime import visualize
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
  def analyze(tflite_model, result_format):
    """Analyzes the given tflite_model.

    Args:
      tflite_model: TFLite flatbuffer model.
      result_format: txt|html|webserver.

    Returns:
      Analyzed report with the given result_format.
    """
    if result_format == "html":
      return visualize.create_html(tflite_model)
    elif result_format == "webserver":
      html_body = visualize.create_html(tflite_model)
      _handle_webserver("localhost", 8080, html_body)
    elif result_format == "txt":
      return _analyzer_wrapper.ModelAnalyzer(tflite_model)
    else:
      raise ValueError(f"result_format '{result_format}' is not supported")
