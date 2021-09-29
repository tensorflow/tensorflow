# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""A utility class to generate the report HTML based on a common template."""

import io
import os

from tensorflow.lite.toco.logging import toco_conversion_log_pb2 as _toco_conversion_log_pb2
from tensorflow.python.lib.io import file_io as _file_io
from tensorflow.python.platform import resource_loader as _resource_loader

html_escape_table = {
    "&": "&amp;",
    '"': "&quot;",
    "'": "&apos;",
    ">": "&gt;",
    "<": "&lt;",
}


def html_escape(text):
  return "".join(html_escape_table.get(c, c) for c in text)


def get_input_type_from_signature(op_signature):
  """Parses op_signature and returns a string denoting the input tensor type.

  Args:
    op_signature: a string specifying the signature of a particular operator.
      The signature of an operator contains the input tensor's shape and type,
      output tensor's shape and type, operator's name and its version. It has
      the following schema:
      INPUT:input_1_shape::input_1_type::input_2_shape::input_2_type::..
        ::OUTPUT:output_1_shape::output_1_type::output_2_shape::output_2_type::
        ..::NAME:operator_name ::VERSION:operator_version
     An example of an operator signature is:
     INPUT:[1,73,73,160]::float::[64,1,1,160]::float::[64]::float::
     OUTPUT:[1,73,73,64]::float::NAME:Conv::VERSION:1

  Returns:
    A string denoting the input tensors' type. In the form of shape/type
    separated
    by comma. For example:
    shape:[1,73,73,160],type:float,shape:[64,1,1,160],type:float,shape:[64],
    type:float
  """
  start = op_signature.find(":")
  end = op_signature.find("::OUTPUT")
  inputs = op_signature[start + 1:end]
  lst = inputs.split("::")
  out_str = ""
  for i in range(len(lst)):
    if i % 2 == 0:
      out_str += "shape:"
    else:
      out_str += "type:"
    out_str += lst[i]
    out_str += ","
  return out_str[:-1]


def get_operator_type(op_name, conversion_log):
  if op_name in conversion_log.built_in_ops:
    return "BUILT-IN"
  elif op_name in conversion_log.custom_ops:
    return "CUSTOM OP"
  else:
    return "SELECT OP"


class HTMLGenerator(object):
  """Utility class to generate an HTML report."""

  def __init__(self, html_template_path, export_report_path):
    """Reads the HTML template content.

    Args:
      html_template_path: A string, path to the template HTML file.
      export_report_path: A string, path to the generated HTML report. This path
        should point to a '.html' file with date and time in its name.
        e.g. 2019-01-01-10:05.toco_report.html.

    Raises:
      IOError: File doesn't exist.
    """
    # Load the template HTML.
    if not _file_io.file_exists(html_template_path):
      raise IOError("File '{0}' does not exist.".format(html_template_path))
    with _file_io.FileIO(html_template_path, "r") as f:
      self.html_template = f.read()

    _file_io.recursive_create_dir(os.path.dirname(export_report_path))
    self.export_report_path = export_report_path

  def generate(self,
               toco_conversion_log_before,
               toco_conversion_log_after,
               post_training_quant_enabled,
               dot_before,
               dot_after,
               toco_err_log="",
               tflite_graph_path=""):
    """Generates the HTML report and writes it to local directory.

    This function uses the fields in `toco_conversion_log_before` and
    `toco_conversion_log_after` to populate the HTML content. Certain markers
    (placeholders) in the HTML template are then substituted with the fields
    from the protos. Once finished it will write the HTML file to the specified
    local file path.

    Args:
      toco_conversion_log_before: A `TocoConversionLog` protobuf generated
        before the model is converted by TOCO.
      toco_conversion_log_after: A `TocoConversionLog` protobuf generated after
        the model is converted by TOCO.
      post_training_quant_enabled: A boolean, whether post-training quantization
        is enabled.
      dot_before: A string, the dot representation of the model
        before the conversion.
      dot_after: A string, the dot representation of the model after
        the conversion.
      toco_err_log: A string, the logs emitted by TOCO during conversion. Caller
        need to ensure that this string is properly anonymized (any kind of
        user data should be eliminated).
      tflite_graph_path: A string, the filepath to the converted TFLite model.

    Raises:
      RuntimeError: When error occurs while generating the template.
    """
    html_dict = {}
    html_dict["<!--CONVERSION_STATUS-->"] = (
        r'<span class="label label-danger">Fail</span>'
    ) if toco_err_log else r'<span class="label label-success">Success</span>'
    html_dict["<!--TOTAL_OPS_BEFORE_CONVERT-->"] = str(
        toco_conversion_log_before.model_size)
    html_dict["<!--TOTAL_OPS_AFTER_CONVERT-->"] = str(
        toco_conversion_log_after.model_size)
    html_dict["<!--BUILT_IN_OPS_COUNT-->"] = str(
        sum(toco_conversion_log_after.built_in_ops.values()))
    html_dict["<!--SELECT_OPS_COUNT-->"] = str(
        sum(toco_conversion_log_after.select_ops.values()))
    html_dict["<!--CUSTOM_OPS_COUNT-->"] = str(
        sum(toco_conversion_log_after.custom_ops.values()))
    html_dict["<!--POST_TRAINING_QUANT_ENABLED-->"] = (
        "is" if post_training_quant_enabled else "isn't")

    pre_op_profile = ""
    post_op_profile = ""

    # Generate pre-conversion op profiles as a list of HTML table rows.
    for i in range(len(toco_conversion_log_before.op_list)):
      # Append operator name column.
      pre_op_profile += "<tr><td>" + toco_conversion_log_before.op_list[
          i] + "</td>"
      # Append input type column.
      if i < len(toco_conversion_log_before.op_signatures):
        pre_op_profile += "<td>" + get_input_type_from_signature(
            toco_conversion_log_before.op_signatures[i]) + "</td></tr>"
      else:
        pre_op_profile += "<td></td></tr>"

    # Generate post-conversion op profiles as a list of HTML table rows.
    for op in toco_conversion_log_after.op_list:
      supported_type = get_operator_type(op, toco_conversion_log_after)
      post_op_profile += ("<tr><td>" + op + "</td><td>" + supported_type +
                          "</td></tr>")

    html_dict["<!--REPEAT_TABLE1_ROWS-->"] = pre_op_profile
    html_dict["<!--REPEAT_TABLE2_ROWS-->"] = post_op_profile
    html_dict["<!--DOT_BEFORE_CONVERT-->"] = dot_before
    html_dict["<!--DOT_AFTER_CONVERT-->"] = dot_after
    if toco_err_log:
      html_dict["<!--TOCO_INFO_LOG-->"] = html_escape(toco_err_log)
    else:
      success_info = ("TFLite graph conversion successful. You can preview the "
                      "converted model at: ") + tflite_graph_path
      html_dict["<!--TOCO_INFO_LOG-->"] = html_escape(success_info)

    # Replace each marker (as keys of html_dict) with the actual text (as values
    # of html_dict) in the HTML template string.
    template = self.html_template
    for marker in html_dict:
      template = template.replace(marker, html_dict[marker], 1)
      # Check that the marker text is replaced.
      if template.find(marker) != -1:
        raise RuntimeError("Could not populate marker text %r" % marker)

    with _file_io.FileIO(self.export_report_path, "w") as f:
      f.write(template)


def gen_conversion_log_html(conversion_log_dir, quantization_enabled,
                            tflite_graph_path):
  """Generates an HTML report about the conversion process.

  Args:
    conversion_log_dir: A string specifying the file directory of the conversion
      logs. It's required that before calling this function, the
      `conversion_log_dir`
      already contains the following files: `toco_log_before.pb`,
        `toco_log_after.pb`, `toco_tf_graph.dot`,
        `toco_tflite_graph.dot`.
    quantization_enabled: A boolean, passed from the tflite converter to
      indicate whether post-training quantization is enabled during conversion.
    tflite_graph_path: A string, the filepath to the converted TFLite model.

  Raises:
    IOError: When any of the required files doesn't exist.
  """
  template_filename = _resource_loader.get_path_to_datafile("template.html")
  if not os.path.exists(template_filename):
    raise IOError("Failed to generate HTML: file '{0}' doesn't exist.".format(
        template_filename))

  toco_log_before_path = os.path.join(conversion_log_dir, "toco_log_before.pb")
  toco_log_after_path = os.path.join(conversion_log_dir, "toco_log_after.pb")
  dot_before_path = os.path.join(conversion_log_dir, "toco_tf_graph.dot")
  dot_after_path = os.path.join(conversion_log_dir, "toco_tflite_graph.dot")
  if not os.path.exists(toco_log_before_path):
    raise IOError("Failed to generate HTML: file '{0}' doesn't exist.".format(
        toco_log_before_path))
  if not os.path.exists(toco_log_after_path):
    raise IOError("Failed to generate HTML: file '{0}' doesn't exist.".format(
        toco_log_after_path))
  if not os.path.exists(dot_before_path):
    raise IOError("Failed to generate HTML: file '{0}' doesn't exist.".format(
        dot_before_path))
  if not os.path.exists(dot_after_path):
    raise IOError("Failed to generate HTML: file '{0}' doesn't exist.".format(
        dot_after_path))

  html_generator = HTMLGenerator(
      template_filename,
      os.path.join(conversion_log_dir, "toco_conversion_summary.html"))

  # Parse the generated `TocoConversionLog`.
  toco_conversion_log_before = _toco_conversion_log_pb2.TocoConversionLog()
  toco_conversion_log_after = _toco_conversion_log_pb2.TocoConversionLog()
  with open(toco_log_before_path, "rb") as f:
    toco_conversion_log_before.ParseFromString(f.read())
  with open(toco_log_after_path, "rb") as f:
    toco_conversion_log_after.ParseFromString(f.read())

  # Read the dot file before/after the conversion.
  with io.open(dot_before_path, "r", encoding="utf-8") as f:
    dot_before = f.read().rstrip()
  with io.open(dot_after_path, "r", encoding="utf-8") as f:
    dot_after = f.read().rstrip()

  html_generator.generate(toco_conversion_log_before, toco_conversion_log_after,
                          quantization_enabled, dot_before, dot_after,
                          toco_conversion_log_after.toco_err_logs,
                          tflite_graph_path)
