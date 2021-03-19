# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Make HTML tables that report where TF and TOCO failed to convert models.

This is primarily used by generate_examples.py. See it or
`make_report_table` for more details on usage.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import html
import json
import re

FAILED = "FAILED"
SUCCESS = "SUCCESS"
NOTRUN = "NOTRUN"


def make_report_table(fp, title, reports):
  """Make an HTML report of the success/failure reports.

  Args:
    fp: File-like object in which to put the html.
    title: "Title of the zip file this pertains to."
    reports: a list of conversion attempts. (report_args, report_vals) i.e.
      ({"shape": [1,2,3], "type": "tf.float32"},
       {"tf": "SUCCESS", "converter": "FAILURE",
       "converter_log": "Unsupported type.", "tf_log": ""})
  """
  # sort reports by if TOCO failure and then TF failure (reversed)
  reports.sort(key=lambda x: x[1]["converter"], reverse=False)
  reports.sort(key=lambda x: x[1]["tf"], reverse=True)
  def result_cell(x, row, col):
    """Produce a cell with the condition string `x`."""
    s = html.escape(repr(x), quote=True)
    color = "#44ff44" if x == SUCCESS else (
        "#ff4444" if x == FAILED else "#eeeeee")
    handler = "ShowLog(%d, %d)" % (row, col)
    fp.write("<td style='background-color: %s' onclick='%s'>%s</td>\n" % (
        color, handler, s))

  fp.write("""<html>
<head>
<title>tflite report</title>
<style>
body { font-family: Arial; }
th { background-color: #555555; color: #eeeeee; }
td { vertical-align: top; }
td.horiz {width: 50%;}
pre { white-space: pre-wrap; word-break: keep-all; }
table {width: 100%;}
</style>
</head>
""")
  # Write the log data to a javascript variable and also make a function
  # in javascript to show the log when an item is clicked.
  fp.write("<script> \n")
  fp.write("""
function ShowLog(row, col) {

var log = document.getElementById("log");
log.innerHTML = "<pre>" + data[row][col]  + "</pre>";
}
""")
  fp.write("var data = \n")
  logs = json.dumps([[escape_and_normalize(x[1]["tf_log"]),
                      escape_and_normalize(x[1]["converter_log"])
                     ] for x in reports])
  fp.write(logs)
  fp.write(";</script>\n")

  # Write the main table and use onclick on the items that have log items.
  fp.write("""
<body>
<h1>TOCO Conversion</h1>
<h2>%s</h2>
""" % title)

  # Get a list of keys that are in any of the records.
  param_keys = {}
  for params, _ in reports:
    for k in params.keys():
      param_keys[k] = True

  fp.write("<table>\n")
  fp.write("<tr><td class='horiz'>\n")
  fp.write("<div style='height:1000px; overflow:auto'>\n")
  fp.write("<table>\n")
  fp.write("<tr>\n")
  for p in param_keys:
    fp.write("<th>%s</th>\n" % html.escape(p, quote=True))
  fp.write("<th>TensorFlow</th>\n")
  fp.write("<th>TOCO</th>\n")
  fp.write("</tr>\n")
  for idx, (params, vals) in enumerate(reports):
    fp.write("<tr>\n")
    for p in param_keys:
      fp.write("  <td>%s</td>\n" % html.escape(repr(params[p]), quote=True))

    result_cell(vals["tf"], idx, 0)
    result_cell(vals["converter"], idx, 1)
    fp.write("</tr>\n")
  fp.write("</table>\n")
  fp.write("</div>\n")
  fp.write("</td>\n")
  fp.write("<td class='horiz' id='log'></td></tr>\n")
  fp.write("</table>\n")
  fp.write("<script>\n")
  fp.write("</script>\n")
  fp.write("""
    </body>
    </html>
    """)


def escape_and_normalize(log):
  # These logs contain paths like /tmp/tmpgmypg3xa that are inconsistent between
  # builds. This replaces these inconsistent paths with a consistent placeholder
  # so the output is deterministic.
  log = re.sub(r"/tmp/[^ ]+ ", "/NORMALIZED_TMP_FILE_PATH ", log)
  log = re.sub(r"/build/work/[^/]+", "/NORMALIZED_BUILD_PATH", log)
  return html.escape(log, quote=True)
