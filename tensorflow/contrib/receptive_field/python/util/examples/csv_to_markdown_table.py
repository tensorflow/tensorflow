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
"""Simple script to convert CSV output from rf_benchmark to Markdown format.

The input CSV should have the following fields:
- CNN
- input resolution
- end_point
- RF size hor
- RF size ver
- effective stride hor
- effective stride ver
- effective padding hor
- effective padding ver

Since usually in all cases the parameters in the horizontal and vertical
directions are the same, this is assumed by this script, which only prints one
of them to the Markdown file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import sys

from tensorflow.python.platform import app

cmd_args = None


def main(unused_argv):
  with open(cmd_args.markdown_path, 'w') as f:
    # Write table header and field size.
    f.write('CNN | resolution | end-point | RF | effective stride | '
            'effective padding|\n')
    f.write(
        ':--------------------: | :----------: | :---------------: | :-----: |'
        ' :----: | :----:|\n')
    with open(cmd_args.csv_path) as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        # Make sure horizontal and parameters are the same.
        assert row['RF size hor'] == row['RF size ver']
        assert row['effective stride hor'] == row['effective stride ver']
        assert row['effective padding hor'] == row['effective padding ver']

        f.write('%s|%s|%s|%s|%s|%s\n' %
                (row['CNN'], row['input resolution'], row['end_point'],
                 row['RF size hor'], row['effective stride hor'],
                 row['effective padding hor']))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--csv_path',
      type=str,
      default='/tmp/rf.csv',
      help='Path where CSV output of rf_benchmark was saved.')
  parser.add_argument(
      '--markdown_path',
      type=str,
      default='/tmp/rf.md',
      help='Path where Markdown output will be saved.')
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
