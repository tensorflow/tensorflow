# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

import argparse
import ast
import sys
import re


def parse_file(filename):
    with open(filename) as f:
        f_data = f.read()
    results = {}
    def regex_match(regex):
        match = re.match(regex, f_data, re.DOTALL)
        if match is not None:
            results[match.group(1)] = match.group(2)
    regex_match('.*(model): (\w*)')
    regex_match('.*(accuracy): (\d*\.\d*)')
    assert len(results) == 2, '{}'.format(results)
    return results

def check_accuracy(res, tol):
    dest = {
        'mobilenet_v1': 71.02,
        'mobilenet_v2': 74.11,
        'nasnet_large': 82.71,
        'nasnet_mobile': 73.97,
        'resnet_v1_50': 75.91,
        'resnet_v2_50': 76.05,
        'vgg_16': 70.89,
        'vgg_19': 71.00,
        'inception_v3': 77.98,
        'inception_v4': 80.18,
    }
    if abs(float(res['accuracy']) - dest[res['model']]) < tol:
        print("PASS")
    else:
        print("FAIL: accuracy {} vs. {}".format(res['accuracy'], dest[res['model']]))
        sys.exit(1)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input')
    parser.add_argument('--tolerance', dest='tolerance', type=float, default=0.1)
    
    args = parser.parse_args()
    filename = args.input
    tolerance = args.tolerance

    print()
    print('checking accuracy...')
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))
    check_accuracy(parse_file(filename), tolerance)

