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
# ==============================================================================
"""objection_detection for tflite"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tensorflow.lite.python import interpreter as interpreter_wrapper

def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels

if __name__ == "__main__":
  floating_model = False

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--image", default="/tmp/image2.jpg", help="image to be classified")
  parser.add_argument("-g", "--graph", default="/tmp/detect.tflite", help=".tflite model to be executed")
  parser.add_argument("-l", "--labels", default="/tmp/labelmap.txt", help="name of file containing labels")
  parser.add_argument("-o", "--output_file", default="/tmp/output.pdf", help="output image file")
  parser.add_argument("-b", "--input_mean", default=127.5, help="input_mean")
  parser.add_argument("-s", "--input_std", default=127.5, help="input standard deviation")
  parser.add_argument("--show_image", default=False, help="show image")
  parser.add_argument("-v", "--verbose", default=False, help="show some debug info")
  args = parser.parse_args()

  interpreter = interpreter_wrapper.Interpreter(model_path=args.graph)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  if args.verbose:
    print(input_details)
    print(output_details)

  # check the type of the input tensor
  if input_details[0]['dtype'] == np.float32:
    floating_model = True

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(args.image)
  img = img.resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - arags.input_mean) / args.input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  finish_time = time.time()
  print("time spent:", ((finish_time - start_time) * 1000), "ms")

  labels = load_labels(args.labels)

  detected_boxes = interpreter.get_tensor(output_details[0]['index']) * height
  detected_classes = interpreter.get_tensor(output_details[1]['index'])
  detected_scores = interpreter.get_tensor(output_details[2]['index'])
  num_boxes = interpreter.get_tensor(output_details[3]['index'])

  if args.verbose:
    print("num_boxes:", num_boxes[0])
    print("detected boxes:", detected_boxes)
    print("detected classes:", detected_classes)
    print("detected scores:", detected_scores)

  fig, ax = plt.subplots(1)

  for r in range(1, int(num_boxes)):
    top, left, bottom, right = detected_boxes[0][r]
    rect = patches.Rectangle((left, top), (right - left), (bottom - top), \
           linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    label_string = labels[int(detected_classes[0][r])+1]
    score_string = '{0:2.0f}%'.format(detected_scores[0][r] * 100)
    ax.text(left, top, label_string + ': ' + score_string, \
              fontsize=6, bbox=dict(facecolor='y', edgecolor='y', alpha=0.5))

  plt.title(args.graph)
  ax.imshow(img)

  if args.show_image:
    plt.show()
  else:
    print("savin output to", args.output_file)
    plt.savefig(args.output_file)
