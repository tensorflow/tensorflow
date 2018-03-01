# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.core.protobuf import config_pb2

import contextlib
import re

@contextlib.contextmanager
def ipu_session():
  opts = config_pb2.IPUOptions()
  dev = opts.device_config.add()
  dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
  dev.enable_profile = True
  with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:
    yield sess

def get_compute_sets_from_report(report):
  lines = report.split('\n')
  return [x for x in lines if re.search('(\d+ execution.?)', x)]

def check_all_compute_sets_in_list(cs_list, whitelist):
  for cs in cs_list:
    if len([x for x in whitelist if cs.startswith(x)]) == 0:
      return False
  return True
