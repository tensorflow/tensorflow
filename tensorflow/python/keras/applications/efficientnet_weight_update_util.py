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
"""Utils for EfficientNet models for Keras.
Write weights from  ckpt file as in original repo
(https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
to h5 file for keras implementation of the models.

Usage: 

# use checkpoint efficientnet-b0/model.ckpt (can be downloaded from
# https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b0.tar.gz)
# to update weight without top layers, saving to efficientnetb0_notop.h5
python efficientnet_weight_update_util.py --model b0 --notop --ckpt efficientnet-b0/model.ckpt --o efficientnetb0_notop.h5

# use checkpoint noisy_student_efficientnet-b3/model.ckpt (providing improved result for b3, can be downloaded from
# https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b3.tar.gz)
# to update weight with top layers, saving to efficientnetb3_new.h5
python efficientnet_weight_update_util.py --model b0 --notop --ckpt noisy_student_efficientnet-b3/model.ckpt --o efficientnetb3_new.h5

"""

import tensorflow as tf
import tensorflow.keras as keras
import h5py
import numpy as np
import argparse
from tensorflow.keras.applications.efficientnet import *

def write_ckpt_to_h5(path_h5, path_ckpt, keras_model, use_ema=True):
  """ Map the weights in checkpoint file (tf) to h5 file (keras)
  
  Args:
    path_h5: str, path to output hdf5 file to write weights loaded
      from ckpt files.
    path_ckpt: str, path to the ckpt files (e.g. 'efficientnet-b0/model.ckpt')
      that records efficientnet weights from original repo 
      https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
    keras_model: keras model, built from keras.applications efficientnet
      functions (e.g. EfficientNetB0)
    use_ema: Bool, whether to use ExponentialMovingAverage result or not
  """
  model_name_keras = keras_model.name
  model_name_tf = model_name_keras_to_tf(model_name_keras)
  keras_model.save_weights(path_h5)

  keras_weights = get_h5_names(path_h5)
  tf_weights = get_tf_names(path_ckpt)
  blocks_keras = get_keras_blocks(keras_weights)
  blocks_tf = get_tf_blocks(tf_weights)

  with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph(f'{path_ckpt}.meta')
    saver.restore(sess, path_ckpt)
    graph = tf.compat.v1.get_default_graph()
    v_all = tf.compat.v1.global_variables()

    for keras_block, tf_block in zip(blocks_keras, blocks_tf):
      print(f'working on block {keras_block}, {tf_block}')
      for keras_name in keras_weights:
        if keras_block in keras_name:
          tf_name = keras_name_to_tf_name_block(keras_name, keras_block=keras_block, tf_block=tf_block, use_ema=use_ema, model_name_tf=model_name_tf)
          for v in v_all:
            if v.name == tf_name:
              v_val = sess.run(v)
              with h5py.File(path_h5, 'a') as f:
                v_prev = f[keras_name][()]
                f[keras_name].write_direct(v_val)
              print(f'writing from: {tf_name}\n  to: {keras_name}')
              print(f'  average change: {abs(v_prev - v_val).mean()}')
              v_all.remove(v)
              break
          else:
            raise ValueError(f'{keras_name} has no match in ckpt file')

    for keras_name in keras_weights:
      if any([x in keras_name for x in ['stem', 'top', 'predictions', 'probs']]):
        tf_name = keras_name_to_tf_name_stem_top(keras_name, use_ema=use_ema, model_name_tf=model_name_tf)
        for v in v_all:
          if v.name == tf_name:
            v_val = sess.run(v)
            with h5py.File(path_h5, 'a') as f:
              v_prev = f[keras_name][()]
              try:
                f[keras_name].write_direct(v_val)
              except:
                raise ValueError(f'weight in {tf_name} does not ift into {keras_name}')
            print(f'writing from: {tf_name}\n  to: {keras_name}')
            print(f'  average change: {abs(v_prev - v_val).mean()}')
            v_all.remove(v)
            break


def model_name_keras_to_tf(model_name_keras):
  """Infer model name in both keras and tf implementations"""
  model_name_tf = model_name_keras.replace('efficientnet', 'efficientnet-')
  return model_name_tf


def get_h5_names(path_h5):
  """Get list of variable names from the h5 file """
  h5_namelst = []
  def append_to_lst(x):
    h5_namelst.append(x)

  with h5py.File(path_h5, 'r') as f:
    for x in f.keys():
      f[x].visit(append_to_lst)

  # all weights end with ':0'
  h5_namelst = [x for x in h5_namelst if ':' in x]

  # append group name to the front
  h5_namelst = ['/'.join([x.split('/')[0], x]) for x in h5_namelst]
    
  return h5_namelst


def get_tf_names(path_ckpt, use_ema=True):
  """Get list of tensor names from checkpoint"""

  tf2_listvar = tf.train.list_variables(path_ckpt)

  if use_ema:
    tf2_listvar = [x for x in tf2_listvar if 'ExponentialMovingAverage' in x[0]]
  else:
    tf2_listvar = [x for x in tf2_listvar if 'ExponentialMovingAverage' not in x[0]]

  # remove util variables used for RMSprop
  tf2_listvar = [x for x in tf2_listvar if 'RMS' not in x[0]]

  tf2_listvar = [x[0] for x in tf2_listvar]
  return tf2_listvar


def get_tf_blocks(tf_weights):
  """Extract the block names from list of full weight names"""
  tf_blocks = set([x.split('/')[1] for x in tf_weights if 'block' in x])
  tf_blocks = sorted(tf_blocks, key=lambda x:int(x.split('_')[1]))
  return tf_blocks


def get_keras_blocks(keras_weights):
  """Extract the block names from list of full weight names"""
  return sorted(set([x.split('_')[0] for x in keras_weights if 'block' in x]))


def keras_name_to_tf_name_stem_top(keras_name, use_ema=True, model_name_tf='efficientnet-b0'):
  """ map name in h5 to ckpt that is in stem or top (head)
  
  we map name keras_name that points to a weight in h5 file 
  to a name of weight in ckpt file. 
  
  Args:
    keras_name: str, the name of weight in the h5 file of keras implementation
    use_ema: Bool, use the ExponentialMovingAverage resuolt in ckpt or not 
    model_name_tf: str, the name of model in ckpt.

  Returns:
    String for the name of weight as in ckpt file.

  Raises:
    KeyError if we cannot parse the keras_name
  """
  if use_ema:
    ema = '/ExponentialMovingAverage'
  else:
    ema = ''

  stem_top_dict = {
      'probs/probs/bias:0':f'{model_name_tf}/head/dense/bias{ema}:0',
      'probs/probs/kernel:0':f'{model_name_tf}/head/dense/kernel{ema}:0',
      'predictions/predictions/bias:0':f'{model_name_tf}/head/dense/bias{ema}:0',
      'predictions/predictions/kernel:0':f'{model_name_tf}/head/dense/kernel{ema}:0',
      'stem_conv/stem_conv/kernel:0':f'{model_name_tf}/stem/conv2d/kernel{ema}:0',
      'top_conv/top_conv/kernel:0':f'{model_name_tf}/head/conv2d/kernel{ema}:0',
  }

  # stem batch normalization
  for bn_weights in ['beta', 'gamma', 'moving_mean', 'moving_variance']:
    stem_top_dict[f'stem_bn/stem_bn/{bn_weights}:0'] = f'{model_name_tf}/stem/tpu_batch_normalization/{bn_weights}{ema}:0'
  # top / head batch normalization
  for bn_weights in ['beta', 'gamma', 'moving_mean', 'moving_variance']:
    stem_top_dict[f'top_bn/top_bn/{bn_weights}:0'] = f'{model_name_tf}/head/tpu_batch_normalization/{bn_weights}{ema}:0'

  if keras_name in stem_top_dict:
    return stem_top_dict[keras_name]
  else:
    raise KeyError(f'{keras_name} from h5 file cannot be parsed')


def keras_name_to_tf_name_block(keras_name, keras_block='block1a', tf_block='blocks_0', use_ema=True, model_name_tf='efficientnet-b0'):
  """ map name in h5 to ckpt that belongs to a block
  
  we map name keras_name that points to a weight in h5 file 
  to a name of weight in ckpt file. 
  
  Args:
    keras_name: str, the name of weight in the h5 file of keras implementation
    keras_block: str, the block name for keras implementation (e.g. 'block1a')
    tf_block: str, the block name for tf implementation (e.g. 'blocks_0')
    use_ema: Bool, use the ExponentialMovingAverage resuolt in ckpt or not 
    model_name_tf: str, the name of model in ckpt.

  Returns:
    String for the name of weight as in ckpt file.

  Raises:
    ValueError if keras_block does not show up in keras_name
  """

  if f'{keras_block}' not in keras_name:
    raise ValueError(f'block name {keras_block} not found in {keras_name}')

  # all blocks in the first group will not have expand conv and bn
  is_first_blocks = (keras_block[5]=='1')

  tf_name = [model_name_tf, tf_block]

  # depthwide conv
  if 'dwconv' in keras_name:
    tf_name.append('depthwise_conv2d')
    tf_name.append('depthwise_kernel')

  # conv layers
  if is_first_blocks:
    # first blocks only have one conv2d
    if 'project_conv' in keras_name:
      tf_name.append('conv2d')
      tf_name.append('kernel')
  else:
    if 'project_conv' in keras_name:
      tf_name.append('conv2d_1')
      tf_name.append('kernel')
    elif 'expand_conv' in keras_name:
      tf_name.append('conv2d')
      tf_name.append('kernel')
      
  # squeeze expansion layers 
  if '_se_' in keras_name:
    if 'reduce' in keras_name:
      tf_name.append('se/conv2d')
    elif 'expand' in keras_name:
      tf_name.append('se/conv2d_1')

    if 'kernel' in keras_name:
      tf_name.append('kernel')
    elif 'bias' in keras_name:
      tf_name.append('bias')

  # batch normalization layers 
  if 'bn' in keras_name:
    if is_first_blocks:
      if 'project' in keras_name:
        tf_name.append('tpu_batch_normalization_1')
      else:
        tf_name.append('tpu_batch_normalization')
    else:
      if 'project' in keras_name:
        tf_name.append('tpu_batch_normalization_2')
      elif 'expand' in keras_name:
        tf_name.append('tpu_batch_normalization')
      else:
        tf_name.append('tpu_batch_normalization_1')

    for x in ['moving_mean', 'moving_variance', 'beta', 'gamma']:
      if x in keras_name:
        tf_name.append(x)
  if use_ema:
    tf_name.append('ExponentialMovingAverage')
  return '/'.join(tf_name) + ':0'


def check_match(keras_block, tf_block, keras_weights, tf_weights):
  """ Check if the weights in h5 and ckpt match
  
  we match each name from keras_weights that is in keras_block 
  and check if there is 1-1 correspondence to names from tf_weights
  that is in tf_block
  
  Args:
    keras_block: str, the block name for keras implementation (e.g. 'block1a')
    tf_block: str, the block name for tf implementation (e.g. 'blocks_0')
    keras_weights: list of str, each string is a name for weights in keras implementation
    tf_weights: list of str, each string is a name for weights in tf implementation
  """
  for x in keras_weights:
    if keras_block in x:
      y = keras_name_to_tf_name_block(x, keras_block=bk, tf_block=bt)
    match_lst.append(y)

  assert len(match_lst) > 0

  for x in tf_weights:
    if tf_block in x[0] and x[0].split('/')[1].endswith(tf_block):
      match_lst.remove(x[0]+':0')
  assert len(match_lst) == 0 


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Load Models ")
  parser.add_argument("--model", required=True, type=str, help="name of efficient model. e.g. b2 or b5notop")
  parser.add_argument("--ckpt", required=True, type=str, help="checkpoint path")
  parser.add_argument("--o", required=True, type=str, help="output (h5) file path")
  args = parser.parse_args()

  include_top = True
  if args.model.endswith('notop'):
    include_top = False

  arg_to_model = {
      'b0':EfficientNetB0,
      'b1':EfficientNetB1,
      'b2':EfficientNetB2,
      'b3':EfficientNetB3,
      'b4':EfficientNetB4,
      'b5':EfficientNetB5,
      'b6':EfficientNetB6,
      'b7':EfficientNetB7
  }

  model = arg_to_model[args.model[0:2]](weights=None, include_top=include_top)
  write_ckpt_to_h5(args.o, args.ckpt, keras_model=model)
