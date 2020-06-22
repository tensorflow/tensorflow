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
python efficientnet_weight_update_util.py --model b3 --notop --ckpt noisy_student_efficientnet-b3/model.ckpt --o efficientnetb3_new.h5

"""

import tensorflow as tf
import tensorflow.keras as keras
import h5py
import numpy as np
import argparse
from tensorflow.keras.applications import efficientnet

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
  model_name_tf = model_name_keras.replace('efficientnet', 'efficientnet-')

  keras_weight_names = [w.name for w in keras_model.weights]
  tf_weight_names = get_variable_names_from_ckpt(path_ckpt)

  keras_blocks = get_keras_blocks(keras_weight_names)
  tf_blocks = get_tf_blocks(tf_weight_names)

  print('check variables match in each block')
  for keras_block, tf_block in zip(keras_blocks, tf_blocks):
    check_match(keras_block, tf_block, keras_weight_names, tf_weight_names, model_name_tf)
    print(f'{keras_block} and {tf_block} match.')

  block_mapping = {x[0]: x[1] for x in zip(keras_blocks, tf_blocks)}

  changed_weights = 0
  for w in keras_model.weights:
    if 'block' in w.name:
      # example: 'block1a_dwconv/depthwise_kernel:0' -> 'block1a'
      keras_block = w.name.split('/')[0].split('_')[0]
      tf_block = block_mapping[keras_block]
      tf_name = keras_name_to_tf_name_block(w.name, keras_block=keras_block, tf_block=tf_block, use_ema=use_ema, model_name_tf=model_name_tf)
    elif any([x in w.name for x in ['stem', 'top', 'predictions', 'probs']]):
      tf_name = keras_name_to_tf_name_stem_top(w.name, use_ema=use_ema, model_name_tf=model_name_tf)
    elif 'normalization' in w.name:
      print(f'skipping variable {w.name}: normalization is a layer in keras implementation, but preprocessing in TF implementation.')
      continue
    else:
      raise ValueError(f'{w.name} failed to parse.')

    try:
      w_tf = tf.train.load_variable(path_ckpt, tf_name)
      if (w.value().numpy() != w_tf).any():
        w.assign(w_tf)
        changed_weights += 1
    except Exception as e:
      if any([x in w.name for x in ['top', 'predictions', 'probs']]):
        warnings.warn(f'Fail to load top layer variable {w.name} from {tf_name} because of {e}.')
      else:
        raise ValueError(f'Fail to load {w.name} from {tf_name}')
  print(f'{changed_weights}/{len(keras_model.weights)} weights updated')
  keras_model.save_weights(path_h5)


def get_variable_names_from_ckpt(path_ckpt, use_ema=True):
  """Get list of tensor names from checkpoint

  Args:
    path_ckpt: str, path to the ckpt files
    use_ema: Bool, whether to use ExponentialMovingAverage result or not
  """
  v_all = tf.train.list_variables(path_ckpt)

  # keep name only
  v_name_all = [x[0] for x in v_all]

  if use_ema:
    v_name_all = [x for x in v_name_all if 'ExponentialMovingAverage' in x]
  else:
    v_name_all = [x for x in v_name_all if 'ExponentialMovingAverage' not in x]

  # remove util variables used for RMSprop
  v_name_all = [x for x in v_name_all if 'RMS' not in x]
  return v_name_all


def get_tf_blocks(tf_weight_names):
  """Extract the block names from list of full weight names"""
  # Example: 'efficientnet-b0/blocks_0/conv2d/kernel' -> 'blocks_0'
  tf_blocks = set([x.split('/')[1] for x in tf_weight_names if 'block' in x])
  # sort by number
  tf_blocks = sorted(tf_blocks, key=lambda x: int(x.split('_')[1]))
  return tf_blocks


def get_keras_blocks(keras_weight_names):
  """Extract the block names from list of full weight names"""
  # example: 'block1a_dwconv/depthwise_kernel:0' -> 'block1a'
  keras_blocks = set([x.split('_')[0] for x in keras_weight_names if 'block' in x])
  return sorted(keras_blocks)


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
      'probs/bias:0': f'{model_name_tf}/head/dense/bias{ema}',
      'probs/kernel:0': f'{model_name_tf}/head/dense/kernel{ema}',
      'predictions/bias:0': f'{model_name_tf}/head/dense/bias{ema}',
      'predictions/kernel:0': f'{model_name_tf}/head/dense/kernel{ema}',
      'stem_conv/kernel:0': f'{model_name_tf}/stem/conv2d/kernel{ema}',
      'top_conv/kernel:0': f'{model_name_tf}/head/conv2d/kernel{ema}',
  }

  # stem batch normalization
  for bn_weights in ['beta', 'gamma', 'moving_mean', 'moving_variance']:
    stem_top_dict[f'stem_bn/{bn_weights}:0'] = f'{model_name_tf}/stem/tpu_batch_normalization/{bn_weights}{ema}'
  # top / head batch normalization
  for bn_weights in ['beta', 'gamma', 'moving_mean', 'moving_variance']:
    stem_top_dict[f'top_bn/{bn_weights}:0'] = f'{model_name_tf}/head/tpu_batch_normalization/{bn_weights}{ema}'

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
  return '/'.join(tf_name)


def check_match(keras_block, tf_block, keras_weight_names, tf_weight_names, model_name_tf):
  """ Check if the weights in h5 and ckpt match
  
  we match each name from keras_weight_names that is in keras_block 
  and check if there is 1-1 correspondence to names from tf_weight_names
  that is in tf_block
  
  Args:
    keras_block: str, the block name for keras implementation (e.g. 'block1a')
    tf_block: str, the block name for tf implementation (e.g. 'blocks_0')
    keras_weight_names: list of str, each string is a name for weights in keras implementation
    tf_weight_names: list of str, each string is a name for weights in tf implementation
  """
  match_lst = []
  for x in keras_weight_names:
    if keras_block in x:
      y = keras_name_to_tf_name_block(x, keras_block=keras_block, tf_block=tf_block, model_name_tf=model_name_tf)
      match_lst.append(y)
  
  assert len(match_lst) > 0, f'there is no weight in block {keras_block}'
  for x in tf_weight_names:
    if tf_block in x and x.split('/')[1].endswith(tf_block):
      try:
        match_lst.remove(x)
      except:
        raise ValueError(f'{x} not in tf_weight_names')
  assert len(match_lst) == 0 , f'{len(match_lst)} variables in {tf_block} are not in {keras_block} of keras model. '


if __name__ == '__main__':
  arg_to_model = {
      'b0': efficientnet.EfficientNetB0,
      'b1': efficientnet.EfficientNetB1,
      'b2': efficientnet.EfficientNetB2,
      'b3': efficientnet.EfficientNetB3,
      'b4': efficientnet.EfficientNetB4,
      'b5': efficientnet.EfficientNetB5,
      'b6': efficientnet.EfficientNetB6,
      'b7': efficientnet.EfficientNetB7
  }
  parser = argparse.ArgumentParser(description="write weights from checkpoint to h5")
  parser.add_argument('--model', required=True, type=str, help="name of efficient model", choices=arg_to_model.keys())
  parser.add_argument('--notop',  action="store_true", help="do not include top layers", default=False)
  parser.add_argument('--ckpt', required=True, type=str, help="checkpoint path")
  parser.add_argument('--output', '-o', required=True, type=str, help="output (h5) file path")
  args = parser.parse_args()

  include_top = not args.notop

  model = arg_to_model[args.model](include_top=include_top)
  write_ckpt_to_h5(args.output, args.ckpt, keras_model=model)
