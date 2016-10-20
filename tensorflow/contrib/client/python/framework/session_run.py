# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import session

def register_session_fetch_feed_conversion_functions(tensor_type,
    fetch_function, feed_function=None, feed_function_for_partial_run=None):
  """ Register fetch and feed conversion functions for session.run.
  This function is for registering fetch and feed conversion functions for
  user defined type which may be passed to session.run as fetch or feed params.

  An example

  class SquaredTensor(object):
    def __init__(self, tensor):
      self.sq = tf.square(tensor)

  you can define conversion functions like that:

  fetch_function = lambda squared_tensor:([squared_tensor.sq],
                                          lambda val: val[0])
  feed_function = lambda feed, feed_val: [(feed.sq, feed_val)]
  feed_function_for_partial_run = lambda feed: [feed.sq]

  then after invoke this register function, you can use like that:

  session.run(squared_tensor1, feed_dict = {squared_tensor2 : some_numpy_array}

  Args:
    tensor_type: a type that you want to register conversion functions.
    fetch_function: a function or lambda expression that takes user defined
      type object as input, and returns a tuple
      (list_of_tensors,  contraction_function), list_of_tensors is a list
      contains all tensors that should be passed to fetch in the object,
      contraction_function is a function or lambda expression that takes a list
      of result corresponding to the list_of_tensors as input, and returns
      the result as whatever type or structure you defined in this function.
    feed_function: a function or lambda expression that takes feed_key and
      feed_value as input, and returns a list of tuples (feed_tensor, feed_val),
      feed_key may be user defined type object, but feed_tensor should be
      tensorflow's Tensor Object.
    feed_function_for_partial_run: a function or expression for specifying Graph
      Nodes in partial run case, which takes a user defined type object as
      input, and returns a list of Tensors for partial run case.
  """
  for conversion_function in session._REGISTERED_EXPANSIONS:
    if conversion_function[0] == tensor_type:
      raise ValueError(
          '%s has already been registered so ignore it.', tensor_type)
      return

  #we should put Object type as the last choice.
  if len(session._REGISTERED_EXPANSIONS) > 0 and \
      session._REGISTERED_EXPANSIONS[-1][0] == object:
    object_function = session._REGISTERED_EXPANSIONS[-1]
    session._REGISTERED_EXPANSIONS[-1] = (tensor_type, fetch_function,
        feed_function, feed_function_for_partial_run)
    session._REGISTERED_EXPANSIONS.append(object_function)
  else:
    session._REGISTERED_EXPANSIONS.append(tensor_type, fetch_function,
        feed_function, feed_function_for_partial_run)
