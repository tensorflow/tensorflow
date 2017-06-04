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
"""Module for constructing Multi-Dimentional LSTM cells"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

def ln(tensor, scope = None, epsilon = 1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert(len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = vs.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = vs.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return LN_initial * scale + shift


class MultiDimentionalLSTMCell(tf.contrib.rnn.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=0.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM).
        @param: inputs (batch,n)
        @param state: the states and hidden unit of the two cells
        """
        with tf.variable_scope(scope or type(self).__name__):
            c1,c2,h1,h2 = state

            # change bias argument to False since LN will add bias via shift
            concat = _linear([inputs, h1, h2], 5 * self._num_units, False)

            i, j, f1, f2, o = array_ops.split(concat, 5, 1)

            # add layer normalization to each gate
            i =  ln(i, scope = 'i/')
            j =  ln(j, scope = 'j/')
            f1 = ln(f1, scope = 'f1/')
            f2 = ln(f2, scope = 'f2/')
            o =  ln(o, scope = 'o/')

            new_c = (c1 * tf.nn.sigmoid(f1 + self._forget_bias) + 
                     c2 * tf.nn.sigmoid(f2 + self._forget_bias) + tf.nn.sigmoid(i) *
                   self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(ln(new_c, scope = 'new_h/')) * tf.nn.sigmoid(o)
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

            return new_h, new_state

        
def multiDimentionalRNN_whileLoop(rnn_size,input_data,sh,dims=None,scopeN="layer1"):
        """Implements naive multidimentional recurent neural networks
        
        @param rnn_size: the hidden units
        @param input_data: the data to process of shape [batch,h,w,chanels]
        @param sh: [heigth,width] of the windows 
        @param dims: dimentions to reverse the input data
        @param scopeN : the scope
        
        returns [batch,h/sh[0],w/sh[1],chanels*sh[0]*sh[1]] the output of the lstm
        """
        with tf.variable_scope("MultiDimentionalLSTMCell-"+scopeN):
            cell = MultiDimentionalLSTMCell(rnn_size)
        
            shape = input_data.get_shape().as_list()
            
            #pad if the dimmention are not correct for the block size
            if shape[1]%sh[0] != 0:
                offset = tf.zeros([shape[0], sh[0]-(shape[1]%sh[0]), shape[2], shape[3]])
                input_data = tf.concat([input_data,offset],1)
                shape = input_data.get_shape().as_list()
            if shape[2]%sh[1] != 0:
                offset = tf.zeros([shape[0], shape[1], sh[1]-(shape[2]%sh[1]), shape[3]])
                input_data = tf.concat([input_data,offset],2)
                shape = input_data.get_shape().as_list()

            h,w = int(shape[1]/sh[0]),int(shape[2]/sh[1])
            features = sh[1]*sh[0]*shape[3]
            batch_size = shape[0]

            lines = array_ops.split(input_data,h,axis=1)
            line_blocks = []
            for line in lines:
              line = tf.transpose(line,[0,2,3,1])
              line = tf.reshape(line,[batch_size,w,features])
              line_blocks.append(line)
            x = tf.stack(line_blocks,axis=1)
            if dims is not None:
                x = tf.reverse(x, dims)
            x = tf.transpose(x, [1,2,0,3])
            x =  tf.reshape(x, [-1, features])
            x = array_ops.split(x, h*w, 0)     

            sequence_length = tf.ones(shape=(batch_size,), dtype=tf.int32)*shape[0]
            inputs_ta = tf.TensorArray(dtype=tf.float32, size=h*w,name='input_ta')
            inputs_ta = inputs_ta.unstack(x)
            states_ta = tf.TensorArray(dtype=tf.float32, size=h*w+1,name='state_ta',
                                       clear_after_read=False)
            outputs_ta = tf.TensorArray(dtype=tf.float32, size=h*w,name='output_ta')

            states_ta = states_ta.write(h*w, 
                                        tf.contrib.rnn.LSTMStateTuple(
                                            tf.zeros([batch_size,rnn_size], tf.float32),
                                                         tf.zeros([batch_size,rnn_size],
                                                                  tf.float32)))
            def getindex1(t,w):
                return tf.cond(tf.less_equal(tf.constant(w),t),
                               lambda:t-tf.constant(w),
                               lambda:tf.constant(h*w))
            def getindex2(t,w):
                return tf.cond(tf.less(tf.constant(0),tf.mod(t,tf.constant(w))),
                               lambda:t-tf.constant(1),
                               lambda:tf.constant(h*w))

            time = tf.constant(0)

            def body(time, outputs_ta, states_ta):
                constant_val = tf.constant(0)
                stateUp = tf.cond(tf.less_equal(tf.constant(w),time),
                                  lambda: states_ta.read(getindex1(time,w)),
                                  lambda: states_ta.read(h*w))
                stateLast = tf.cond(tf.less(constant_val,tf.mod(time,tf.constant(w))),
                                    lambda: states_ta.read(getindex2(time,w)),
                                    lambda: states_ta.read(h*w)) 

                currentState = stateUp[0],stateLast[0],stateUp[1],stateLast[1]
                out , state = cell(inputs_ta.read(time),currentState)  
                outputs_ta = outputs_ta.write(time,out)
                states_ta = states_ta.write(time,state)
                return time + 1, outputs_ta, states_ta

            def condition(time,outputs_ta,states_ta):
                return tf.less(time ,  tf.constant(h*w)) 

            result , outputs_ta, states_ta = tf.while_loop(condition, body, [time,outputs_ta,states_ta]
                                                           ,parallel_iterations=1)


            outputs = outputs_ta.stack()
            states  = states_ta.stack()

            outputs =  tf.reshape(outputs, [h,w,batch_size,rnn_size])
            outputs = tf.transpose(y, [2,0,1,3])
            if dims is not None:
                outputs = tf.reverse(outputs, dims)

            return outputs

    
def tanAndSum(rnn_size, input_data, sh, scope):
    """
    Sum and tan over four directions for MDLSTM
    @param: rnn_size - the size of the cell
    @param input_data - the 4 dimentional tensor
    @param sh[0] the height of the block, s[1] - the width of the lstm block
    @param scope - the function scope
    
    return the mean over four iteration with MDLSTM
    """
    outs = []
    for i in range(2):
        for j in range(2):
            dims = []
            if i!=0:
                dims.append(1)
            if j!=0:
                dims.append(2)                 
            outputs  = multiDimentionalRNN_whileLoop(rnn_size,input_data,sh,
                                                       dims,scope+"-multi-l{0}".format(i*2+j))
            outs.append(outputs)
        
    outs = tf.stack(outs, axis=0)
    mean = tf.reduce_mean(outs, 0)
    return tf.nn.tanh(mean)

