from tensorflow.math import square, reduce_sum, sqrt
from tensorflow.keras.backend import epsilon
from tensorflow.keras.activations import softmax
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.layers import *
import tensorflow as tf

class Capsule(Layer):
    ''' 
    Original Paper: https://arxiv.org/pdf/1710.09829v2.pdf
        Args:
        num_capsules: number of capsules in the layer
        dim_capsule: output dimension of each capsule
        rountings: number of rounting iterations (default 3)
        activation: layer activation (default squash (according to the paper))
        kernel_initializer: Initializer for the `W` weights matrix.
        kernel_regularizer: Regularizer function applied to the `W` 
        weights matrix.
        bias_regularizer: Regularizer function applied to the 
        bias vector.activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the `W` weights 
        matrix.
        bias_constraint: Constraint function applied to the bias vector.
        
        Input shape:
             2-D tensor with shape: `(batch_size, input_dim)`.
        
        Output shape:
            3-D tensor with shape: `(batch_size,num_capsule, dim_capsule)`.
    '''

    def __init__(self,
                 num_capsule, 
                 dim_capsule, 
                 routings=3, 
                 share_weights=True,
                 activation='squash',
                 kernel_initializer='glorot_uniform',
                 bias_initializer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        
        # Capsule Hyperparameters
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights

        # General Hyperparameters
        #self.use_bias = use_bias
        #Bias is not being explicitly used in the current version
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.trainable=trainable
        if activation == 'squash':
            self.activation = self.squash
        else:
            self.activation = activations.get(activation)
        

    def build(self, input_shape):
        # This function initializes the weights of this layer
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     trainable=self.trainable,
                                     dtype = self.dtype)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     trainable=self.trainable,
                                     dtype = self.dtype)

    def squash(x):
        squared_norm = reduce_sum(square(x),axis=-1)
        additional_squashing = squared_norm/(1+squared_norm)
        unit_squashing = x/(sqrt(squared_norm)+epsilon)
        squashed_vector = additional_squashing*unit_squashing
        return squashed_vector

    def softmax(x, axis=-1):
        ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
        return ex / K.sum(ex, axis=axis, keepdims=True)

    def call(self, u_vecs):
        # Here the actual operations and flow happens
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.kernel)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.kernel, [1], [1])
        
        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
       
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        
        b = K.zeros_like(u_hat_vecs[:,:,:,0]) 

        for i in range(self.routings):
            feautre_importance = self.softmax(b, 1)
            o = self.activation(tf.einsum('bin,binj->bij', c, u_hat_vecs))
            if i < self.routings - 1:
                #o = K.l2_normalize(o, -1)
                b+=tf.einsum('bij,binj->bin', o, u_hat_vecs)
        return o
    
    def get_config(self):
        config = super(Capsule, self).get_config()
        config = {
            'num_capsule':
                num_capsule.serialize(self.num_capsule)
            'dim_capsule':
                num_capsule.serialize(self.dim_capsule)
            'routings':
                num_capsule.serialize(self.routings)
            'share_weights':
                num_capsule.serialize(self.share_weights)
            'activation':
                activations.serialize(self.activation),
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        return config
    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)