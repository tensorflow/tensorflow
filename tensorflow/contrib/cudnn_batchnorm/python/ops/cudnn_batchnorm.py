from tensorflow.python.framework import ops
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.contrib.cudnn_batchnorm import gen_cudnn_batchnorm

def batch_norm_training(t, beta, gamma, variance_epsilon=0.0001, data_format="NCHW"):
    """ Batch Normalization in training mode
    Args:
      t: A 4D input Tensor.
      beta: A 1D beta Tenor with size matching the 'C' dimention of t.
        An offset to be addded to the normalized tensor.
      gamma: A 1D beta Tenor with size matching the 'C' dimention of t.
        A scalar to be multipled with the normalized tensor
      variance_epsilon: A small float number to avoid dividing by 0.

    Returns:
      output: a batch-normalized `t`.
      mean: 1D tensor with computed means for the batch.
      inv_var: A 1D tensor with computed inverse variances of the batch.
    """
    output, _, _= gen_cudnn_batchnorm.batch_norm_training(t, gamma, beta, epsilon=variance_epsilon, data_format=data_format)
    return output

def batch_norm_inference(t, beta, gamma, population_means,
                         population_inv_var, variance_epsilon, data_format="NCHW"):
    """ Batch Normalization in inference mode
    Args:
      t: A 4D input Tensor.
      beta: A 1D beta Tenor with size matching the 'C' dimention of t.
        An offset to be addded to the normalized tensor.
      gamma: A 1D beta Tenor with size matching the 'C' dimention of t.
        A scalar to be multipled with the normalized tensor
      variance_epsilon: A small float number to avoid dividing by 0.
      population_means: A 1D Tensor with size matching the 'C' dimention of t.
        Computed means of activations.
      population_inv_var: A 1D Tensor with size matching the 'C' dimention of t.
        Computed inverse variance of activations.
    Returns:
      A batch normalized `t`
    """
    channel_dim = t.get_shape()[1].value

    if beta.get_shape()[0].value != channel_dim:
        raise AttributeError("beta size does not match input's channels.")
    if gamma.get_shape()[0].value != channel_dim:
        raise AttributeError("gamma size does not match input's channels.")
    if population_means.get_shape()[0].value != channel_dim:
        raise AttributeError("population_mean size does not match input's channels.")
    if population_inv_var.get_shape()[0].value != channel_dim:
        raise AttributeError("population_inv_var size does not match input's channels.")

    # TODO implement / check me
    beta_reshape = tf.reshape(beta, [1, channel_dims, 1, 1])
    gamma_reshape = tf.reshape(gamma, [1, channel_dims, 1, 1])
    means_reshape = tf.reshape(population_means, [1, channel_dims, 1, 1])
    inv_var_reshape = tf.reshape(population_inv_var, [1, channel_dims, 1, 1])

    scaled_var = (t  - means_reshape) * inv_var_reshape

    return gamma_reshape * scaled_var + beta_reshape

@ops.RegisterGradient("BatchNormTraining")
def _BatchNormTrainingGrad(op, output_grad, saved_mean_grad, saved_inv_var_grad):
    input_data = op.inputs[0]
    scale_data = op.inputs[1]

    saved_mean = op.outputs[1]
    saved_inv_var = op.outputs[2]

    dinput, dscale, dbias = gen_cudnn_batchnorm._batch_norm_training_grad(input_data,
        output_grad, scale_data, saved_mean, saved_inv_var,
        epsilon=op.get_attr("epsilon"), data_format=op.get_attr("data_format"))

    return dinput, dscale, dbias
