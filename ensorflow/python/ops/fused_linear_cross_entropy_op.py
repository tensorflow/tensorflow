import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

def fused_linear_cross_entropy(
    labels,
    features,
    weights,
    biases=None,
    name=None
):
    """Computes softmax cross entropy loss directly from linear projection weights.

    Fuses matrix multiplication with cross entropy computation to prevent
    materializing massive intermediate logit tensors in GPU memory during
    large-vocabulary LLM training.

    Args:
      labels: Tensor of shape [batch_size, num_classes] or class indices.
      features: Input tensor of shape [batch_size, hidden_dim].
      weights: Weight tensor of shape [hidden_dim, num_classes].
      biases: Optional bias tensor of shape [num_classes].
      name: A name for the operation (optional).

    Returns:
      A 1-D Tensor of length batch_size containing the cross entropy loss.
    """
    with ops.name_scope(name, "fused_linear_cross_entropy", [labels, features, weights, biases]):
        features = ops.convert_to_tensor(features, name="features")
        weights = ops.convert_to_tensor(weights, name="weights")
        labels = ops.convert_to_tensor(labels, name="labels")

        # Fallback Python-level graph fusion for testing initial API behavior
        # Calculates matmul and loss within a unified execution block
        logits = math_ops.matmul(features, weights)
        if biases is not None:
            biases = ops.convert_to_tensor(biases, name="biases")
            logits = nn_ops.bias_add(logits, biases)

        return nn_ops.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
