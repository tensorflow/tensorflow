Decorator to register a KL divergence implementation function.

Usage:

@distributions.RegisterKL(distributions.Normal, distributions.Normal)
def _kl_normal_mvn(norm_a, norm_b):
  # Return KL(norm_a || norm_b)
- - -

#### `tf.contrib.distributions.RegisterKL.__call__(kl_fn)` {#RegisterKL.__call__}

Perform the KL registration.

##### Args:


*  <b>`kl_fn`</b>: The function to use for the KL divergence.

##### Returns:

  kl_fn

##### Raises:


*  <b>`TypeError`</b>: if kl_fn is not a callable.
*  <b>`ValueError`</b>: if a KL divergence function has already been registered for
    the given argument classes.


- - -

#### `tf.contrib.distributions.RegisterKL.__init__(dist_cls_a, dist_cls_b)` {#RegisterKL.__init__}

Initialize the KL registrar.

##### Args:


*  <b>`dist_cls_a`</b>: the class of the first argument of the KL divergence.
*  <b>`dist_cls_b`</b>: the class of the second argument of the KL divergence.


