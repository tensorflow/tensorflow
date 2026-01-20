## STOP!

This folder contains the legacy Keras code which is stale and about to be
deleted. The current Keras code lives in
[github/keras-team/keras](https://github.com/keras-team/keras).

Please do not use the code from this folder.
### XLA compilation note

When using XLA (`jit_compile=True`), static configuration values such as
layer dimensions (`d_model`, `num_heads`, etc.) should be Python primitives
(e.g. `int`) and not `tf.Variable`.

Using `tf.Variable` for static configuration may cause XLA compilation
errors.
