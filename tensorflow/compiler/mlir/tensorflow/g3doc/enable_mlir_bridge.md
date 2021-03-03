# Enable MLIR-Based new TPU Bridge

**MLIR-Based new TPU Bridge is an experimental feature, tread lightly.**

## For TF 1.x-Based Models

In tf.ConfigProto.Experimental, there is a knob controlling whether the new TPU
Bridge is enabled or not. You can set it by using the following example code:

```
session_config = tf.ConfigProto(
  ......
  experimental=tf.ConfigProto.Experimental(
    enable_mlir_bridge=True,
  ),
  ......
)
```

## For TF 2.x-Based Models

Sessions and Session Configs are no longer available in TF 2.x. Instead, there
is a global **Context** that holds all the equivalences. You can manipulate the
**Context** with following code. Note that it must be added early in your
program (at least before any of your model computation).

```
tf.config.experimental.enable_mlir_bridge()
```

## How to disable the old TPU bridge?

Due to how TPU bridges are designed to work, you don't actually need to disable
the old bridge as they would not interfere with each other.

