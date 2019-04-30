Troubleshooting
---------------

The following error (especially the lines containing ``VariableV2``) indicate
that a variable has been created which is not a resource variable.

::

    InvalidArgumentError (see above for traceback): Cannot assign a device for operation
      'InceptionV1/Logits/Conv2d_0c_1x1/biases': Could not satisfy explicit device specification
      '/device:IPU:0' because no supported kernel for IPU devices is available.
    Colocation Debug Info:
    Colocation group had the following types and devices: 
    Const: CPU IPU XLA_CPU 
    Identity: CPU IPU XLA_CPU 
    Fill: CPU IPU XLA_CPU 
    Assign: CPU 
    VariableV2: CPU 

