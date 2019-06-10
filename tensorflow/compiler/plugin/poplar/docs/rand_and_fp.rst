IEEE half precision floating point numbers and stochastic rounding
------------------------------------------------------------------

The IPU supports IEEE half precision floating point numbers, and supports
hardware stochastic rounding.  The IPU extensions to TensorFlow expose this
floating point functionality through two interfaces.

See the :ref:`api-section` for more details of the functions described here.

Controlling the half precision floating point unit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The floating point unit has a control register that controls its behaviour.
When configuring the IPU system hardware, the function
:py:func:`tensorflow.contrib.ipu.utils.set_floating_point_behaviour_options`
will set the control register.

The `esr` bit enables the stochastic rounding unit. Three of the remaining
options control the generation of hardware exceptions on various conditions.
The `nanoo` bit selects between clipping on overflow of a half precision number
or generating a NaN.

Resetting the global random number seed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The stochastic rounding unit, and the TensorFlow stateful random number
generators both use a common global random number seed to initialize the
random number generator hardware. Each TensorFlow IPU device has its own seed.

By default this seed is set randomly, but it can be reset by using the
:py:func:`tensorflow.contrib.ipu.utils.reset_ipu_seed` function.

Due to the hardware threading in the device, if the seed reset function is used
then the `target.deterministicWorkers` Poplar Engine option will need to be set
to `true`.

This can be done with using the
:py:func:`tensorflow.contrib.ipu.utils.set_compilation_options` function.

Debugging Numerical Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

The values held in a tensor can be printed by calling ipu.internal.print_tensor.
This function takes in a tensor and will print it to standard error as a side
effect.

See :py:func:`tensorflow.contrib.ipu.print_tensor`.
