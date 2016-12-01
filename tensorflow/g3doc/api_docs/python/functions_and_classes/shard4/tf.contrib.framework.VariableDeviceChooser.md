Device chooser for variables.

When using a parameter server it will assign them in a round-robin fashion.
When not using a parameter server it allows GPU or CPU placement.
- - -

#### `tf.contrib.framework.VariableDeviceChooser.__call__(op)` {#VariableDeviceChooser.__call__}




- - -

#### `tf.contrib.framework.VariableDeviceChooser.__init__(num_tasks=0, job_name='ps', device_type='CPU', device_index=0)` {#VariableDeviceChooser.__init__}

Initialize VariableDeviceChooser.

##### Usage:

  To use with 2 parameter servers:
    VariableDeviceChooser(2)

  To use without parameter servers:
    VariableDeviceChooser()
    VariableDeviceChooser(device_type='GPU') # For GPU placement

##### Args:


*  <b>`num_tasks`</b>: number of tasks.
*  <b>`job_name`</b>: String, a name for the parameter server job.
*  <b>`device_type`</b>: Optional device type string (e.g. "CPU" or "GPU")
*  <b>`device_index`</b>: int.  Optional device index.  If left
    unspecified, device represents 'any' device_index.


