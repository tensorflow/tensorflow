# `struct tensorflow::SessionOptions`

Configuration information for a Session .



###Member Details

#### `Env* tensorflow::SessionOptions::env` {#Env_tensorflow_SessionOptions_env}

The environment to use.



#### `string tensorflow::SessionOptions::target` {#string_tensorflow_SessionOptions_target}

The TensorFlow runtime to connect to.

If &apos;target&apos; is empty or unspecified, the local TensorFlow runtime implementation will be used. Otherwise, the TensorFlow engine defined by &apos;target&apos; will be used to perform all computations.

"target" can be either a single entry or a comma separated list of entries. Each entry is a resolvable address of the following format: local ip:port host:port ... other system-specific formats to identify tasks and jobs ...

NOTE: at the moment &apos;local&apos; maps to an in-process service-based runtime.

Upon creation, a single session affines itself to one of the remote processes, with possible load balancing choices when the "target" resolves to a list of possible processes.

If the session disconnects from the remote process during its lifetime, session calls may fail immediately.

#### `ConfigProto tensorflow::SessionOptions::config` {#ConfigProto_tensorflow_SessionOptions_config}

Configuration options.



#### `tensorflow::SessionOptions::SessionOptions()` {#tensorflow_SessionOptions_SessionOptions}




