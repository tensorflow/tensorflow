Parameters required to configure supervisor for training.

Fields:
  is_chief: Whether the current process is the chief supervisor in charge of
    restoring the model and running standard services.
  master: The master string to use when preparing the session.
  save_model_secs: Save a checkpoint every `save_model_secs` seconds when
    training.
  save_summaries_secs: Save summaries every `save_summaries_secs` seconds when
    training.
- - -

#### `tf.contrib.learn.SupervisorParams.is_chief` {#SupervisorParams.is_chief}

Alias for field number 0


- - -

#### `tf.contrib.learn.SupervisorParams.master` {#SupervisorParams.master}

Alias for field number 1


- - -

#### `tf.contrib.learn.SupervisorParams.save_model_secs` {#SupervisorParams.save_model_secs}

Alias for field number 2


- - -

#### `tf.contrib.learn.SupervisorParams.save_summaries_secs` {#SupervisorParams.save_summaries_secs}

Alias for field number 3


