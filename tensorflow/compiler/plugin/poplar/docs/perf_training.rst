Training a model
----------------

TensorFlow XLA and Poplar provide the opportunity to fuse an entire training
graph into a single operation in the TensorFlow graph.  This accelerates
training by preventing the need to make calls to the compute hardware for each
operation.

However, if the python code with the training pass on it is called multiple
times, once for each batch in the training data set, then there is still
the overhead of calling the hardware for each batch.

The GraphCore IPU support for TensorFlow provides three mechanisms for improving
the training performance:  training loops, data set feeds, and replicated
graphs.

Training loops
~~~~~~~~~~~~~~

TODO stuff to do with training in a loop

Data feeds
~~~~~~~~~~

TODO data feed ops and their relationship to datasets.
Note https://www.tensorflow.org/guide/performance/datasets


Replicated graphs
~~~~~~~~~~~~~~~~~

TODO replicated graphs - they increase the batch size by doing stuff in parallel
an d must be done inside a training loop, with a dataset feed, and require
cross replica summation ops to do the parameter updates.
