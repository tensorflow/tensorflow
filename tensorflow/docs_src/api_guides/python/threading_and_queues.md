# Threading and Queues

Note: In versions of TensorFlow before 1.2, we recommended using multi-threaded,
queue-based input pipelines for performance. Beginning with TensorFlow 1.4,
however, we recommend using the `tf.data` module instead. (See
@{$datasets$Datasets} for details. In TensorFlow 1.2 and 1.3, the module was
called `tf.contrib.data`.) The `tf.data` module offers an easier-to-use
interface for constructing efficient input pipelines. Furthermore, we've stopped
developing the old multi-threaded, queue-based input pipelines.  We've retained
the documentation in this file to help developers who are still maintaining
older code.

Multithreaded queues are a powerful and widely used mechanism supporting
asynchronous computation.

Following the [dataflow programming model](graphs.md), TensorFlow's queues are
implemented using nodes in the computation graph.  A queue is a stateful node,
like a variable: other nodes can modify its content. In particular, nodes can
enqueue new items in to the queue, or dequeue existing items from the
queue. TensorFlow's queues provide a way to coordinate multiple steps of a
computation: a queue will **block** any step that attempts to dequeue from it
when it is empty, or enqueue to it when it is full. When that condition no
longer holds, the queue will unblock the step and allow execution to proceed.

TensorFlow implements several classes of queue. The principal difference between
these classes is the order that items are removed from the queue.  To get a feel
for queues, let's consider a simple example. We will create a "first in, first
out" queue (@{tf.FIFOQueue}) and fill it with zeros.  Then we'll construct a
graph that takes an item off the queue, adds one to that item, and puts it back
on the end of the queue. Slowly, the numbers on the queue increase.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/IncremeterFifoQueue.gif">
</div>

`Enqueue`, `EnqueueMany`, and `Dequeue` are special nodes. They take a pointer
to the queue instead of a normal value, allowing them to mutate its state. We
recommend that you think of these operations as being like methods of the queue
in an object-oriented sense. In fact, in the Python API, these operations are
created by calling methods on a queue object (e.g. `q.enqueue(...)`).

Note: Queue methods (such as `q.enqueue(...)`) *must* run on the same device
as the queue. Incompatible device placement directives will be ignored when
creating these operations.

Now that you have a bit of a feel for queues, let's dive into the details...

## Queue usage overview

Queues, such as @{tf.FIFOQueue}
and @{tf.RandomShuffleQueue},
are important TensorFlow objects that aid in computing tensors asynchronously
in a graph.

For example, a typical queue-based input pipeline uses a `RandomShuffleQueue` to
prepare inputs for training a model as follows:

* Multiple threads prepare training examples and enqueue them.
* A training thread executes a training op that dequeues mini-batches from the
  queue

We recommend using the @{tf.data.Dataset.shuffle$`shuffle`}
and @{tf.data.Dataset.batch$`batch`} methods of a
@{tf.data.Dataset$`Dataset`} to accomplish this. However, if you'd prefer
to use a queue-based version instead, you can find a full implementation in the
@{tf.train.shuffle_batch} function.

For demonstration purposes a simplified implementation is given below.

This function takes a source tensor, a capacity, and a batch size as arguments
and returns a tensor that dequeues a shuffled batch when executed.

``` python
def simple_shuffle_batch(source, capacity, batch_size=10):
  # Create a random shuffle queue.
  queue = tf.RandomShuffleQueue(capacity=capacity,
                                min_after_dequeue=int(0.9*capacity),
                                shapes=source.shape, dtypes=source.dtype)

  # Create an op to enqueue one item.
  enqueue = queue.enqueue(source)

  # Create a queue runner that, when started, will launch 4 threads applying
  # that enqueue op.
  num_threads = 4
  qr = tf.train.QueueRunner(queue, [enqueue] * num_threads)

  # Register the queue runner so it can be found and started by
  # `tf.train.start_queue_runners` later (the threads are not launched yet).
  tf.train.add_queue_runner(qr)

  # Create an op to dequeue a batch
  return queue.dequeue_many(batch_size)
```

Once started by @{tf.train.start_queue_runners}, or indirectly through
@{tf.train.MonitoredSession}, the `QueueRunner` will launch the
threads in the background to fill the queue. Meanwhile the main thread will
execute the `dequeue_many` op to pull data from it. Note how these ops do not
depend on each other, except indirectly through the internal state of the queue.

The simplest possible use of this function might be something like this:

``` python
# create a dataset that counts from 0 to 99
input = tf.constant(list(range(100)))
input = tf.data.Dataset.from_tensor_slices(input)
input = input.make_one_shot_iterator().get_next()

# Create a slightly shuffled batch from the sorted elements
get_batch = simple_shuffle_batch(input, capacity=20)

# `MonitoredSession` will start and manage the `QueueRunner` threads.
with tf.train.MonitoredSession() as sess:
  # Since the `QueueRunners` have been started, data is available in the
  # queue, so the `sess.run(get_batch)` call will not hang.
  while not sess.should_stop():
    print(sess.run(get_batch))
```

```
[ 8 10  7  5  4 13 15 14 25  0]
[23 29 28 31 33 18 19 11 34 27]
[12 21 37 39 35 22 44 36 20 46]
...
```

For most use cases, the automatic thread startup and management provided
by @{tf.train.MonitoredSession} is sufficient. In the rare case that it is not,
TensorFlow provides tools for manually managing your threads and queues.

## Manual Thread Management

As we have seen, the TensorFlow `Session` object is multithreaded and
thread-safe, so multiple threads can
easily use the same session and run ops in parallel.  However, it is not always
easy to implement a Python program that drives threads as required.  All
threads must be able to stop together, exceptions must be caught and
reported, and queues must be properly closed when stopping.

TensorFlow provides two classes to help:
@{tf.train.Coordinator} and
@{tf.train.QueueRunner}. These two classes
are designed to be used together. The `Coordinator` class helps multiple threads
stop together and report exceptions to a program that waits for them to stop.
The `QueueRunner` class is used to create a number of threads cooperating to
enqueue tensors in the same queue.

### Coordinator

The @{tf.train.Coordinator} class manages background threads in a TensorFlow
program and helps multiple threads stop together.

Its key methods are:

* @{tf.train.Coordinator.should_stop}: returns `True` if the threads should stop.
* @{tf.train.Coordinator.request_stop}: requests that threads should stop.
* @{tf.train.Coordinator.join}: waits until the specified threads have stopped.

You first create a `Coordinator` object, and then create a number of threads
that use the coordinator.  The threads typically run loops that stop when
`should_stop()` returns `True`.

Any thread can decide that the computation should stop.  It only has to call
`request_stop()` and the other threads will stop as `should_stop()` will then
return `True`.

```python
# Using Python's threading library.
import threading

# Thread body: loop until the coordinator indicates a stop was requested.
# If some condition becomes true, ask the coordinator to stop.
def MyLoop(coord):
  while not coord.should_stop():
    ...do something...
    if ...some condition...:
      coord.request_stop()

# Main thread: create a coordinator.
coord = tf.train.Coordinator()

# Create 10 threads that run 'MyLoop()'
threads = [threading.Thread(target=MyLoop, args=(coord,)) for i in xrange(10)]

# Start the threads and wait for all of them to stop.
for t in threads:
  t.start()
coord.join(threads)
```

Obviously, the coordinator can manage threads doing very different things.
They don't have to be all the same as in the example above.  The coordinator
also has support to capture and report exceptions.  See the @{tf.train.Coordinator} documentation for more details.

### QueueRunner

The @{tf.train.QueueRunner} class creates a number of threads that repeatedly
run an enqueue op.  These threads can use a coordinator to stop together.  In
addition, a queue runner will run a *closer operation* that closes the queue if
an exception is reported to the coordinator.

You can use a queue runner to implement the architecture described above.

First build a graph that uses a TensorFlow queue (e.g. a `tf.RandomShuffleQueue`) for input examples.  Add ops that
process examples and enqueue them in the queue.  Add training ops that start by
dequeueing from the queue.

```python
example = ...ops to create one example...
# Create a queue, and an op that enqueues examples one at a time in the queue.
queue = tf.RandomShuffleQueue(...)
enqueue_op = queue.enqueue(example)
# Create a training graph that starts by dequeueing a batch of examples.
inputs = queue.dequeue_many(batch_size)
train_op = ...use 'inputs' to build the training part of the graph...
```

In the Python training program, create a `QueueRunner` that will run a few
threads to process and enqueue examples.  Create a `Coordinator` and ask the
queue runner to start its threads with the coordinator.  Write a training loop
that also uses the coordinator.

```python
# Create a queue runner that will run 4 threads in parallel to enqueue
# examples.
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

# Launch the graph.
sess = tf.Session()
# Create a coordinator, launch the queue runner threads.
coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
# Run the training loop, controlling termination with the coordinator.
for step in xrange(1000000):
  if coord.should_stop():
    break
  sess.run(train_op)
# When done, ask the threads to stop.
coord.request_stop()
# And wait for them to actually do it.
coord.join(enqueue_threads)
```

### Handling exceptions

Threads started by queue runners do more than just run the enqueue ops.  They
also catch and handle exceptions generated by queues, including the
`tf.errors.OutOfRangeError` exception, which is used to report that a queue was
closed.

A training program that uses a coordinator must similarly catch and report
exceptions in its main loop.

Here is an improved version of the training loop above.

```python
try:
  for step in xrange(1000000):
    if coord.should_stop():
      break
    sess.run(train_op)
except Exception, e:
  # Report exceptions to the coordinator.
  coord.request_stop(e)
finally:
  # Terminate as usual. It is safe to call `coord.request_stop()` twice.
  coord.request_stop()
  coord.join(threads)
```
