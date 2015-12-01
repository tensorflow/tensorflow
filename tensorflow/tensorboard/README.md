# TensorBoard

TensorBoard is a suite of web applications for inspecting and understanding your
TensorFlow runs and graphs. Before running TensorBoard, make sure you have
generated summary data in a log directory by creating a `SummaryWriter`:

```python
# sess.graph_def is the graph definition.
summary_writer = tf.train.SummaryWriter('/path/to/logs', sess.graph_def)
```

For more details, see [this tutorial](http://www.tensorflow.org/how_tos/summaries_and_tensorboard/index.html#serializing-the-data).
Then run TensorBoard and provide the log directory:

```
python tensorflow/tensorboard/tensorboard.py --logdir=path/to/logs
# or if installed via pip, run:
tensorboard --logdir=path/to/logs

# if building from source
bazel build tensorflow/tensorboard:tensorboard
./bazel-bin/tensorflow/tensorboard/tensorboard --logdir=path/to/logs

# then connect to http://localhost:6006
```

Note that TensorBoard requires a `logdir` to read logs from. For info on
configuring TensorBoard, run `tensorboard --help`.

TensorBoard includes a backend (tensorboard.py) that reads TensorFlow event data
from the *tfevents* files, and then  serves this data to the browser. It also
includes a frontend (app/tf-tensorboard.html) that contains html and javascript
for displaying this data in a UI.


## TensorBoard Development Instructions

The following instructions are useful if you want to develop the TensorBoard
frontend in a lightweight frontend-only environment. It sets up gulp with
automatic recompiling and serves just the frontend assets without a connected
backend.

If you just want to use TensorBoard, there is no need to read any further.

### Install Node, npm, gulp, bower, and tsd in your machine
Get nodejs and npm through whatever package distribution system is appropriate
for your machine. For example, on Ubuntu 14.04, run
`sudo apt-get install nodejs nodejs-legacy npm`. Then, run
`sudo npm install -g gulp bower tsd`.

### Install project dependencies

Inside this directory (`tensorflow/tensorboard`),
run the following commands.

    npm install
    bower install
    tsd install

### Run Gulp

Inside this directory, run `gulp`. That will compile all of the
html/js/css dependencies for TensorBoard, and also spin up a server
(by default at port 8000). You can navigate to component-specific demo pages to
check out their behavior.

Running `gulp test` will run all unit tests, the linter, etc.
