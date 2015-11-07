# TensorBoard

TensorBoard is a suite of web applications for inspecting and understanding your
TensorFlow runs and graphs.

Example Usage:

```
python tensorflow/tensorboard/tensorboard.py --logdir=path/to/logs
# if installed via pip
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


## Building the TensorBoard frontend

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

### Run Gulp Vulcanize

Inside this directory, run `gulp vulcanize`. That will compile all of the
html/js/css dependencies for TensorBoard into a monolithic index.html file under
dist/. Once you've done this, you can locally run your own TensorBoard instance
and it will have a working frontend.

### Frontend General Dev Instructions

To speed up the development process, we can run the frontend code independently
of the backend, and mock out the backend with static JSON files. This allows
testing the frontend's correctness without needing to find  real data and spin
up a real server. Look at app/demo/index.html for an example.

The following gulp commands are useful:

* `gulp test` - build, test, and lint the code
* `gulp watch` - build, test, and rebuild on change
* `gulp server` - start a livereload server on localhost:8000
* `gulp` - alias for `gulp watch`
* `gulp vulcanize` -
