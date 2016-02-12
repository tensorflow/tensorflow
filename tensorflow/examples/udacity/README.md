Assignments for Udacity Deep Learning class with TensorFlow
===========================================================

Course information can be found at https://www.udacity.com/course/deep-learning--ud730

Running the Docker container from the Google Cloud repository
-------------------------------------------------------------

    docker run -p 8888:8888 -it --rm b.gcr.io/tensorflow-udacity/assignments

Accessing the Notebooks
-----------------------

On linux, go to: http://127.0.0.1:8888

On mac, find the virtual machine's IP using:

    docker-machine ip default

Then go to: http://IP:8888 (likely http://192.168.99.100:8888)

FAQ
---

* **I'm getting a MemoryError when loading data in the first notebook.**

If you're using a Mac, Docker works by running a VM locally (which
is controlled by `docker-machine`). It's quite likely that you'll
need to bump up the amount of RAM allocated to the VM beyond the
default (which is 1G).
[This Stack Overflow question](http://stackoverflow.com/questions/32834082/how-to-increase-docker-machine-memory-mac)
has two good suggestions; we recommend using 8G.

In addition, you may need to pass `--memory=8g` as an extra argument to
`docker run`.

Notes for anyone needing to build their own containers (mostly instructors)
===========================================================================

Building a local Docker container
---------------------------------

    cd tensorflow/examples/udacity
    docker build -t $USER/assignments .

Running the local container
---------------------------

To run a disposable container:

    docker run -p 8888:8888 -it --rm $USER/assignments

Note the above command will create an ephemeral container and all data stored in the container will be lost when the container stops.

To avoid losing work between sessions in the container, it is recommended that you mount the `tensorflow/examples/udacity` directory into the container:

    docker run -p 8888:8888 -v </path/to/tensorflow/examples/udacity>:/notebooks -it --rm $USER/assignments

This will allow you to save work and have access to generated files on the host filesystem.

Pushing a Google Cloud release
------------------------------

    V=0.2.0
    docker tag $USER/assignments b.gcr.io/tensorflow-udacity/assignments:$V
    docker tag $USER/assignments b.gcr.io/tensorflow-udacity/assignments:latest
    gcloud docker push b.gcr.io/tensorflow-udacity/assignments

History
-------

* 0.1.0: Initial release.
* 0.2.0: Many fixes, including lower memory footprint and support for Python 3.
