Assignments for Udacity Deep Learning class with TensorFlow
===========================================================

Course information can be found at https://www.udacity.com/course/deep-learning--ud730

Running the Docker container from the Google Cloud repository
-------------------------------------------------------------

    docker run -p 8888:8888 --name tensorflow-udacity -it gcr.io/tensorflow/udacity-assignments:1.0.0

Note that if you ever exit the container, you can return to it using:

    docker start -ai tensorflow-udacity

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

* **I want to create a new virtual machine instead of the default one.**

`docker-machine` is a tool to provision and manage docker hosts, it supports multiple platform (ex. aws, gce, azure, virtualbox, ...). To create a new virtual machine locally with built-in docker engine, you can use

    docker-machine create -d virtualbox --virtualbox-memory 8196 tensorflow

`-d` means the driver for the cloud platform, supported drivers listed [here](https://docs.docker.com/machine/drivers/). Here we use virtualbox to create a new virtual machine locally. `tensorflow` means the name of the virtual machine, feel free to use whatever you like. You can use

    docker-machine ip tensorflow

to get the ip of the new virtual machine. To switch from default virtual machine to a new one (here we use tensorflow), type

    eval $(docker-machine env tensorflow)

Note that `docker-machine env tensorflow` outputs some environment variables such like `DOCKER_HOST`. Then your docker client is now connected to the docker host in virtual machine `tensorflow`

* **I'm getting a TLS connection error.**

If you get an error about the TLS connection of your docker, run the command below to confirm the problem.

	docker-machine ip tensorflow

Then if it is the case use the instructions on [this page](https://docs.docker.com/toolbox/faqs/troubleshoot/) to solve the issue.


* **I'm getting the error - docker: Cannot connect to the Docker daemon. Is the docker daemon running on this host? - when I run 'docker run'.**

This is a permissions issue, and a popular answer is provided for Linux and Max OSX [here](http://stackoverflow.com/questions/21871479/docker-cant-connect-to-docker-daemon) on StackOverflow.

Notes for anyone needing to build their own containers (mostly instructors)
===========================================================================

Building a local Docker container
---------------------------------

    cd tensorflow/examples/udacity
    docker build --pull -t $USER/assignments .

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

    V=1.0.0
    docker tag $USER/assignments gcr.io/tensorflow/udacity-assignments:$V
    gcloud docker push gcr.io/tensorflow/udacity-assignments
    docker tag $USER/assignments gcr.io/tensorflow/udacity-assignments:latest
    gcloud docker push gcr.io/tensorflow/udacity-assignments

History
-------

* 0.1.0: Initial release.
* 0.2.0: Many fixes, including lower memory footprint and support for Python 3.
* 0.3.0: Use 0.7.1 release.
* 0.4.0: Move notMMNIST data for Google Cloud.
* 0.5.0: Actually use 0.7.1 release.
* 0.6.0: Update to TF 0.10.0, add libjpeg (for Pillow).
* 1.0.0: Update to TF 1.0.0 release.
