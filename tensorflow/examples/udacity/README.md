Assignments for Udacity Deep Learning class with TensorFlow
===========================================================

Running the Docker container from the Google Cloud repository
-------------------------------------------------------------

    docker run -p 8888:8888 -it --rm b.gcr.io/tensorflow-udacity/assignments

Accessing the Notebooks
-----------------------

On linux, go to: http://127.0.0.1:8888

On mac, find the virtual machine's IP using:

    docker-machine ip default

Then go to: http://IP:8888 (likely http://192.168.99.100:8888)

Building a local Docker container
---------------------------------

    cd tensorflow/examples/udacity
    docker build -t $USER/assignments .

Running the local container
---------------------------

    docker run -p 8888:8888 -it --rm $USER/assignments

Pushing a Google Cloud release
------------------------------

    V=0.1.0
    docker tag $USER/assignments b.gcr.io/tensorflow-udacity/assignments:$V
    docker tag $USER/assignments b.gcr.io/tensorflow-udacity/assignments:latest
    gcloud docker push b.gcr.io/tensorflow-udacity/assignments
