# Testing Distributed Runtime in TensorFlow
This folder containers tools and test suites for the GRPC-based distributed
runtime in TensorFlow.

There are three general modes of testing:

**1) Launch a local Kubernetes (k8s) cluster and run the test suites on it**

For example:

    ./local_test.sh

This option makes use of the docker-in-docker (dind) containers. It requires
the docker0 network interface to be set to the promiscuous mode on the host:

    sudo ip link set docker0 promisc on

The environment variable "TF_DIST_SERVER_DOCKER_IMAGE" can be used to override
the Docker image used to generate the TensorFlow GRPC server pods
("tensorflow/tf_grpc_test_server"). For example:

    export TF_DIST_SERVER_DOCKER_IMAGE=<docker_image_name>
    ./local_test.sh

**2) Launch a remote k8s cluster on Google Container Engine (GKE) and run the
test suite on it**

For example:

    export TF_DIST_GCLOUD_PROJECT="tensorflow-testing"
    export TF_DIST_GCLOUD_COMPUTE_ZONE="us-central1-f"
    export CONTAINER_CLUSTER="test-cluster-1"
    export TF_DIST_GCLOUD_KEY_FILE_DIR="/tmp/gcloud-secrets"
    ./remote_test.sh

Here you specify the Google Compute Engine (GCE) project, compute zone and
container cluster with the first three environment variables, in that order.
The environment variable "TF_DIST_GCLOUD_KEY_FILE_DIR" is a directory in which
the JSON service account key file named "tensorflow-testing.json" is located.
You can use the flag "--setup-cluster-only" to perform only the cluster setup
step and skip the testing step:

    ./remote_test.sh --setup-cluster-only

**3) Run the test suite on an existing k8s TensorFlow cluster**

For example:

    export TF_DIST_GRPC_SERVER_URL="grpc://11.22.33.44:2222"
    ./remote_test.sh

The IP address above is a dummy example. Such a cluster may have been set up
using the command described at the end of the previous section.


**Building the test server Docker image**

To build the Docker image for a test server of TensorFlow distributed runtime,
run:

    ./build_server.sh <docker_image_name>


**Generating configuration file for TensorFlow k8s clusters**

The script at "scripts/k8s_tensorflow.py" can be used to generate yaml
configuration files for a TensorFlow k8s cluster consisting of a number of
workers and parameter servers. For example:

    scripts/k8s_tensorflow.py \
        --num_workers 2 \
        --num_parameter_servers 2 \
        --grpc_port 2222 \
        --request_load_balancer \
        --docker_image "tensorflow/tf_grpc_test_server" \
        > tf-k8s-with-lb.yaml
