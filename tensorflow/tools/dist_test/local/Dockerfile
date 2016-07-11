FROM jpetazzo/dind

MAINTAINER Shanqing Cai <cais@google.com>

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    build-essential \
    git \
    software-properties-common

# Install the latest golang
RUN wget https://storage.googleapis.com/golang/go1.4.2.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.4.2.linux-amd64.tar.gz
RUN rm -f go1.4.2.linux-amd64.tar.gz
RUN echo 'PATH=/usr/local/go/bin:${PATH}' >> /root/.bashrc

ADD start_local_k8s_cluster.sh /var/k8s/start_local_k8s_cluster.sh
ADD ../scripts /var/k8s/dist_test/scripts
ADD ../python /var/k8s/dist_test/python
