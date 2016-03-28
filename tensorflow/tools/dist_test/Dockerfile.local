FROM jpetazzo/dind

MAINTAINER Shanqing Cai <cais@google.com>

RUN apt-get update

RUN apt-get install -y \
    bc \
    build-essential \
    dbus \
    git \
    software-properties-common

# Install the latest golang
RUN wget https://storage.googleapis.com/golang/go1.4.2.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.4.2.linux-amd64.tar.gz
RUN rm -f go1.4.2.linux-amd64.tar.gz
RUN echo 'PATH=/usr/local/go/bin:${PATH}' >> /root/.bashrc

ADD . /var/tf-k8s
