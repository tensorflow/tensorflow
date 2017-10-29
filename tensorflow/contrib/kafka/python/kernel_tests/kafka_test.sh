#!/usr/bin/env bash

set -e
set -o pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 start|stop <kafka container name>" >&2
  exit 1
fi

container=$2
if [ "$1" == "start" ]; then
    docker run -d --rm --net=host --name=$container spotify/kafka
    echo Wait 5 secs until kafka is up and running
    sleep 5
    echo Create test topic
    docker exec $container bash -c '/opt/kafka_2.11-0.10.1.0/bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test'
    echo Create test message
    docker exec $container bash -c 'echo -e "D0\nD1\nD2\nD3\nD4\nD5\nD6\nD7\nD8\nD9" > /test'
    echo Produce test message
    docker exec $container bash -c '/opt/kafka_2.11-0.10.1.0/bin/kafka-console-producer.sh --topic test --broker-list 127.0.0.1:9092 < /test'

    echo Container $container started successfully
elif [ "$1" == "stop" ]; then
    docker rm -f $container

    echo Container $container stopped successfully
else
  echo "Usage: $0 start|stop <kafka container name>" >&2
  exit 1
fi



