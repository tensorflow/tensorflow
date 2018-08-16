#!/usr/bin/env bash

# My Docker requires root, but for other hosts it might not be required.
sudo ./start-docker.sh

sleep 5

python3 test-plain.py
python3 test-ssl.py
python3 test-ssl-auth.py

# My Docker requires root, but for other hosts it might not be required.
sudo ./stop-docker.sh
