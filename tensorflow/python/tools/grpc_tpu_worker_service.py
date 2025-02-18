# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=g-import-not-at-top
"""Script to start GRPC worker as a service.

Usage:
    python3 grpc_tpu_worker_service.py
"""

import os
import subprocess
import sys


def get_sys_path():
  return ":".join(sys.path)


def get_username():
  return os.environ.get("USER")


username = get_username()
sys_path = get_sys_path()

SERVICE_FILE_CONTENT = f"""
[Unit]
Description=GRPC TPU Worker Service
After=network.target
[Service]
Type=simple
Environment="PYTHONPATH=$PYTHONPATH{sys_path}"
EnvironmentFile=/home/tpu-runtime/tpu-env
#ExecStartPre=/bin/mkdir -p /tmp/tflogs
#ExecStartPre=/bin/touch /tmp/tflogs/grpc_tpu_worker.log
#ExecStartPre=/bin/chmod +r /tmp/tflogs
ExecStart=/home/{get_username()}/.local/bin/start_grpc_tpu_worker #2>&1 | tee -a /tmp/tflogs/grpc_tpu_worker.log
Restart=on-failure
# Restart service after 10 seconds if the service crashes:
RestartSec=10
[Install]
WantedBy=multi-user.target
"""
SERVICE_NAME = "grpc_tpu_worker.service"


def create_systemd_service_file(service_content, service_name):
  with open(service_name, "w") as file:
    file.write(service_content)
  print(f"Service file {service_name} created")


def move_file_to_systemd(service_name):
  if not os.path.exists("~/.config/systemd/user/"):
    mkdir_command = "mkdir -p ~/.config/systemd/user"
    subprocess.run(mkdir_command, shell=True, check=True)
    print("Created directory ~/.config/systemd/user/")
  command = f"mv {service_name} ~/.config/systemd/user/{service_name}"
  subprocess.run(command, shell=True, check=True)
  print(f"Service file moved to ~/.config/systemd/user/{service_name}")


def enable_start_service(service_name):
  commands = [
      "systemctl --user import-environment",
      "systemctl --user daemon-reload",
      f"systemctl --user enable {service_name}",
      f"systemctl --user start {service_name}",
  ]
  for command in commands:
    subprocess.run(command, shell=True, check=True)
    print(f"Executed: {command}")


def run():
  if os.path.exists(f"~/.config/systemd/user/{SERVICE_NAME}"):
    print(f"Service file ~/.config/systemd/user/{SERVICE_NAME} already exists")
    sys.exit(1)
  else:
    create_systemd_service_file(SERVICE_FILE_CONTENT, SERVICE_NAME)
    move_file_to_systemd(SERVICE_NAME)
    enable_start_service(SERVICE_NAME)


if __name__ == "__main__":
  run()
