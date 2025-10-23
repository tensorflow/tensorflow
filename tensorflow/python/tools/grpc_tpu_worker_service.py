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
import shutil
import subprocess
import sys
from pathlib import Path


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
  user_systemd_dir = Path.home() / ".config" / "systemd" / "user"
  if not user_systemd_dir.exists():
    user_systemd_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory {user_systemd_dir}")

  source_file = Path(service_name)
  dest_file = user_systemd_dir / service_name
  shutil.move(str(source_file), str(dest_file))
  print(f"Service file moved to {dest_file}")


def enable_start_service(service_name):
  commands = [
      ["systemctl", "--user", "import-environment"],
      ["systemctl", "--user", "daemon-reload"],
      ["systemctl", "--user", "enable", service_name],
      ["systemctl", "--user", "start", service_name],
  ]
  for command in commands:
    subprocess.run(command, check=True)
    print(f"Executed: {' '.join(command)}")


def run():
  service_file_path = (
      Path.home() / ".config" / "systemd" / "user" / SERVICE_NAME
  )
  if service_file_path.exists():
    print(f"Service file {service_file_path} already exists")
    sys.exit(1)
  else:
    create_systemd_service_file(SERVICE_FILE_CONTENT, SERVICE_NAME)
    move_file_to_systemd(SERVICE_NAME)
    enable_start_service(SERVICE_NAME)


if __name__ == "__main__":
  run()
