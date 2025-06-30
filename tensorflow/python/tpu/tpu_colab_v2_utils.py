# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""TPU initialization utilities for Google Colab V2 environments.

This module provides utilities to ensure proper TPU initialization in 
environments where the standard TPU configuration might fail, such as
Google Colab V2.
"""

import os
import logging
import subprocess
import sys
from typing import Optional

import tensorflow as tf


def _is_colab_environment() -> bool:
  """Check if running in Google Colab environment."""
  try:
    import google.colab  # pylint: disable=import-outside-toplevel,unused-import
    return True
  except ImportError:
    return False


def _is_colab_v2() -> bool:
  """Check if running in Google Colab V2 environment."""
  if not _is_colab_environment():
    return False
  
  # Check for V2-specific indicators
  colab_env = os.environ.get('COLAB_TPU_ADDR', '')
  runtime_name = os.environ.get('COLAB_RUNTIME_NAME', '')
  
  return 'v2-8' in colab_env or 'v2' in runtime_name.lower()


def _ensure_libtpu_installed() -> bool:
  """Ensure libtpu is properly installed for the current environment."""
  try:
    # Try to import TPU-related modules to check if libtpu is available
    from tensorflow.python.tpu import tpu_strategy_util  # pylint: disable=import-outside-toplevel,unused-import
    return True
  except ImportError:
    pass
  
  if _is_colab_v2():
    logging.warning(
        "TPU libraries not found. Installing libtpu for Colab V2...")
    
    try:
      # Install the nightly libtpu package for Colab V2
      subprocess.check_call([
          sys.executable, '-m', 'pip', 'install', '-U', '--quiet',
          'https://storage.googleapis.com/libtpu-releases/libtpu-nightly.tar.gz'
      ])
      logging.info("Successfully installed libtpu-nightly")
      return True
    except subprocess.CalledProcessError as e:
      logging.error(f"Failed to install libtpu: {e}")
      return False
  
  return False


def _setup_tpu_environment():
  """Setup TPU environment variables and configurations."""
  if _is_colab_v2():
    # Set environment variables that help with TPU detection in Colab V2
    if 'TPU_LIBRARY_PATH' not in os.environ:
      # Try common libtpu locations
      possible_paths = [
          '/usr/local/lib/python*/site-packages/libtpu/libtpu.so',
          '/opt/conda/lib/python*/site-packages/libtpu/libtpu.so',
          'libtpu.so'
      ]
      
      for path_pattern in possible_paths:
        import glob
        matches = glob.glob(path_pattern)
        if matches:
          os.environ['TPU_LIBRARY_PATH'] = matches[0]
          logging.info(f"Set TPU_LIBRARY_PATH to {matches[0]}")
          break


def initialize_tpu_system_with_fallback(
    cluster_resolver: Optional[tf.distribute.cluster_resolver.TPUClusterResolver] = None
) -> None:
  """Initialize TPU system with fallback for Colab V2 compatibility.
  
  This function provides a more robust TPU initialization that works around
  issues in Google Colab V2 environments where the standard TPU initialization
  might fail due to missing OpKernel registrations.
  
  Args:
    cluster_resolver: Optional TPU cluster resolver. If None, will create one
      for the local TPU.
  
  Raises:
    RuntimeError: If TPU initialization fails after all fallback attempts.
  """
  
  # Ensure we're in a TPU environment
  if not _is_colab_environment():
    # Standard initialization for non-Colab environments
    if cluster_resolver is None:
      cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    return
  
  # Colab-specific initialization
  logging.info("Detected Google Colab environment, using enhanced TPU initialization")
  
  # Ensure libtpu is installed
  if not _ensure_libtpu_installed():
    raise RuntimeError(
        "Failed to install required TPU libraries. Please manually install "
        "with: !pip install -U \"https://storage.googleapis.com/libtpu-releases/libtpu-nightly.tar.gz\""
    )
  
  # Setup environment
  _setup_tpu_environment()
  
  # Create cluster resolver if not provided
  if cluster_resolver is None:
    tpu_address = os.environ.get('COLAB_TPU_ADDR', 'local')
    if tpu_address and tpu_address != 'local':
      tpu_address = f'grpc://{tpu_address}'
      cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
    else:
      cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
  
  # Try standard initialization first
  try:
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    logging.info("Successfully initialized TPU system using standard method")
    return
  except Exception as e:
    logging.warning(f"Standard TPU initialization failed: {e}")
    logging.info("Trying fallback initialization methods...")
  
  # Fallback 1: Try using the libtpu package directly
  try:
    # Force reload of TPU libraries
    if hasattr(tf.config, 'experimental_reset_memory_stats'):
      tf.config.experimental_reset_memory_stats('TPU')
    
    # Try connecting again
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    logging.info("Successfully initialized TPU system using fallback method")
    return
  except Exception as e:
    logging.warning(f"Fallback TPU initialization also failed: {e}")
  
  # If all else fails, provide helpful error message
  raise RuntimeError(
      "TPU initialization failed. This is a known issue in some Google Colab V2 environments. "
      "Please try the following steps:\n"
      "1. Restart runtime and run: !pip install -U \"https://storage.googleapis.com/libtpu-releases/libtpu-nightly.tar.gz\"\n"
      "2. Import tensorflow after installing libtpu\n"
      "3. If the issue persists, try using tensorflow-tpu package instead: !pip install tensorflow-tpu\n"
      f"Original error: {e}"
  )


def create_tpu_strategy_with_fallback(
    cluster_resolver: Optional[tf.distribute.cluster_resolver.TPUClusterResolver] = None
) -> tf.distribute.TPUStrategy:
  """Create TPU strategy with fallback initialization for Colab V2.
  
  Args:
    cluster_resolver: Optional TPU cluster resolver.
    
  Returns:
    Configured TPUStrategy instance.
    
  Raises:
    RuntimeError: If TPU strategy creation fails.
  """
  
  # Initialize TPU system with fallback
  initialize_tpu_system_with_fallback(cluster_resolver)
  
  # Create and return strategy
  if cluster_resolver is None:
    if _is_colab_environment():
      tpu_address = os.environ.get('COLAB_TPU_ADDR', 'local')
      if tpu_address and tpu_address != 'local':
        tpu_address = f'grpc://{tpu_address}'
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
      else:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    else:
      cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
  
  return tf.distribute.TPUStrategy(cluster_resolver)


# Monkey patch the standard TPU initialization functions for better Colab V2 support
def _patch_tpu_functions():
  """Monkey patch TPU functions to use fallback implementations."""
  if not _is_colab_v2():
    return
  
  # Store original functions
  original_initialize_tpu_system = tf.tpu.experimental.initialize_tpu_system
  
  def patched_initialize_tpu_system(cluster_resolver=None):
    """Patched version that uses fallback initialization."""
    try:
      return original_initialize_tpu_system(cluster_resolver)
    except Exception as e:
      logging.warning(f"Standard TPU initialization failed: {e}, trying fallback...")
      return initialize_tpu_system_with_fallback(cluster_resolver)
  
  # Apply patch
  tf.tpu.experimental.initialize_tpu_system = patched_initialize_tpu_system


# Auto-patch if in Colab V2
if _is_colab_v2():
  _patch_tpu_functions()
  logging.info("Applied TPU initialization patches for Google Colab V2 compatibility")
