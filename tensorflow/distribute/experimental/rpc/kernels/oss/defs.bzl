"""Helper for GRPC credentials dependency for RPC Ops for OSS"""

def grpc_credentials_dependency():
    """Returns credentials dependency for RPC OPs"""
    return ["//tensorflow/distribute/experimental/rpc/kernels/oss:grpc_credentials"]
