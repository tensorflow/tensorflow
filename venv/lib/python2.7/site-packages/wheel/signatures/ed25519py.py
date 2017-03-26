# -*- coding: utf-8 -*-

import warnings
import os

from collections import namedtuple
from . import djbec

__all__ = ['crypto_sign', 'crypto_sign_open', 'crypto_sign_keypair', 'Keypair',
           'PUBLICKEYBYTES', 'SECRETKEYBYTES', 'SIGNATUREBYTES']

PUBLICKEYBYTES=32
SECRETKEYBYTES=64
SIGNATUREBYTES=64

Keypair = namedtuple('Keypair', ('vk', 'sk')) # verifying key, secret key

def crypto_sign_keypair(seed=None):
    """Return (verifying, secret) key from a given seed, or os.urandom(32)"""    
    if seed is None:
        seed = os.urandom(PUBLICKEYBYTES)
    else:
        warnings.warn("ed25519ll should choose random seed.",
                      RuntimeWarning)
    if len(seed) != 32:
        raise ValueError("seed must be 32 random bytes or None.")
    skbytes = seed
    vkbytes = djbec.publickey(skbytes)
    return Keypair(vkbytes, skbytes+vkbytes)


def crypto_sign(msg, sk):
    """Return signature+message given message and secret key.
    The signature is the first SIGNATUREBYTES bytes of the return value.
    A copy of msg is in the remainder."""
    if len(sk) != SECRETKEYBYTES:
        raise ValueError("Bad signing key length %d" % len(sk))
    vkbytes = sk[PUBLICKEYBYTES:]
    skbytes = sk[:PUBLICKEYBYTES]
    sig = djbec.signature(msg, skbytes, vkbytes)
    return sig + msg


def crypto_sign_open(signed, vk):
    """Return message given signature+message and the verifying key."""
    if len(vk) != PUBLICKEYBYTES:
        raise ValueError("Bad verifying key length %d" % len(vk))
    rc = djbec.checkvalid(signed[:SIGNATUREBYTES], signed[SIGNATUREBYTES:], vk)
    if not rc:
        raise ValueError("rc != True", rc)    
    return signed[SIGNATUREBYTES:]

