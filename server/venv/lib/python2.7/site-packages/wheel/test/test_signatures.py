from wheel import signatures
from wheel.signatures import djbec, ed25519py
from wheel.util import binary

def test_getlib():
    signatures.get_ed25519ll()

def test_djbec():
    djbec.dsa_test()    
    djbec.dh_test()
    
def test_ed25519py():
    kp0 = ed25519py.crypto_sign_keypair(binary(' '*32))
    kp = ed25519py.crypto_sign_keypair()
        
    signed = ed25519py.crypto_sign(binary('test'), kp.sk)
    
    ed25519py.crypto_sign_open(signed, kp.vk)
    
    try:
        ed25519py.crypto_sign_open(signed, kp0.vk)
    except ValueError:
        pass
    else:
        raise Exception("Expected ValueError")
    
    try:
        ed25519py.crypto_sign_keypair(binary(' '*33))
    except ValueError:
        pass
    else:
        raise Exception("Expected ValueError")
    
    try:
        ed25519py.crypto_sign(binary(''), binary(' ')*31)
    except ValueError:
        pass
    else:
        raise Exception("Expected ValueError")
    
    try:
        ed25519py.crypto_sign_open(binary(''), binary(' ')*31)
    except ValueError:
        pass
    else:
        raise Exception("Expected ValueError")
    