# Ed25519 digital signatures
# Based on http://ed25519.cr.yp.to/python/ed25519.py
# See also http://ed25519.cr.yp.to/software.html
# Adapted by Ron Garret
# Sped up considerably using coordinate transforms found on:
# http://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html
# Specifically add-2008-hwcd-4 and dbl-2008-hwcd

try: # pragma nocover
    unicode
    PY3 = False
    def asbytes(b):
        """Convert array of integers to byte string"""
        return ''.join(chr(x) for x in b)
    def joinbytes(b):
        """Convert array of bytes to byte string"""
        return ''.join(b)
    def bit(h, i):
        """Return i'th bit of bytestring h"""
        return (ord(h[i//8]) >> (i%8)) & 1

except NameError: # pragma nocover
    PY3 = True
    asbytes = bytes
    joinbytes = bytes
    def bit(h, i):
        return (h[i//8] >> (i%8)) & 1

import hashlib

b = 256
q = 2**255 - 19
l = 2**252 + 27742317777372353535851937790883648493

def H(m):
    return hashlib.sha512(m).digest()

def expmod(b, e, m):
    if e == 0: return 1
    t = expmod(b, e // 2, m) ** 2 % m
    if e & 1: t = (t * b) % m
    return t

# Can probably get some extra speedup here by replacing this with
# an extended-euclidean, but performance seems OK without that
def inv(x):
    return expmod(x, q-2, q)

d = -121665 * inv(121666)
I = expmod(2,(q-1)//4,q)

def xrecover(y):
    xx = (y*y-1) * inv(d*y*y+1)
    x = expmod(xx,(q+3)//8,q)
    if (x*x - xx) % q != 0: x = (x*I) % q
    if x % 2 != 0: x = q-x
    return x

By = 4 * inv(5)
Bx = xrecover(By)
B = [Bx % q,By % q]

#def edwards(P,Q):
#    x1 = P[0]
#    y1 = P[1]
#    x2 = Q[0]
#    y2 = Q[1]
#    x3 = (x1*y2+x2*y1) * inv(1+d*x1*x2*y1*y2)
#    y3 = (y1*y2+x1*x2) * inv(1-d*x1*x2*y1*y2)
#    return (x3 % q,y3 % q)

#def scalarmult(P,e):
#    if e == 0: return [0,1]
#    Q = scalarmult(P,e/2)
#    Q = edwards(Q,Q)
#    if e & 1: Q = edwards(Q,P)
#    return Q

# Faster (!) version based on:
# http://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html

def xpt_add(pt1, pt2):
    (X1, Y1, Z1, T1) = pt1
    (X2, Y2, Z2, T2) = pt2
    A = ((Y1-X1)*(Y2+X2)) % q
    B = ((Y1+X1)*(Y2-X2)) % q
    C = (Z1*2*T2) % q
    D = (T1*2*Z2) % q
    E = (D+C) % q
    F = (B-A) % q
    G = (B+A) % q
    H = (D-C) % q
    X3 = (E*F) % q
    Y3 = (G*H) % q
    Z3 = (F*G) % q
    T3 = (E*H) % q
    return (X3, Y3, Z3, T3)

def xpt_double (pt):
    (X1, Y1, Z1, _) = pt
    A = (X1*X1)
    B = (Y1*Y1)
    C = (2*Z1*Z1)
    D = (-A) % q
    J = (X1+Y1) % q
    E = (J*J-A-B) % q
    G = (D+B) % q
    F = (G-C) % q
    H = (D-B) % q
    X3 = (E*F) % q
    Y3 = (G*H) % q
    Z3 = (F*G) % q
    T3 = (E*H) % q
    return (X3, Y3, Z3, T3)

def pt_xform (pt):
    (x, y) = pt
    return (x, y, 1, (x*y)%q)

def pt_unxform (pt):
    (x, y, z, _) = pt
    return ((x*inv(z))%q, (y*inv(z))%q)

def xpt_mult (pt, n):
    if n==0: return pt_xform((0,1))
    _ = xpt_double(xpt_mult(pt, n>>1))
    return xpt_add(_, pt) if n&1 else _

def scalarmult(pt, e):
    return pt_unxform(xpt_mult(pt_xform(pt), e))

def encodeint(y):
    bits = [(y >> i) & 1 for i in range(b)]
    e = [(sum([bits[i * 8 + j] << j for j in range(8)]))
                                    for i in range(b//8)]
    return asbytes(e)

def encodepoint(P):
    x = P[0]
    y = P[1]
    bits = [(y >> i) & 1 for i in range(b - 1)] + [x & 1]
    e = [(sum([bits[i * 8 + j] << j for j in range(8)]))
                                    for i in range(b//8)]
    return asbytes(e)
    
def publickey(sk):
    h = H(sk)
    a = 2**(b-2) + sum(2**i * bit(h,i) for i in range(3,b-2))
    A = scalarmult(B,a)
    return encodepoint(A)

def Hint(m):
    h = H(m)
    return sum(2**i * bit(h,i) for i in range(2*b))

def signature(m,sk,pk):
    h = H(sk)
    a = 2**(b-2) + sum(2**i * bit(h,i) for i in range(3,b-2))
    inter = joinbytes([h[i] for i in range(b//8,b//4)])
    r = Hint(inter + m)
    R = scalarmult(B,r)
    S = (r + Hint(encodepoint(R) + pk + m) * a) % l
    return encodepoint(R) + encodeint(S)

def isoncurve(P):
    x = P[0]
    y = P[1]
    return (-x*x + y*y - 1 - d*x*x*y*y) % q == 0

def decodeint(s):
    return sum(2**i * bit(s,i) for i in range(0,b))

def decodepoint(s):
    y = sum(2**i * bit(s,i) for i in range(0,b-1))
    x = xrecover(y)
    if x & 1 != bit(s,b-1): x = q-x
    P = [x,y]
    if not isoncurve(P): raise Exception("decoding point that is not on curve")
    return P

def checkvalid(s, m, pk):
    if len(s) != b//4: raise Exception("signature length is wrong")
    if len(pk) != b//8: raise Exception("public-key length is wrong")
    R = decodepoint(s[0:b//8])
    A = decodepoint(pk)
    S = decodeint(s[b//8:b//4])
    h = Hint(encodepoint(R) + pk + m)
    v1 = scalarmult(B,S)
#  v2 = edwards(R,scalarmult(A,h))
    v2 = pt_unxform(xpt_add(pt_xform(R), pt_xform(scalarmult(A, h))))
    return v1==v2

##########################################################
#
# Curve25519 reference implementation by Matthew Dempsky, from:
# http://cr.yp.to/highspeed/naclcrypto-20090310.pdf

# P = 2 ** 255 - 19
P = q
A = 486662

#def expmod(b, e, m):
#    if e == 0: return 1
#    t = expmod(b, e / 2, m) ** 2 % m
#    if e & 1: t = (t * b) % m
#    return t

# def inv(x): return expmod(x, P - 2, P)

def add(n, m, d):
    (xn, zn) = n
    (xm, zm) = m 
    (xd, zd) = d
    x = 4 * (xm * xn - zm * zn) ** 2 * zd
    z = 4 * (xm * zn - zm * xn) ** 2 * xd
    return (x % P, z % P)

def double(n):
    (xn, zn) = n
    x = (xn ** 2 - zn ** 2) ** 2
    z = 4 * xn * zn * (xn ** 2 + A * xn * zn + zn ** 2)
    return (x % P, z % P)

def curve25519(n, base=9):
    one = (base,1)
    two = double(one)
    # f(m) evaluates to a tuple
    # containing the mth multiple and the
    # (m+1)th multiple of base.
    def f(m):
        if m == 1: return (one, two)
        (pm, pm1) = f(m // 2)
        if (m & 1):
            return (add(pm, pm1, one), double(pm1))
        return (double(pm), add(pm, pm1, one))
    ((x,z), _) = f(n)
    return (x * inv(z)) % P

import random

def genkey(n=0):
    n = n or random.randint(0,P)
    n &= ~7
    n &= ~(128 << 8 * 31)
    n |= 64 << 8 * 31
    return n

#def str2int(s):
#    return int(hexlify(s), 16)
#    # return sum(ord(s[i]) << (8 * i) for i in range(32))
#
#def int2str(n):
#    return unhexlify("%x" % n)
#    # return ''.join([chr((n >> (8 * i)) & 255) for i in range(32)])

#################################################

def dsa_test():
    import os
    msg = str(random.randint(q,q+q)).encode('utf-8')
    sk = os.urandom(32)
    pk = publickey(sk)
    sig = signature(msg, sk, pk)
    return checkvalid(sig, msg, pk)

def dh_test():
    sk1 = genkey()
    sk2 = genkey()
    return curve25519(sk1, curve25519(sk2)) == curve25519(sk2, curve25519(sk1))

