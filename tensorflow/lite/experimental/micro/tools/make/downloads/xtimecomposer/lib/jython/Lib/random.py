"""Random variable generators.

    integers
    --------
           uniform within range

    sequences
    ---------
           pick random element
           pick random sample
           generate random permutation

    distributions on the real line:
    ------------------------------
           uniform
           normal (Gaussian)
           lognormal
           negative exponential
           gamma
           beta
           pareto
           Weibull

    distributions on the circle (angles 0 to 2pi)
    ---------------------------------------------
           circular uniform
           von Mises

General notes on the underlying Mersenne Twister core generator:

* The period is 2**19937-1.
* It is one of the most extensively tested generators in existence.
* Without a direct way to compute N steps forward, the semantics of
  jumpahead(n) are weakened to simply jump to another distant state and rely
  on the large period to avoid overlapping sequences.
* The random() method is implemented in C, executes in a single Python step,
  and is, therefore, threadsafe.

"""

from warnings import warn as _warn
from types import MethodType as _MethodType, BuiltinMethodType as _BuiltinMethodType
from math import log as _log, exp as _exp, pi as _pi, e as _e, ceil as _ceil
from math import sqrt as _sqrt, acos as _acos, cos as _cos, sin as _sin
from os import urandom as _urandom
from binascii import hexlify as _hexlify

__all__ = ["Random","seed","random","uniform","randint","choice","sample",
           "randrange","shuffle","normalvariate","lognormvariate",
           "expovariate","vonmisesvariate","gammavariate",
           "gauss","betavariate","paretovariate","weibullvariate",
           "getstate","setstate","jumpahead", "WichmannHill", "getrandbits",
           "SystemRandom"]

NV_MAGICCONST = 4 * _exp(-0.5)/_sqrt(2.0)
TWOPI = 2.0*_pi
LOG4 = _log(4.0)
SG_MAGICCONST = 1.0 + _log(4.5)
BPF = 53        # Number of bits in a float
RECIP_BPF = 2**-BPF


# Translated by Guido van Rossum from C source provided by
# Adrian Baddeley.  Adapted by Raymond Hettinger for use with
# the Mersenne Twister  and os.urandom() core generators.

import _random

class Random(_random.Random):
    """Random number generator base class used by bound module functions.

    Used to instantiate instances of Random to get generators that don't
    share state.  Especially useful for multi-threaded programs, creating
    a different instance of Random for each thread, and using the jumpahead()
    method to ensure that the generated sequences seen by each thread don't
    overlap.

    Class Random can also be subclassed if you want to use a different basic
    generator of your own devising: in that case, override the following
    methods:  random(), seed(), getstate(), setstate() and jumpahead().
    Optionally, implement a getrandombits() method so that randrange()
    can cover arbitrarily large ranges.

    """

    VERSION = 2     # used by getstate/setstate

    def __init__(self, x=None):
        """Initialize an instance.

        Optional argument x controls seeding, as for Random.seed().
        """

        self.seed(x)
        self.gauss_next = None

    def seed(self, a=None):
        """Initialize internal state from hashable object.

        None or no argument seeds from current time or from an operating
        system specific randomness source if available.

        If a is not None or an int or long, hash(a) is used instead.
        """

        if a is None:
            try:
                a = long(_hexlify(_urandom(16)), 16)
            except NotImplementedError:
                import time
                a = long(time.time() * 256) # use fractional seconds

        super(Random, self).seed(a)
        self.gauss_next = None

    def getstate(self):
        """Return internal state; can be passed to setstate() later."""
        return self.VERSION, super(Random, self).getstate(), self.gauss_next

    def setstate(self, state):
        """Restore internal state from object returned by getstate()."""
        version = state[0]
        if version == 2:
            version, internalstate, self.gauss_next = state
            super(Random, self).setstate(internalstate)
        else:
            raise ValueError("state with version %s passed to "
                             "Random.setstate() of version %s" %
                             (version, self.VERSION))

## ---- Methods below this point do not need to be overridden when
## ---- subclassing for the purpose of using a different core generator.

## -------------------- pickle support  -------------------

    def __getstate__(self): # for pickle
        return self.getstate()

    def __setstate__(self, state):  # for pickle
        self.setstate(state)

    def __reduce__(self):
        return self.__class__, (), self.getstate()

## -------------------- integer methods  -------------------

    def randrange(self, start, stop=None, step=1, int=int, default=None,
                  maxwidth=1L<<BPF):
        """Choose a random item from range(start, stop[, step]).

        This fixes the problem with randint() which includes the
        endpoint; in Python this is usually not what you want.
        Do not supply the 'int', 'default', and 'maxwidth' arguments.
        """

        # This code is a bit messy to make it fast for the
        # common case while still doing adequate error checking.
        istart = int(start)
        if istart != start:
            raise ValueError, "non-integer arg 1 for randrange()"
        if stop is default:
            if istart > 0:
                if istart >= maxwidth:
                    return self._randbelow(istart)
                return int(self.random() * istart)
            raise ValueError, "empty range for randrange()"

        # stop argument supplied.
        istop = int(stop)
        if istop != stop:
            raise ValueError, "non-integer stop for randrange()"
        width = istop - istart
        if step == 1 and width > 0:
            # Note that
            #     int(istart + self.random()*width)
            # instead would be incorrect.  For example, consider istart
            # = -2 and istop = 0.  Then the guts would be in
            # -2.0 to 0.0 exclusive on both ends (ignoring that random()
            # might return 0.0), and because int() truncates toward 0, the
            # final result would be -1 or 0 (instead of -2 or -1).
            #     istart + int(self.random()*width)
            # would also be incorrect, for a subtler reason:  the RHS
            # can return a long, and then randrange() would also return
            # a long, but we're supposed to return an int (for backward
            # compatibility).

            if width >= maxwidth:
                return int(istart + self._randbelow(width))
            return int(istart + int(self.random()*width))
        if step == 1:
            raise ValueError, "empty range for randrange() (%d,%d, %d)" % (istart, istop, width)

        # Non-unit step argument supplied.
        istep = int(step)
        if istep != step:
            raise ValueError, "non-integer step for randrange()"
        if istep > 0:
            n = (width + istep - 1) // istep
        elif istep < 0:
            n = (width + istep + 1) // istep
        else:
            raise ValueError, "zero step for randrange()"

        if n <= 0:
            raise ValueError, "empty range for randrange()"

        if n >= maxwidth:
            return istart + istep*self._randbelow(n)
        return istart + istep*int(self.random() * n)

    def randint(self, a, b):
        """Return random integer in range [a, b], including both end points.
        """

        return self.randrange(a, b+1)

    def _randbelow(self, n, _log=_log, int=int, _maxwidth=1L<<BPF,
                   _Method=_MethodType, _BuiltinMethod=_BuiltinMethodType):
        """Return a random int in the range [0,n)

        Handles the case where n has more bits than returned
        by a single call to the underlying generator.
        """

        try:
            getrandbits = self.getrandbits
        except AttributeError:
            pass
        else:
            # Only call self.getrandbits if the original random() builtin method
            # has not been overridden or if a new getrandbits() was supplied.
            # This assures that the two methods correspond.
            if type(self.random) is _BuiltinMethod or type(getrandbits) is _Method:
                k = int(1.00001 + _log(n-1, 2.0))   # 2**k > n-1 > 2**(k-2)
                r = getrandbits(k)
                while r >= n:
                    r = getrandbits(k)
                return r
        if n >= _maxwidth:
            _warn("Underlying random() generator does not supply \n"
                "enough bits to choose from a population range this large")
        return int(self.random() * n)

## -------------------- sequence methods  -------------------

    def choice(self, seq):
        """Choose a random element from a non-empty sequence."""
        return seq[int(self.random() * len(seq))]  # raises IndexError if seq is empty

    def shuffle(self, x, random=None, int=int):
        """x, random=random.random -> shuffle list x in place; return None.

        Optional arg random is a 0-argument function returning a random
        float in [0.0, 1.0); by default, the standard random.random.
        """

        if random is None:
            random = self.random
        for i in reversed(xrange(1, len(x))):
            # pick an element in x[:i+1] with which to exchange x[i]
            j = int(random() * (i+1))
            x[i], x[j] = x[j], x[i]

    def sample(self, population, k):
        """Chooses k unique random elements from a population sequence.

        Returns a new list containing elements from the population while
        leaving the original population unchanged.  The resulting list is
        in selection order so that all sub-slices will also be valid random
        samples.  This allows raffle winners (the sample) to be partitioned
        into grand prize and second place winners (the subslices).

        Members of the population need not be hashable or unique.  If the
        population contains repeats, then each occurrence is a possible
        selection in the sample.

        To choose a sample in a range of integers, use xrange as an argument.
        This is especially fast and space efficient for sampling from a
        large population:   sample(xrange(10000000), 60)
        """

        # XXX Although the documentation says `population` is "a sequence",
        # XXX attempts are made to cater to any iterable with a __len__
        # XXX method.  This has had mixed success.  Examples from both
        # XXX sides:  sets work fine, and should become officially supported;
        # XXX dicts are much harder, and have failed in various subtle
        # XXX ways across attempts.  Support for mapping types should probably
        # XXX be dropped (and users should pass mapping.keys() or .values()
        # XXX explicitly).

        # Sampling without replacement entails tracking either potential
        # selections (the pool) in a list or previous selections in a set.

        # When the number of selections is small compared to the
        # population, then tracking selections is efficient, requiring
        # only a small set and an occasional reselection.  For
        # a larger number of selections, the pool tracking method is
        # preferred since the list takes less space than the
        # set and it doesn't suffer from frequent reselections.

        n = len(population)
        if not 0 <= k <= n:
            raise ValueError, "sample larger than population"
        random = self.random
        _int = int
        result = [None] * k
        setsize = 21        # size of a small set minus size of an empty list
        if k > 5:
            setsize += 4 ** _ceil(_log(k * 3, 4)) # table size for big sets
        if n <= setsize or hasattr(population, "keys"):
            # An n-length list is smaller than a k-length set, or this is a
            # mapping type so the other algorithm wouldn't work.
            pool = list(population)
            for i in xrange(k):         # invariant:  non-selected at [0,n-i)
                j = _int(random() * (n-i))
                result[i] = pool[j]
                pool[j] = pool[n-i-1]   # move non-selected item into vacancy
        else:
            try:
                selected = set()
                selected_add = selected.add
                for i in xrange(k):
                    j = _int(random() * n)
                    while j in selected:
                        j = _int(random() * n)
                    selected_add(j)
                    result[i] = population[j]
            except (TypeError, KeyError):   # handle (at least) sets
                if isinstance(population, list):
                    raise
                return self.sample(tuple(population), k)
        return result

## -------------------- real-valued distributions  -------------------

## -------------------- uniform distribution -------------------

    def uniform(self, a, b):
        """Get a random number in the range [a, b)."""
        return a + (b-a) * self.random()

## -------------------- normal distribution --------------------

    def normalvariate(self, mu, sigma):
        """Normal distribution.

        mu is the mean, and sigma is the standard deviation.

        """
        # mu = mean, sigma = standard deviation

        # Uses Kinderman and Monahan method. Reference: Kinderman,
        # A.J. and Monahan, J.F., "Computer generation of random
        # variables using the ratio of uniform deviates", ACM Trans
        # Math Software, 3, (1977), pp257-260.

        random = self.random
        while 1:
            u1 = random()
            u2 = 1.0 - random()
            z = NV_MAGICCONST*(u1-0.5)/u2
            zz = z*z/4.0
            if zz <= -_log(u2):
                break
        return mu + z*sigma

## -------------------- lognormal distribution --------------------

    def lognormvariate(self, mu, sigma):
        """Log normal distribution.

        If you take the natural logarithm of this distribution, you'll get a
        normal distribution with mean mu and standard deviation sigma.
        mu can have any value, and sigma must be greater than zero.

        """
        return _exp(self.normalvariate(mu, sigma))

## -------------------- exponential distribution --------------------

    def expovariate(self, lambd):
        """Exponential distribution.

        lambd is 1.0 divided by the desired mean.  (The parameter would be
        called "lambda", but that is a reserved word in Python.)  Returned
        values range from 0 to positive infinity.

        """
        # lambd: rate lambd = 1/mean
        # ('lambda' is a Python reserved word)

        random = self.random
        u = random()
        while u <= 1e-7:
            u = random()
        return -_log(u)/lambd

## -------------------- von Mises distribution --------------------

    def vonmisesvariate(self, mu, kappa):
        """Circular data distribution.

        mu is the mean angle, expressed in radians between 0 and 2*pi, and
        kappa is the concentration parameter, which must be greater than or
        equal to zero.  If kappa is equal to zero, this distribution reduces
        to a uniform random angle over the range 0 to 2*pi.

        """
        # mu:    mean angle (in radians between 0 and 2*pi)
        # kappa: concentration parameter kappa (>= 0)
        # if kappa = 0 generate uniform random angle

        # Based upon an algorithm published in: Fisher, N.I.,
        # "Statistical Analysis of Circular Data", Cambridge
        # University Press, 1993.

        # Thanks to Magnus Kessler for a correction to the
        # implementation of step 4.

        random = self.random
        if kappa <= 1e-6:
            return TWOPI * random()

        a = 1.0 + _sqrt(1.0 + 4.0 * kappa * kappa)
        b = (a - _sqrt(2.0 * a))/(2.0 * kappa)
        r = (1.0 + b * b)/(2.0 * b)

        while 1:
            u1 = random()

            z = _cos(_pi * u1)
            f = (1.0 + r * z)/(r + z)
            c = kappa * (r - f)

            u2 = random()

            if u2 < c * (2.0 - c) or u2 <= c * _exp(1.0 - c):
                break

        u3 = random()
        if u3 > 0.5:
            theta = (mu % TWOPI) + _acos(f)
        else:
            theta = (mu % TWOPI) - _acos(f)

        return theta

## -------------------- gamma distribution --------------------

    def gammavariate(self, alpha, beta):
        """Gamma distribution.  Not the gamma function!

        Conditions on the parameters are alpha > 0 and beta > 0.

        """

        # alpha > 0, beta > 0, mean is alpha*beta, variance is alpha*beta**2

        # Warning: a few older sources define the gamma distribution in terms
        # of alpha > -1.0
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError, 'gammavariate: alpha and beta must be > 0.0'

        random = self.random
        if alpha > 1.0:

            # Uses R.C.H. Cheng, "The generation of Gamma
            # variables with non-integral shape parameters",
            # Applied Statistics, (1977), 26, No. 1, p71-74

            ainv = _sqrt(2.0 * alpha - 1.0)
            bbb = alpha - LOG4
            ccc = alpha + ainv

            while 1:
                u1 = random()
                if not 1e-7 < u1 < .9999999:
                    continue
                u2 = 1.0 - random()
                v = _log(u1/(1.0-u1))/ainv
                x = alpha*_exp(v)
                z = u1*u1*u2
                r = bbb+ccc*v-x
                if r + SG_MAGICCONST - 4.5*z >= 0.0 or r >= _log(z):
                    return x * beta

        elif alpha == 1.0:
            # expovariate(1)
            u = random()
            while u <= 1e-7:
                u = random()
            return -_log(u) * beta

        else:   # alpha is between 0 and 1 (exclusive)

            # Uses ALGORITHM GS of Statistical Computing - Kennedy & Gentle

            while 1:
                u = random()
                b = (_e + alpha)/_e
                p = b*u
                if p <= 1.0:
                    x = p ** (1.0/alpha)
                else:
                    x = -_log((b-p)/alpha)
                u1 = random()
                if p > 1.0:
                    if u1 <= x ** (alpha - 1.0):
                        break
                elif u1 <= _exp(-x):
                    break
            return x * beta

## -------------------- Gauss (faster alternative) --------------------

    def gauss(self, mu, sigma):
        """Gaussian distribution.

        mu is the mean, and sigma is the standard deviation.  This is
        slightly faster than the normalvariate() function.

        Not thread-safe without a lock around calls.

        """

        # When x and y are two variables from [0, 1), uniformly
        # distributed, then
        #
        #    cos(2*pi*x)*sqrt(-2*log(1-y))
        #    sin(2*pi*x)*sqrt(-2*log(1-y))
        #
        # are two *independent* variables with normal distribution
        # (mu = 0, sigma = 1).
        # (Lambert Meertens)
        # (corrected version; bug discovered by Mike Miller, fixed by LM)

        # Multithreading note: When two threads call this function
        # simultaneously, it is possible that they will receive the
        # same return value.  The window is very small though.  To
        # avoid this, you have to use a lock around all calls.  (I
        # didn't want to slow this down in the serial case by using a
        # lock here.)

        random = self.random
        z = self.gauss_next
        self.gauss_next = None
        if z is None:
            x2pi = random() * TWOPI
            g2rad = _sqrt(-2.0 * _log(1.0 - random()))
            z = _cos(x2pi) * g2rad
            self.gauss_next = _sin(x2pi) * g2rad

        return mu + z*sigma

## -------------------- beta --------------------
## See
## http://sourceforge.net/bugs/?func=detailbug&bug_id=130030&group_id=5470
## for Ivan Frohne's insightful analysis of why the original implementation:
##
##    def betavariate(self, alpha, beta):
##        # Discrete Event Simulation in C, pp 87-88.
##
##        y = self.expovariate(alpha)
##        z = self.expovariate(1.0/beta)
##        return z/(y+z)
##
## was dead wrong, and how it probably got that way.

    def betavariate(self, alpha, beta):
        """Beta distribution.

        Conditions on the parameters are alpha > 0 and beta > 0.
        Returned values range between 0 and 1.

        """

        # This version due to Janne Sinkkonen, and matches all the std
        # texts (e.g., Knuth Vol 2 Ed 3 pg 134 "the beta distribution").
        y = self.gammavariate(alpha, 1.)
        if y == 0:
            return 0.0
        else:
            return y / (y + self.gammavariate(beta, 1.))

## -------------------- Pareto --------------------

    def paretovariate(self, alpha):
        """Pareto distribution.  alpha is the shape parameter."""
        # Jain, pg. 495

        u = 1.0 - self.random()
        return 1.0 / pow(u, 1.0/alpha)

## -------------------- Weibull --------------------

    def weibullvariate(self, alpha, beta):
        """Weibull distribution.

        alpha is the scale parameter and beta is the shape parameter.

        """
        # Jain, pg. 499; bug fix courtesy Bill Arms

        u = 1.0 - self.random()
        return alpha * pow(-_log(u), 1.0/beta)

## -------------------- Wichmann-Hill -------------------

class WichmannHill(Random):

    VERSION = 1     # used by getstate/setstate

    def seed(self, a=None):
        """Initialize internal state from hashable object.

        None or no argument seeds from current time or from an operating
        system specific randomness source if available.

        If a is not None or an int or long, hash(a) is used instead.

        If a is an int or long, a is used directly.  Distinct values between
        0 and 27814431486575L inclusive are guaranteed to yield distinct
        internal states (this guarantee is specific to the default
        Wichmann-Hill generator).
        """

        if a is None:
            try:
                a = long(_hexlify(_urandom(16)), 16)
            except NotImplementedError:
                import time
                a = long(time.time() * 256) # use fractional seconds

        if not isinstance(a, (int, long)):
            a = hash(a)

        a, x = divmod(a, 30268)
        a, y = divmod(a, 30306)
        a, z = divmod(a, 30322)
        self._seed = int(x)+1, int(y)+1, int(z)+1

        self.gauss_next = None

    def random(self):
        """Get the next random number in the range [0.0, 1.0)."""

        # Wichman-Hill random number generator.
        #
        # Wichmann, B. A. & Hill, I. D. (1982)
        # Algorithm AS 183:
        # An efficient and portable pseudo-random number generator
        # Applied Statistics 31 (1982) 188-190
        #
        # see also:
        #        Correction to Algorithm AS 183
        #        Applied Statistics 33 (1984) 123
        #
        #        McLeod, A. I. (1985)
        #        A remark on Algorithm AS 183
        #        Applied Statistics 34 (1985),198-200

        # This part is thread-unsafe:
        # BEGIN CRITICAL SECTION
        x, y, z = self._seed
        x = (171 * x) % 30269
        y = (172 * y) % 30307
        z = (170 * z) % 30323
        self._seed = x, y, z
        # END CRITICAL SECTION

        # Note:  on a platform using IEEE-754 double arithmetic, this can
        # never return 0.0 (asserted by Tim; proof too long for a comment).
        return (x/30269.0 + y/30307.0 + z/30323.0) % 1.0

    def getstate(self):
        """Return internal state; can be passed to setstate() later."""
        return self.VERSION, self._seed, self.gauss_next

    def setstate(self, state):
        """Restore internal state from object returned by getstate()."""
        version = state[0]
        if version == 1:
            version, self._seed, self.gauss_next = state
        else:
            raise ValueError("state with version %s passed to "
                             "Random.setstate() of version %s" %
                             (version, self.VERSION))

    def jumpahead(self, n):
        """Act as if n calls to random() were made, but quickly.

        n is an int, greater than or equal to 0.

        Example use:  If you have 2 threads and know that each will
        consume no more than a million random numbers, create two Random
        objects r1 and r2, then do
            r2.setstate(r1.getstate())
            r2.jumpahead(1000000)
        Then r1 and r2 will use guaranteed-disjoint segments of the full
        period.
        """

        if not n >= 0:
            raise ValueError("n must be >= 0")
        x, y, z = self._seed
        x = int(x * pow(171, n, 30269)) % 30269
        y = int(y * pow(172, n, 30307)) % 30307
        z = int(z * pow(170, n, 30323)) % 30323
        self._seed = x, y, z

    def __whseed(self, x=0, y=0, z=0):
        """Set the Wichmann-Hill seed from (x, y, z).

        These must be integers in the range [0, 256).
        """

        if not type(x) == type(y) == type(z) == int:
            raise TypeError('seeds must be integers')
        if not (0 <= x < 256 and 0 <= y < 256 and 0 <= z < 256):
            raise ValueError('seeds must be in range(0, 256)')
        if 0 == x == y == z:
            # Initialize from current time
            import time
            t = long(time.time() * 256)
            t = int((t&0xffffff) ^ (t>>24))
            t, x = divmod(t, 256)
            t, y = divmod(t, 256)
            t, z = divmod(t, 256)
        # Zero is a poor seed, so substitute 1
        self._seed = (x or 1, y or 1, z or 1)

        self.gauss_next = None

    def whseed(self, a=None):
        """Seed from hashable object's hash code.

        None or no argument seeds from current time.  It is not guaranteed
        that objects with distinct hash codes lead to distinct internal
        states.

        This is obsolete, provided for compatibility with the seed routine
        used prior to Python 2.1.  Use the .seed() method instead.
        """

        if a is None:
            self.__whseed()
            return
        a = hash(a)
        a, x = divmod(a, 256)
        a, y = divmod(a, 256)
        a, z = divmod(a, 256)
        x = (x + a) % 256 or 1
        y = (y + a) % 256 or 1
        z = (z + a) % 256 or 1
        self.__whseed(x, y, z)

## --------------- Operating System Random Source  ------------------

class SystemRandom(Random):
    """Alternate random number generator using sources provided
    by the operating system (such as /dev/urandom on Unix or
    CryptGenRandom on Windows).

     Not available on all systems (see os.urandom() for details).
    """

    def random(self):
        """Get the next random number in the range [0.0, 1.0)."""
        return (long(_hexlify(_urandom(7)), 16) >> 3) * RECIP_BPF

    def getrandbits(self, k):
        """getrandbits(k) -> x.  Generates a long int with k random bits."""
        if k <= 0:
            raise ValueError('number of bits must be greater than zero')
        if k != int(k):
            raise TypeError('number of bits should be an integer')
        bytes = (k + 7) // 8                    # bits / 8 and rounded up
        x = long(_hexlify(_urandom(bytes)), 16)
        return x >> (bytes * 8 - k)             # trim excess bits

    def _stub(self, *args, **kwds):
        "Stub method.  Not used for a system random number generator."
        return None
    seed = jumpahead = _stub

    def _notimplemented(self, *args, **kwds):
        "Method should not be called for a system random number generator."
        raise NotImplementedError('System entropy source does not have state.')
    getstate = setstate = _notimplemented

## -------------------- test program --------------------

def _test_generator(n, func, args):
    import time
    print n, 'times', func.__name__
    total = 0.0
    sqsum = 0.0
    smallest = 1e10
    largest = -1e10
    t0 = time.time()
    for i in range(n):
        x = func(*args)
        total += x
        sqsum = sqsum + x*x
        smallest = min(x, smallest)
        largest = max(x, largest)
    t1 = time.time()
    print round(t1-t0, 3), 'sec,',
    avg = total/n
    stddev = _sqrt(sqsum/n - avg*avg)
    print 'avg %g, stddev %g, min %g, max %g' % \
              (avg, stddev, smallest, largest)


def _test(N=2000):
    _test_generator(N, random, ())
    _test_generator(N, normalvariate, (0.0, 1.0))
    _test_generator(N, lognormvariate, (0.0, 1.0))
    _test_generator(N, vonmisesvariate, (0.0, 1.0))
    _test_generator(N, gammavariate, (0.01, 1.0))
    _test_generator(N, gammavariate, (0.1, 1.0))
    _test_generator(N, gammavariate, (0.1, 2.0))
    _test_generator(N, gammavariate, (0.5, 1.0))
    _test_generator(N, gammavariate, (0.9, 1.0))
    _test_generator(N, gammavariate, (1.0, 1.0))
    _test_generator(N, gammavariate, (2.0, 1.0))
    _test_generator(N, gammavariate, (20.0, 1.0))
    _test_generator(N, gammavariate, (200.0, 1.0))
    _test_generator(N, gauss, (0.0, 1.0))
    _test_generator(N, betavariate, (3.0, 3.0))

# Create one instance, seeded from current time, and export its methods
# as module-level functions.  The functions share state across all uses
#(both in the user's code and in the Python libraries), but that's fine
# for most programs and is easier for the casual user than making them
# instantiate their own Random() instance.

_inst = Random()
seed = _inst.seed
random = _inst.random
uniform = _inst.uniform
randint = _inst.randint
choice = _inst.choice
randrange = _inst.randrange
sample = _inst.sample
shuffle = _inst.shuffle
normalvariate = _inst.normalvariate
lognormvariate = _inst.lognormvariate
expovariate = _inst.expovariate
vonmisesvariate = _inst.vonmisesvariate
gammavariate = _inst.gammavariate
gauss = _inst.gauss
betavariate = _inst.betavariate
paretovariate = _inst.paretovariate
weibullvariate = _inst.weibullvariate
getstate = _inst.getstate
setstate = _inst.setstate
jumpahead = _inst.jumpahead
getrandbits = _inst.getrandbits

if __name__ == '__main__':
    _test()
