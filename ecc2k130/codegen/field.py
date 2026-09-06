# Field models for the ECC2K-130 code generator.
#
# Two independent representations of GF(2^m) are built here and cross-checked
# against each other:
#
#   Onb  -- the permuted type-II optimal normal basis.  An element is a
#           symmetric vector over Z/n, n = 2m+1, taken modulo the all-ones
#           vector; multiplication is cyclic convolution.  This is exactly the
#           subalgebra of F_2[z]/(z^n - 1) fixed by z -> 1/z, whose quotient by
#           (1 + z + ... + z^(n-1)) is GF(2^m).  Basis element i is
#           gamma_i = zeta^i + zeta^-i, so squaring is the index permutation
#           i -> fold(2i) and the Hamming weight is squaring-invariant.
#
#   Pb   -- the ordinary polynomial basis F_2[z]/(F), used only for the
#           challenge parameters (Certicom states them in polynomial basis)
#           and as an independent oracle.
#
# No type hints, camelCase identifiers, no itertools (project convention).

def isPrime(n):
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def multiplicativeOrder(a, n):
    o = 1
    v = a % n
    while v != 1:
        v = v * a % n
        o += 1
        if o > n:
            return 0
    return o


def popcount(x):
    return bin(x).count('1')


class Onb:
    """GF(2^m) as symmetric vectors mod the all-ones vector, n = 2m+1."""

    def __init__(self, m):
        self.m = m
        self.n = 2 * m + 1
        if not isPrime(self.n):
            raise ValueError("2m+1 = %d is not prime; no type-II ONB" % self.n)
        self.ord2 = multiplicativeOrder(2, self.n)
        if self.ord2 != m and self.ord2 != 2 * m:
            raise ValueError("ord_%d(2) = %d, not m or 2m; no type-II ONB" % (self.n, self.ord2))
        self.allOnes = (1 << self.n) - 1

    # ---- conversion between coordinate vectors and internal symmetric form
    def fold(self, i):
        i %= self.n
        return i if i <= self.m else self.n - i

    def fromCoords(self, a):
        """a is an m-bit int, bit (i-1) = coefficient of gamma_i."""
        u = 0
        for i in range(1, self.m + 1):
            if (a >> (i - 1)) & 1:
                u |= (1 << i) | (1 << (self.n - i))
        return u

    def toCoords(self, u):
        u = self.normalize(u)
        a = 0
        for i in range(1, self.m + 1):
            if (u >> i) & 1:
                a |= 1 << (i - 1)
        return a

    def normalize(self, u):
        if u & 1:
            u ^= self.allOnes
        return u

    def isSymmetric(self, u):
        for i in range(1, self.n):
            if ((u >> i) & 1) != ((u >> (self.n - i)) & 1):
                return False
        return True

    # ---- arithmetic on internal symmetric form
    def rot(self, u, k):
        k %= self.n
        return ((u << k) | (u >> (self.n - k))) & self.allOnes if k else u

    def mul(self, a, b):
        r = 0
        bb = b
        i = 0
        while bb:
            if bb & 1:
                r ^= self.rot(a, i)
            bb >>= 1
            i += 1
        return self.normalize(r)

    def sqr(self, a):
        return self.normalize(self.rot(a, 0) if False else self.frob(a, 1))

    def frob(self, a, k):
        """a -> a^(2^k), i.e. z -> z^(2^k), an index permutation."""
        e = pow(2, k, self.n)
        r = 0
        for i in range(self.n):
            if (a >> i) & 1:
                r |= 1 << (i * e % self.n)
        return self.normalize(r)

    def add(self, a, b):
        return self.normalize(a ^ b)

    def one(self):
        # 1 = sum of all gamma_i (since 1 + sum_{i!=0} z^i = 0)
        return self.fromCoords((1 << self.m) - 1)

    def zero(self):
        return 0

    def pow(self, a, e):
        r = self.one()
        base = a
        while e:
            if e & 1:
                r = self.mul(r, base)
            base = self.mul(base, base)
            e >>= 1
        return r

    def inv(self, a):
        return self.pow(a, (1 << self.m) - 2)

    def gamma(self, i):
        return self.fromCoords(1 << (i - 1))

    def hammingWeight(self, u):
        return popcount(self.toCoords(u))

    def trace(self, u):
        """Tr(a) over GF(2); equals the parity of the normal-basis weight."""
        t = 0
        v = u
        for _ in range(self.m):
            t ^= v
            v = self.frob(v, 1)
        return 1 if self.toCoords(t) else 0

    def selfTest(self, rng):
        one = self.one()
        assert self.mul(one, one) == one
        for _ in range(40):
            a = self.randomElement(rng)
            b = self.randomElement(rng)
            c = self.randomElement(rng)
            assert self.isSymmetric(a) and (a & 1) == 0
            assert self.mul(a, one) == a
            assert self.mul(a, b) == self.mul(b, a)
            assert self.mul(self.mul(a, b), c) == self.mul(a, self.mul(b, c))
            assert self.mul(a, self.add(b, c)) == self.add(self.mul(a, b), self.mul(a, c))
            assert self.frob(a, 1) == self.mul(a, a)
            assert self.frob(a, self.m) == a
            assert self.hammingWeight(a) == self.hammingWeight(self.frob(a, 1))
            assert (self.hammingWeight(a) & 1) == self.trace(a)
            if a:
                assert self.mul(a, self.inv(a)) == one
        return True

    def randomElement(self, rng):
        return self.fromCoords(rng.getrandbits(self.m))

    # ---- the optimal polynomial basis {1, c, c^2, ...}, c = gamma_1
    def cPowers(self, count):
        c = self.gamma(1)
        out = [self.one()]
        for _ in range(count - 1):
            out.append(self.mul(out[-1], c))
        return out

    def minPolyOfC(self):
        """Minimal polynomial of c = gamma_1 over GF(2), as a bit-int."""
        pw = self.cPowers(self.m + 1)
        rows = []
        for k in range(self.m + 1):
            rows.append((self.toCoords(pw[k]), 1 << k))
        # Gaussian elimination to find the first linear dependency.
        basis = []
        for vec, tag in rows:
            v, t = vec, tag
            for bv, bt in basis:
                if v ^ bv < v:
                    v ^= bv
                    t ^= bt
            if v == 0:
                return t
            basis.append((v, t))
            basis.sort(key=lambda p: -p[0])
        raise RuntimeError("no dependency found")


class Pb:
    """GF(2^m) = F_2[z]/(F), F given as a bit-int with bit m set."""

    def __init__(self, m, poly):
        self.m = m
        self.poly = poly
        self.mask = (1 << m) - 1

    def mul(self, a, b):
        r = 0
        while b:
            if b & 1:
                r ^= a
            b >>= 1
            a <<= 1
            if (a >> self.m) & 1:
                a ^= self.poly
        return r

    def sqr(self, a):
        return self.mul(a, a)

    def pow(self, a, e):
        r = 1
        while e:
            if e & 1:
                r = self.mul(r, a)
            a = self.mul(a, a)
            e >>= 1
        return r

    def inv(self, a):
        return self.pow(a, (1 << self.m) - 2)

    def isIrreducible(self):
        # x^(2^k) mod poly, gcd test
        xp = 2
        for _ in range(1, self.m // 2 + 1):
            xp = self.sqr(xp)
            if polyGcd(self.poly, xp ^ 2) != 1:
                return False
        xp = self.sqr(xp) if self.m % 2 == 0 else xp
        return True


def polyDegree(a):
    return a.bit_length() - 1


def polyMod(a, b):
    db = polyDegree(b)
    while a and polyDegree(a) >= db:
        a ^= b << (polyDegree(a) - db)
    return a


def polyGcd(a, b):
    while b:
        a, b = b, polyMod(a, b)
    return a


def frobeniusPowers(f, m):
    """X^(2^i) mod f for i = 0..m-1, f in F_2[X] of degree m."""
    out = []
    cur = 2  # X
    for _ in range(m):
        out.append(cur)
        # square then reduce
        s = 0
        c = cur
        i = 0
        while c:
            if c & 1:
                s |= 1 << (2 * i)
            c >>= 1
            i += 1
        cur = polyMod(s, f)
    return out


def findRootInOnb(onb, f, rng):
    """Find a root of f (in F_2[X], degree m, split over GF(2^m)) inside the
    ONB model.  Cantor-Zassenhaus with the trace map, char 2."""
    m = onb.m
    frob = frobeniusPowers(f, m)          # X^(2^i) mod f, F_2 coefficients
    # Represent polynomials over the ONB field as coefficient lists (low first).
    fk = []
    for j in range(m + 1):
        fk.append(onb.one() if (f >> j) & 1 else 0)

    def polyTrim(p):
        while p and p[-1] == 0:
            p.pop()
        return p

    def polyMulMod(p, q, mod):
        r = [0] * (len(p) + len(q) - 1)
        for i in range(len(p)):
            if p[i] == 0:
                continue
            for j in range(len(q)):
                if q[j]:
                    r[i + j] ^= onb.mul(p[i], q[j])
        return polyRem(polyTrim(r), mod)

    def polyRem(p, mod):
        dm = len(mod) - 1
        p = list(p)
        while len(p) - 1 >= dm and len(p) > 0:
            d = len(p) - 1
            c = p[d]
            if c:
                # mod is monic
                for j in range(dm + 1):
                    if mod[j]:
                        p[d - dm + j] ^= onb.mul(c, mod[j])
            p.pop()
            polyTrim(p)
            if not p:
                break
        return polyTrim(p)

    def polyGcdOnb(p, q):
        p, q = polyTrim(list(p)), polyTrim(list(q))
        while q:
            r = polyRem(p, monic(q))
            p, q = q, r
        return monic(p)

    def monic(p):
        p = polyTrim(list(p))
        if not p:
            return p
        lead = p[-1]
        if lead != onb.one():
            li = onb.inv(lead)
            for i in range(len(p)):
                p[i] = onb.mul(p[i], li)
        return p

    def traceOfAlphaX(alpha, mod):
        """sum_i (alpha X)^(2^i) mod f, reduced mod `mod`."""
        acc = [0] * m
        ai = alpha
        for i in range(m):
            fi = frob[i]
            j = 0
            v = fi
            while v:
                if v & 1:
                    acc[j] ^= ai
                v >>= 1
                j += 1
            ai = onb.frob(ai, 1)
        return polyRem(polyTrim(acc), mod)

    cur = monic(fk)
    guard = 0
    while len(cur) - 1 > 1:
        guard += 1
        if guard > 400:
            raise RuntimeError("root finding failed to converge")
        alpha = onb.randomElement(rng)
        if alpha == 0:
            continue
        t = traceOfAlphaX(alpha, cur)
        if not t:
            continue
        g = polyGcdOnb(cur, t)
        dg = len(g) - 1
        if 0 < dg < len(cur) - 1:
            cur = g if dg <= (len(cur) - 1) // 2 else polyDivExact(cur, g, onb)
            cur = monic(cur)
    if len(cur) - 1 != 1:
        raise RuntimeError("did not isolate a linear factor")
    # cur = X + r  ->  root = r
    return cur[0]


def polyDivExact(p, d, onb):
    p = list(p)
    dd = len(d) - 1
    out = [0] * (len(p) - dd)
    while len(p) - 1 >= dd and p:
        c = p[-1]
        k = len(p) - 1 - dd
        out[k] = c
        for j in range(dd + 1):
            if d[j]:
                p[k + j] ^= onb.mul(c, d[j])
        while p and p[-1] == 0:
            p.pop()
    return out
