import numpy

def findsmallest(primes, f):
    for i in range(f, len(primes)):
        if primes[i] == 1:
            return i

def sieve(n):
    primes = numpy.ones((n+1,), dtype=numpy.bool)
    primes[0] = 0
    primes[1] = 0
    k = 2
    while(k**2 <= n):
        mult = k**2
        while mult <= n:
            primes[mult] = 0
            mult = mult + k
        k = findsmallest(primes, k+1)
    return sum(primes), primes
    
if __name__ == '__main__':
    numprime, l = sieve(100)
    print numprime
    print [i for i, v in enumerate(l) if v]