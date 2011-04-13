#!python

import pyopencl as cl
import numpy
import numpy.linalg as la

n = 1000
block_size = 256

KERNEL_CODE = ''.join( open('sieve.cl').readlines() )

primes = numpy.zeros(n).astype(numpy.int32)

sieve = numpy.zeros(n).astype(numpy.int32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
primes_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=primes)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=primes.nbytes)

kernel_params = {"block_size": block_size, "max_n": n}

prg = cl.Program(ctx, KERNEL_CODE % kernel_params).build()
kernel = prg.findsmallest

event = kernel(queue, primes.shape, primes_buf)

event.wait()

cl.enqueue_read_buffer(queue, primes_buf, sieve).wait()

for i in range(0,len(sieve)):
    if sieve[i] == 0:
        print i
