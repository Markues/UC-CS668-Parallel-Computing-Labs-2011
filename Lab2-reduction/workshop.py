#!python

import pyopencl as cl
import numpy
import numpy.linalg as la

#n = 2**20 * 7 / 10
n = 655360
n = 100000
print n
block_size = None

KERNEL_CODE = open('sieve.cl').read()

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

for i, s in enumerate(sieve):
    if s == 0:
        print i
