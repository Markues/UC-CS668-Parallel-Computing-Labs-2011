#!/usr/bin/env python
import pyopencl as cl
import numpy

# defineable variables
n = 100000
hash_size = n / 10
KERNEL_CODE = open('sieve.cl').read()
#block_size = 8

# pick a graphics card
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# create structures
primes = numpy.zeros(n).astype(numpy.int32)
sieve = numpy.zeros(n).astype(numpy.int32)

# setup buffers
mf = cl.mem_flags
primes_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=primes)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=primes.nbytes)

# load CL program
kernel_params = {"hash_size": hash_size, "max_n": n}
prg = cl.Program(ctx, KERNEL_CODE).build()

# run CL program
kernel = prg.findsmallest
event = kernel(queue, primes.shape, primes_buf)
event.wait()

# get results from card
cl.enqueue_read_buffer(queue, primes_buf, sieve).wait()

print [i for i in sieve]
