#!python

import pyopencl as cl
import numpy as np

block_size = 512

n = 10

KERNEL_CODE = ''.join( open('lab3-kernel.cl').readlines() )

a = np.random.rand(n).astype(np.int32)
c = np.empty(n).astype(np.int32)
print a
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=c.nbytes)

kernel_params = {"block_size": block_size, "max_length":32, "n":n}

prg = cl.Program(ctx, KERNEL_CODE % kernel_params).build()
event = prg.radixSort(queue, a.shape, None, dest_buf, a_buf)

cl.enqueue_read_buffer(queue, dest_buf, c).wait()

print c
print "\n"
print len(c)
# print la.norm(a_plus_b - (a+b))
# print la.norm(a_plus_b)
