#!python

import pyopencl as cl
import numpy
import numpy.linalg as la

block_size = 16

def load_matrix(fn, w, h):
    flatarray = [float(i) for i in open(fn).readlines()[0].split(' ')]
    return numpy.array([[flatarray[j*w+i] for i in range(w)] for j in range(h)], dtype=numpy.float32)

KERNEL_CODE = ''.join( open('matrixmul.cl').readlines() )

a_width = block_size
a_height= block_size
b_width = block_size
b_height= a_width

c_width = b_width
c_height = a_height

a = numpy.random.rand(a_height, a_width).astype(numpy.float32)
a = load_matrix('matrix1.txt',block_size, block_size)
b = numpy.random.rand(b_height, a_width).astype(numpy.float32)
b = load_matrix('matrix2.txt',block_size, block_size)
c = numpy.empty((c_height, c_width)).astype(numpy.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=c.nbytes)

kernel_params = {"block_size": block_size, "w_a":a_width, "h_a":a_height, "w_b":b_width}

prg = cl.Program(ctx, KERNEL_CODE % kernel_params).build()
kernel = prg.matrixMul


event = kernel(queue, c.shape, (block_size, block_size), dest_buf, a_buf, b_buf)


event.wait()

cl.enqueue_read_buffer(queue, dest_buf, c).wait()

for i in c:
    print i
print ""
print len(c)
print numpy.dot(a, b)
# print la.norm(a_plus_b - (a+b))
# print la.norm(a_plus_b)
