#Port from Adventures in OpenCL Part1 to PyOpenCL
# http://enja.org/2010/07/13/adventures-in-opencl-part-1-getting-started/
# http://documen.tician.de/pyopencl/

import pyopencl as cl
import numpy

class CL:
    def __init__(self, n):
        self.ctx = cl.create_some_context()
        cf = cl.command_queue_properties
        self.queue = cl.CommandQueue(self.ctx, properties=cf.PROFILING_ENABLE)
        self.n = n
        self.block_size = 1024
        self.primes = []
        self.offset = 0

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        fstr = open(filename).read()
        kernel_params = {"max_n": self.block_size}

        #create the program
        self.program = cl.Program(self.ctx, fstr % kernel_params).build()

    def popCorn(self):
        mf = cl.mem_flags

        #initialize client side (CPU) arrays
        self.a = numpy.ones((self.block_size, 1), dtype=numpy.uint32)

        #create OpenCL buffers
        self.a_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.a)

    def execute(self):
        event1 = self.program.sieve(self.queue, (self.block_size,), None, self.a_buf)
        cl.enqueue_read_buffer(self.queue, self.a_buf, self.a).wait()

        self.offset += self.block_size

        # store bit mask of primes as integers
        for i,x in enumerate(self.a):
            if x:
                self.primes.append(i)

        self.a = numpy.ones((self.block_size, 1), dtype=numpy.uint32)
        self.b = numpy.array(self.primes, dtype=numpy.uint32)
        self.c = numpy.array(self.offset, dtype=numpy.uint32)

        for x in self.primes:
                self.a[x] = 0

        mf = cl.mem_flags
        self.a_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.a)
        self.b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.b)
        self.c_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.c)

        # send integers and new bit mask to pfilter
        event2 = self.program.pfilter(self.queue, (len(self.primes),), (self.block_size,), (1,), None, self.a_buf, self.b_buf, self.c_buf)
        
        print 'Sieve Duration:', 1e-9 * (event1.profile.end - event1.profile.start)
        print 'Filter Duration:', 1e-9 * (event2.profile.end - event2.profile.start)
        #print [i for i in self.a if not i]
        #print self.a[:10]
        #print self.a
        for i,x in enumerate(self.a):
            if x:
                print i
        #for i,x in enumerate(reversed(self.a)):
        #    if not x:
        #        print self.n - i - 1
        #        break

if __name__ == "__main__":
    example = CL(2**11)
    example.loadProgram("part1.cl")
    example.popCorn()
    example.execute()
