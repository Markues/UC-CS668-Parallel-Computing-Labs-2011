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
        self.block_size = 512
        self.primes = []
        self.offset = 0
        self.mf = cl.mem_flags

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        fstr = open(filename).read()
        kernel_params = {"max_n": self.block_size}

        #create the program
        self.program = cl.Program(self.ctx, fstr % kernel_params).build()


    def popCorn(self):

        #initialize client side (CPU) arrays
        self.a = numpy.ones((self.block_size, 1), dtype=numpy.uint32)

        #create OpenCL buffers
        self.a_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.a)

    def execute(self):

        def perform_sieve(bitarray, offset=0):
            #initialize client side (CPU) arrays
            self.a = bitarray

            #create OpenCL buffers
            self.a_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.a)
            
            event1 = self.program.sieve(self.queue, (self.block_size,), None, self.a_buf)
            cl.enqueue_read_buffer(self.queue, self.a_buf, self.a).wait()
            
            return self.a
        
        def filter_primes(primes_array, bit_array, offset):

            if( not len(primes_array) ):
                return empty_bitarray()

            a = empty_bitarray()
            b = numpy.array(self.primes, dtype=numpy.uint32)
            c = numpy.array(offset, dtype=numpy.uint32)

            a_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=a)
            b_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=b)
            c_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=c)
            
            # send integers and new bit mask to pfilter
            event2 = self.program.pfilter(self.queue, (len(self.primes),), (self.block_size,), (1,), None, a_buf, b_buf, c_buf)
            cl.enqueue_read_buffer(self.queue, a_buf, a)
            
            return a
        
        def empty_bitarray():
            return numpy.ones((self.block_size,1), dtype=numpy.uint32)
        
        def bitarray_to_primes_array(bitarray, offset):
            for i,x in enumerate(self.a):
                i += offset
                if x:
                    self.primes.append(i)
            return self.primes
        
        def print_primes_array():
            for prime in self.primes:
                print prime
        
        offset = 0
        while(self.offset < self.n):
            bitarray = filter_primes(self.primes, bitarray, offset)
            bitarray = perform_sieve(bitarray, 0)
            self.primes += bitarray_to_primes_array(bitarray, offset)
            self.offset += self.block_size
        
        
        print 'Sieve Duration:', 1e-9 * (event1.profile.end - event1.profile.start)
        print 'Filter Duration:', 1e-9 * (event2.profile.end - event2.profile.start)

if __name__ == "__main__":
    example = CL(2**11)
    example.loadProgram("part1.cl")
    example.popCorn()
    example.execute()
