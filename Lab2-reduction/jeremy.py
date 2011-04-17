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

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        fstr = open(filename).read()
        kernel_params = {"max_n": self.n}

        #create the program
        self.program = cl.Program(self.ctx, fstr % kernel_params).build()

    def popCorn(self):
        mf = cl.mem_flags

        #initialize client side (CPU) arrays
        self.a = numpy.array(range(self.n), dtype=numpy.int32)

        #create OpenCL buffers
        self.a_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.a)

    def execute(self):
        event = self.program.part1(self.queue, self.a.shape, None, self.a_buf)
        cl.enqueue_read_buffer(self.queue, self.a_buf, self.a).wait()
        print 1e-9 * (event.profile.end - event.profile.start)
        print "a", self.a



if __name__ == "__main__":
    example = CL(2**22)
    example.loadProgram("part1.cl")
    example.popCorn()
    example.execute()
