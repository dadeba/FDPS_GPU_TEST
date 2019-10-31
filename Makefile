PS_PATH = ../FDPS_dev/src/
INC = -I$(PS_PATH) -I/usr/local/cuda/include

CXX = g++
CXX = mpicxx
CXXFLAGS = -std=c++11 -O3 -ffast-math -funroll-loops
CXXFLAGS += -DPARTICLE_SIMULATOR_THREAD_PARALLEL -fopenmp
CXXFLAGS += -DPARTICLE_SIMULATOR_MPI_PARALLEL

LIBS = -lnetcdf -lnetcdf_c++ -lOpenCL -L/usr/local/cuda/lib64

CPPOBJS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
CPPHDRS = $(wildcard *.hpp)
PROGRAM = run

.PHONY:	clean all

all: $(CPPOBJS) $(CPPHDRS) 
	$(CXX) $(CXXFLAGS) $(CPPOBJS) -o $(PROGRAM) $(LIBS) $(INC)

%.o: %.cpp $(CPPHDRS) kernel.file
	$(CXX) -c $< $(CXXFLAGS) $(INC)

kernel.file: kernel.cl
	./template-converter kernel_str $< >| kernel.file

#clean:
#	rm -f $(CPPOBJS)

clean:
	$(PG_CLEAN)
	rm -f $(CPPOBJS)	
	rm -f $(PROGRAM)
	rm -f kernel.file
	rm -rf result
