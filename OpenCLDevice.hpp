#ifndef OTOO_OPENCLDEVICE_H
#define OTOO_OPENCLDEVICE_H

#ifdef PEZYCL
#include "pzclutil.h"
#define CL_MEM_READ_ONLY CL_MEM_READ_WRITE
#define CL_MEM_WRITE_ONLY CL_MEM_READ_WRITE
#else
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

namespace OTOO {
  class OpenCLDevice {
  public:
    OpenCLDevice(unsigned int);
    ~OpenCLDevice();
    void SetupContext(unsigned int);
    void SetKernelOptions(std::string);
    void BuildOpenCLKernels(const char *, const char *);
    cl_kernel GetKernel(const char *);

    cl_context ctx;
    cl_command_queue q;
  protected:
    uint64 n_cpus;
    uint64 n_gpus;
    cl_platform_id platform_id[16];
    cl_platform_id pl;
    cl_uint npl;
    cl_device_id device_id[16];
    cl_device_id dev;
    cl_uint ndev;
    cl_program program;
    std::string kernel_options;
    char pname[128];
    char dname[128];
    char pver[128];

    void dumperror();
  };

  OpenCLDevice::OpenCLDevice(unsigned int ip)
  {
    cl_int result = CL_SUCCESS;
    if ((result = clGetPlatformIDs(16, platform_id, &npl)) != CL_SUCCESS) {
      fprintf(stderr, "clGetPlatformIDs() failed : %dn", result);
      exit(-1);
    }

    for(unsigned int i = 0; i < npl; i++) {
      clGetPlatformInfo(platform_id[i],  CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
      clGetPlatformInfo(platform_id[i],  CL_PLATFORM_VERSION, sizeof(pver), pver, NULL);
      fprintf(stderr, "platform %d %s %s\n", i, pname, pver);

      // Get Device ID
      cl_uint ndev = 0;
      if ((result = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_ALL, 16, device_id, &ndev)) != CL_SUCCESS) {
	fprintf(stderr, "clGetDeviceIDs() failed : %d\n", result);
	exit(-1);
      }

      for(unsigned int j = 0; j < ndev; j++) {
	clGetDeviceInfo(device_id[j], CL_DEVICE_NAME, sizeof(dname), dname, NULL);
	fprintf(stderr, "\tdevice %d %s\n", j, dname);
	//	list.push_back(std::make_pair(i, j));
      }
    }
    std::cerr << "Selected: "; 
    if (npl <= ip) fprintf(stderr, "FATAL: the specifed platform does not exist");

    if ((result = clGetDeviceIDs(platform_id[ip], CL_DEVICE_TYPE_ALL, 16, device_id, &ndev)) != CL_SUCCESS) {
      fprintf(stderr, "clGetDeviceIDs() failed : %d\n", result);
      exit(-1);
    }
    pl = platform_id[ip];
  }

  void OpenCLDevice::SetKernelOptions(std::string options)
  {
    kernel_options = options;
  }

  void OpenCLDevice::SetupContext(unsigned int id)
  {
    if (ndev <= id) fprintf(stderr, "FATAL: the specifed device does not exist");

    clGetPlatformInfo(pl, CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
    clGetPlatformInfo(pl, CL_PLATFORM_VERSION, sizeof(pver), pver, NULL);
    clGetDeviceInfo(device_id[id], CL_DEVICE_NAME, sizeof(dname), dname, NULL);

    std::cerr << pname << " " << pver << "::"	<< dname << "\n";

    cl_int status = CL_SUCCESS;
    // Create Context
    if( (ctx = clCreateContext(NULL, 1, &device_id[id], NULL, NULL, &status)) == NULL ) {
      fprintf(stderr, "Create context failed %d\n", status);
      exit(-1);
    }

    // use the first device only
    q = clCreateCommandQueue(ctx, device_id[id], CL_QUEUE_PROFILING_ENABLE, &status);
    if (status != CL_SUCCESS) {  
      fprintf(stderr, "Create commandq failed %d\n", status);
      exit(-1);
    }
    dev = device_id[id];
  }

  void OpenCLDevice::BuildOpenCLKernels(const char *kernelfile, const char *more_options = NULL) 
  {
    cl_int status = CL_SUCCESS;
#ifdef PEZYCL
    {
      // Create program object
      std::vector<cl_device_id> device_id_lists;
      device_id_lists.push_back( dev );
      program = PZCLUtil::CreateProgram(ctx, device_id_lists, "./kernel.sc32/kernel.pz" );
      if(program == NULL)
	{
	  fprintf(stderr, "clCreateProgramWithBinary failed\n");
	  exit(-1);
	}
    }
#else
    {
      char *prog;
      size_t ss[1];
      ss[0] = strlen(kernelfile);
      prog = (char *)malloc(ss[0]);
      strcpy(prog, kernelfile);
      program = clCreateProgramWithSource(ctx, 1, (const char **)&prog, ss, &status);
      if (status != CL_SUCCESS) {  
	fprintf(stderr, "cl create program failed %d\n", status);
	exit(-1);
      }
      free(prog);
    }
#endif

    std::stringstream options;
    if (more_options != NULL)
      options << more_options;
    options << kernel_options;

    std::cerr << "Build options :: " << options.str() << "\n";
    //prog.build(v_dev, options.str().c_str());

    status = clBuildProgram(program, 1, &dev, options.str().c_str(), NULL, NULL);
    if(status != CL_SUCCESS) {
      fprintf(stderr, "build failed\n");
      dumperror();
      exit(-1);
    }
  }

  void OpenCLDevice::dumperror()
  {
#ifdef PEZYCL
#else
    cl_int logStatus;
    char * buildLog = NULL;
    size_t buildLogSize = 0;

    logStatus = clGetProgramBuildInfo (program,
                                       dev,
                                       CL_PROGRAM_BUILD_LOG,
                                       buildLogSize,
                                       buildLog,
                                       &buildLogSize);

    buildLog = (char*)malloc(buildLogSize);
    memset(buildLog, 0, buildLogSize);

    logStatus = clGetProgramBuildInfo (program,
                                       dev,
                                       CL_PROGRAM_BUILD_LOG,
                                       buildLogSize,
                                       buildLog,
                                       NULL);

    fprintf(stderr, "%s\n", buildLog);
    std::cout << logStatus << "\n";

    free(buildLog);
#endif
  }

  cl_kernel OpenCLDevice::GetKernel(const char *kernel_main)
  {
    cl_int status = CL_SUCCESS;
    cl_kernel res = clCreateKernel(program, kernel_main, &status);

    if (status != CL_SUCCESS) {  
      fprintf(stderr, "Create kernel %s failed %d\n", kernel_main, status);
      exit(-1);
    }

    return res;
  }
}
#endif
