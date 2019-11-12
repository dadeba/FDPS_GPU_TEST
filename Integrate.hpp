#ifndef OTOO_INTEGRATE_H
#define OTOO_INTEGRATE_H

typedef PS::S64 uint64;
#include "Config.hpp"
#include "IDEALEOS.hpp"
#include "OpenCLDevice.hpp"

OTOO::EOS *eos;

namespace OTOO {
  uint64 nall;
  double t, tend, dt, dt_sys;
  double dt_dump;
  uint64 nstep; 
  double ke, pe, te;
  double ke0, pe0, te0;
  
  double start_time_loop;
  double elapsed_time_loop;

  double kernel_time_grav;
  double kernel_time_sph1;
  double kernel_time_sph2;
  uint64 kernel_count;
  
  std::ofstream log_energy;

  ConfigFile CF;
  
  // parameters
  double grav;
  double lunit, munit, tunit;
  double vunit, eunit, dunit;
  double punit;
  double divunit;
  double aunit, deunit;
  double tunit_inv;
  double lunit_inv;
  double munit_inv;
  double vunit_inv;
  double dunit_inv;
  double punit_inv;
  double eunit_inv;
  double eps;

  double nn_mean;
  double cfl_number;
  double alpha_max;
  double alpha_min;
  
  /*
    Physical constant in cgs unit, from Galactic Dynamics 643p 

    bk : bolzman's constant
    pc : parsec in cm
    sm : solar mass in g
    sr : solar radius in cm
    yr : one year in sec
    gc : gravitational constant
    pm : proton mass
    em : electron mass
    pi : 3.14159265
  */
  /*
  static const double bk;
  static const double pc;
  static const double sm;
  static const double sr;
  static const double yr;
  static const double gc;
  static const double pm;
  static const double em;
  static const double pi;
  static const double Re;
  static const double Me;
  */

  const double bk = 1.38066e-16;
  const double pc = 3.0857e18;
  const double sm = 1.9891e33;
  const double sr = 6.9599e10;
  const double yr = 3.1536e7;
  const double gc = 6.672e-8;
  const double pm = 1.672649e-24;
  const double em = 9.10953e-28;
  const double pi = 3.14159265;
  const double Re = 6.378e8;  // earth radius
  const double Me = 5.974e27; // earth mass 

  const double hf = 2.25;
  const double KERNEL_FACTOR = M_1_PI;
  
#define n_opencldevice0 4
  uint64 n_ocl_dev;
  uint64 n_per_device;
  std::vector<OpenCLDevice *> ov;
  cl_kernel ker[n_opencldevice0*2];
  cl_kernel ker_sph1[n_opencldevice0*2];
  cl_kernel ker_sph2[n_opencldevice0*2];
  cl_event ker_event[n_opencldevice0*2];

  cl_mem b_xi[n_opencldevice0];
  cl_mem b_jj[n_opencldevice0];
  cl_mem b_idx[n_opencldevice0];
  cl_mem b_acc[n_opencldevice0];
  cl_mem b_vel[n_opencldevice0], b_rho[n_opencldevice0];
  cl_mem b_alp[n_opencldevice0];

  std::string kernel_options;
  size_t globalThreads[1];
  uint64 opencl_offset[n_opencldevice0];
  uint64 opencl_grav_thread[n_opencldevice0];
  
  void SetUnit(double lu, double mu, double tu)
  {
    lunit = lu;
    munit = mu;
    tunit = tu;

    vunit = lunit/tunit;
    dunit = munit/pow(lunit, 3.0);

    eunit = vunit*vunit;      // ----> erg/g
    punit = munit*lunit/pow(tunit, 2.0)/pow(lunit, 2.0);

    //    divunit = dunit*(vunit/lunit); // dn * v/l
    aunit  = vunit/tunit;             
    deunit = eunit/tunit;

    std::cout << "# dunit " << dunit << "\n"; 
    std::cout << "# eunit " << eunit << "\n"; 
    std::cout << "# punit " << punit << "\n"; 

    tunit_inv = 1.0/tunit;
    lunit_inv = 1.0/lunit;
    munit_inv = 1.0/munit;
    vunit_inv = 1.0/vunit;
    dunit_inv = 1.0/dunit;
    eunit_inv = 1.0/eunit;
    punit_inv = 1.0/punit;
  }

  void InitSPHmodel() {
    CF.Load("ID.conf");

    // parameters
    nn_mean = CF.fGet("nn_mean");
    alpha_max = CF.fGet("alpha_max");
    alpha_min = CF.fGet("alpha_min");
    cfl_number = CF.fGet("cfl_number");

    dt_sys  = CF.fGet("dt_sys");
    dt_dump = CF.fGet("dt_dump");
    tend    = CF.fGet("tt_end");

    lunit = CF.fGet("lunit");
    munit = CF.fGet("munit");
    tunit = CF.fGet("tunit");
    eps   = CF.fGet("eps") * lunit;
    SetUnit(lunit, munit, tunit);
    
    if (CF.iGet("cgs") == 1) {
      std::cout << "CGS unit\n";
      grav = gc/(pow(lunit, 3.0)*pow(tunit, -2.0)*pow(munit,-1.0));
    } else {
      grav = 1.0/(pow(lunit, 3.0)*pow(tunit, 2.0)*pow(munit,-1.0));
    }
    std::cout << "grav  " << grav << "\n";
    std::cout << "vunit " << vunit << "\n";

    eos = new IDEOS(5.0/3.0);

    OTOO::log_energy.open("energy.log");
    OTOO::log_energy.precision( 6 );
    OTOO::log_energy.setf( std::ios_base::scientific, std::ios_base::floatfield );
  }

  template<class Tpsys>
  void ReadFile(Tpsys & psys, uint64 & n_glb, uint64 & n_loc) {
    NcFile f(CF.cGet("initfile"), NcFile::ReadOnly);
    if (!f.is_valid()) {
      std::cerr << "ReadCDF failed\n";
      exit(-1);
    }

    int i;
    double tmp;
    f.get_var("nall")->get(&i, 1);
    n_glb = i;
    n_loc = i;
    nall = i;
    psys.setNumberOfParticleLocal(nall);

    int n;
    f.get_var("nstep")->get(&n, 1);;
    OTOO::nstep = n;
    
    f.get_var("time")->get(&tmp, 1);;
    OTOO::t = tmp*tunit_inv;
    std::cout << "# Read CDF file at time = " << OTOO::t << "\n";
    
    double *buf_x = new double[nall];
    double *buf_y = new double[nall];
    double *buf_z = new double[nall];

    f.get_var("px")->get(buf_x, nall);
    f.get_var("py")->get(buf_y, nall);
    f.get_var("pz")->get(buf_z, nall);
    for(uint64 i = 0; i < nall; i++) {
      psys[i].x.x = buf_x[i]*lunit_inv;
      psys[i].x.y = buf_y[i]*lunit_inv;
      psys[i].x.z = buf_z[i]*lunit_inv;
    }

    f.get_var("vx")->get(buf_x, nall);
    f.get_var("vy")->get(buf_y, nall);
    f.get_var("vz")->get(buf_z, nall);
    for(uint64 i = 0; i < nall; i++) {
      psys[i].v.x = buf_x[i]*vunit_inv;
      psys[i].v.y = buf_y[i]*vunit_inv;
      psys[i].v.z = buf_z[i]*vunit_inv;
    }

    f.get_var("ms")->get(buf_x, nall);
    f.get_var("eg")->get(buf_y, nall);
    f.get_var("hs")->get(buf_z, nall);
    for(uint64 i = 0; i < nall; i++) {
      psys[i].m  = buf_x[i]*munit_inv;
      psys[i].eg = buf_y[i]*eunit_inv;
      psys[i].h  = buf_z[i]*lunit_inv;
    }

    f.get_var("dn")->get(buf_x, nall);
    for(uint64 i = 0; i < nall; i++) {
      psys[i].dn = buf_x[i]*dunit_inv;
    }
    
    delete buf_x;
    delete buf_y;
    delete buf_z;
  }

  template<class Tpsys>
  void WriteCDF(const char *filename, Tpsys & psys) {
    /*
    NcFile out(filename, NcFile::Replace);
    if (!out.is_valid()) {
      std::cerr << "WriteCDF failed\n";
      return;
    }

    int n = nstep;
    out.add_var("nstep", ncInt)->put(&n, 1);;
    n = nall;
    out.add_var("nall", ncInt)->put(&n, 1);
    double tmp;
    tmp = t*tunit;
    out.add_var("time", ncDouble)->put(&tmp, 1);;
    tmp = dt_sys*tunit;
    out.add_var("dt_sys", ncDouble)->put(&tmp, 1);;

    NcDim *na = out.add_dim("na", nall);
    double *buf_x = new double[nall];
    double *buf_y = new double[nall];
    double *buf_z = new double[nall];

    for(uint64 i = 0; i < nall; i++) {
      buf_x[i] = x(i,0)*lunit;
      buf_y[i] = x(i,1)*lunit;
      buf_z[i] = x(i,2)*lunit;
    }
    out.add_var("px", ncDouble, na)->put(buf_x, nall);
    out.add_var("py", ncDouble, na)->put(buf_y, nall);
    out.add_var("pz", ncDouble, na)->put(buf_z, nall);

    for(uint64 i = 0; i < nall; i++) {
      buf_x[i] = v(i,0)*vunit;
      buf_y[i] = v(i,1)*vunit;
      buf_z[i] = v(i,2)*vunit;
    }
    out.add_var("vx", ncDouble, na)->put(buf_x, nall);
    out.add_var("vy", ncDouble, na)->put(buf_y, nall);
    out.add_var("vz", ncDouble, na)->put(buf_z, nall);

    for(uint64 i = 0; i < nall; i++) {
      buf_x[i] = m(i) *munit;
      buf_y[i] = eg(i)*eunit;
      buf_z[i] = h(i) *lunit;
    }
    out.add_var("ms", ncDouble, na)->put(buf_x, nall);
    out.add_var("eg", ncDouble, na)->put(buf_y, nall);
    out.add_var("hs", ncDouble, na)->put(buf_z, nall);

    for(uint64 i = 0; i < nall; i++) {
      buf_x[i] = dn(i)*dunit;
      buf_y[i] = TT(i);
      buf_z[i] = I(i);
    }
    out.add_var("dn", ncDouble, na)->put(buf_x, nall);
    out.add_var("TT", ncDouble, na)->put(buf_y, nall);
    out.add_var("Index", ncDouble, na)->put(buf_z, nall);

    delete buf_x;
    delete buf_y;
    delete buf_z;
    */
  }

  
  #include "kernel.file"

  template<class typ>
  struct Vec4 {
    typ x, y, z, w;
  };

  template<class typ>
  struct Vec2 {
    typ x, y;
  };
  
#define __NI__  600000
#define __NJ__  18000000

  uint64 nimax, njmax, nnmax;

  struct Vec4<double> *xxx, *vvv, *rho, *ggg;
  //  double xxx[__NJ__][4];
  //  double vvv[__NJ__][4];
  //  double rho[__NJ__][4];

  // double ggg[__NI__][4];
  //  double alp[__NJ__];
  //  int    idx[__NJ__];
  //  int    jjj[__NI__][2];

  double *alp;
  int *idx;
  struct Vec2<int> *jjj;

  void SetupOpenCL(int ip = 0, int id = 0, int __ni = __NI__, int __nj = __NJ__)
  {
    if (id < 0) {
      id = -id;
      n_ocl_dev = id;
    } else {
      n_ocl_dev = 1;   
    }

    nimax = __ni;
    njmax = 2*nimax;
    nnmax = __nj;

    //    n_per_device = nalloc0/n_ocl_dev;
    for(uint64 i = 0; i < n_ocl_dev; i++) {
      ov.push_back(new OpenCLDevice(ip));    
      ov[i]->SetupContext( n_ocl_dev == 1 ? id :  i );

      ov[i]->SetKernelOptions(kernel_options);
      ov[i]->BuildOpenCLKernels(kernel_str);

      ker[i] = ov[i]->GetKernel("grav_index");
      ker_sph1[i] = ov[i]->GetKernel("ker_sph1_index");
      ker_sph2[i] = ov[i]->GetKernel("ker_sph2_index");

      cl_int status = CL_SUCCESS;
      // read only buffers (HOST -> GPU)
      b_jj[i]   = clCreateBuffer(ov[i]->ctx, CL_MEM_READ_ONLY, nimax*sizeof(cl_int2), NULL, &status);

      b_xi[i]   = clCreateBuffer(ov[i]->ctx, CL_MEM_READ_ONLY, njmax*sizeof(cl_double4), NULL, &status);
      b_vel[i]  = clCreateBuffer(ov[i]->ctx, CL_MEM_READ_ONLY, njmax*sizeof(cl_double4), NULL, &status);
      b_rho[i]  = clCreateBuffer(ov[i]->ctx, CL_MEM_READ_ONLY, njmax*sizeof(cl_double4), NULL, &status);
      b_alp[i]  = clCreateBuffer(ov[i]->ctx, CL_MEM_READ_ONLY, njmax*sizeof(cl_double), NULL, &status);

      b_idx[i]  = clCreateBuffer(ov[i]->ctx, CL_MEM_READ_ONLY, nnmax*sizeof(cl_int), NULL, &status);

      // output buffers (GPU -> HOST)
      b_acc[i] = clCreateBuffer(ov[i]->ctx, CL_MEM_WRITE_ONLY, nimax*sizeof(cl_double4), NULL, &status);

      clSetKernelArg(OTOO::ker[i], 0, sizeof(cl_mem), (void *)&OTOO::b_xi[i]);
      clSetKernelArg(OTOO::ker[i], 1, sizeof(cl_mem), (void *)&OTOO::b_jj[i]);
      clSetKernelArg(OTOO::ker[i], 2, sizeof(cl_mem), (void *)&OTOO::b_idx[i]);
      clSetKernelArg(OTOO::ker[i], 3, sizeof(cl_mem), (void *)&OTOO::b_acc[i]);
      
      clSetKernelArg(OTOO::ker_sph1[i], 0, sizeof(cl_mem), (void *)&OTOO::b_xi[i]);
      clSetKernelArg(OTOO::ker_sph1[i], 1, sizeof(cl_mem), (void *)&OTOO::b_vel[i]);
      clSetKernelArg(OTOO::ker_sph1[i], 2, sizeof(cl_mem), (void *)&OTOO::b_jj[i]);
      clSetKernelArg(OTOO::ker_sph1[i], 3, sizeof(cl_mem), (void *)&OTOO::b_idx[i]);
      clSetKernelArg(OTOO::ker_sph1[i], 4, sizeof(cl_mem), (void *)&OTOO::b_acc[i]);

      clSetKernelArg(OTOO::ker_sph2[i], 0, sizeof(cl_mem), (void *)&OTOO::b_xi[i]);
      clSetKernelArg(OTOO::ker_sph2[i], 1, sizeof(cl_mem), (void *)&OTOO::b_vel[i]);
      clSetKernelArg(OTOO::ker_sph2[i], 2, sizeof(cl_mem), (void *)&OTOO::b_rho[i]);
      clSetKernelArg(OTOO::ker_sph2[i], 3, sizeof(cl_mem), (void *)&OTOO::b_alp[i]);
      clSetKernelArg(OTOO::ker_sph2[i], 4, sizeof(cl_mem), (void *)&OTOO::b_jj[i]);
      clSetKernelArg(OTOO::ker_sph2[i], 5, sizeof(cl_mem), (void *)&OTOO::b_idx[i]);
      clSetKernelArg(OTOO::ker_sph2[i], 6, sizeof(cl_mem), (void *)&OTOO::b_acc[i]);
    }

    xxx = new struct Vec4<double>[njmax];
    vvv = new struct Vec4<double>[njmax];
    rho = new struct Vec4<double>[njmax];
    alp = new double[njmax];

    idx = new int[nnmax];

    ggg = new struct Vec4<double>[nimax];
    jjj = new struct Vec2<int>[nimax];
  }  

  void ReallocateOpenCLMemory(int __nnmax)
  {
    cl_int status = CL_SUCCESS;

    //    std::cout << nnmax << " " << __nnmax << "\n";
    
    nnmax = (int)(__nnmax*1.25);

    //    std::cout << nnmax << " " << __nnmax << "\n";
    
    for(uint64 i = 0; i < n_ocl_dev; i++) {
      clReleaseMemObject(b_idx[i]);
      b_idx[i]  = clCreateBuffer(ov[i]->ctx, CL_MEM_READ_ONLY, nnmax*sizeof(cl_int), NULL, &status);
      assert(status == CL_SUCCESS);

      clSetKernelArg(OTOO::ker[i],      2, sizeof(cl_mem), (void *)&OTOO::b_idx[i]);
      clSetKernelArg(OTOO::ker_sph1[i], 3, sizeof(cl_mem), (void *)&OTOO::b_idx[i]);
      clSetKernelArg(OTOO::ker_sph2[i], 5, sizeof(cl_mem), (void *)&OTOO::b_idx[i]);
    }
    
    delete[] idx;
    
    idx = new int[nnmax];    
    //    std::cerr << "Reallocate OpenCL Memory\n";
  }

  void ClearKernelTime()
  {
    kernel_count = 0;
    kernel_time_grav = kernel_time_sph1 = kernel_time_sph2 = 0.0;
  }
  
}
#endif
