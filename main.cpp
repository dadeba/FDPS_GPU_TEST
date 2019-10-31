// Include the standard C++ headers
#include <cmath>
#include <math.h>
#include <cfloat>
#include <cstdio>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <vector>
#include <sys/stat.h>
#include <time.h>

// Include the header file of FDPS
#include <particle_simulator.hpp>

#include <netcdfcpp.h>
#include "Integrate.hpp"
#include "user_defined.hpp"

void OutNN(const PS::ParticleSystem<FP>& psys) {
  double nn_max, nn_min, nn_mean;
    
  uint64 nall = psys.getNumberOfParticleLocal();
  
  nn_max = nn_min = psys[0].nn;
  nn_mean = 0.0;
  for (uint64 i = 0; i < nall; i++) {
    nn_max = std::max(psys[i].nn, nn_max);
    nn_min = std::min(psys[i].nn, nn_min);
    nn_mean += psys[i].nn;
  }

  PS::F64 nn_max_g  = PS::Comm::getMaxValue(nn_max);
  PS::F64 nn_min_g  = PS::Comm::getMinValue(nn_min);
  PS::F64 nn_mean_g = PS::Comm::getSum(nn_mean)/(double)psys.getNumberOfParticleGlobal();
  
  if (PS::Comm::getRank() == 0) {
    std::cout << "\tNNcheck max:" << nn_max_g << " min:" << nn_min_g << " mean:" << nn_mean_g << "\n";
  }
}

template <class func>
void Apply(PS::ParticleSystem<FP>& psys, func f) {
  const uint64 nall = psys.getNumberOfParticleLocal();

#ifdef PARTICLE_SIMULATOR_THREAD_PARALLEL		    
#pragma omp for
#endif
  for(uint64 i = 0; i < nall; i++) f(psys[i]);
}

struct UpdateCS {
  void operator() (FP &p) {
    p.cs = eos->GetS(p.dn, p.eg);
  }
};

struct UpdatePP {
  void operator() (FP &p) {
    p.pp = eos->GetP(p.dn, p.eg)/(p.dn*p.dn);
  }
};

struct UpdateVV {
  void operator() (FP &p) {
    double div = fabs(p.div);
    p.vv = div/(div + p.rot + 1.0e-4*p.cs/p.h);
  }
};

struct UpdateSmoothingLength {
  void operator() (FP &p) {
    double dum1, dum2, hs;
    dum1 = pow((double)(OTOO::nn_mean)/p.nn, 1.0/3.0);
    dum2 = aValue(dum1);
    hs = p.h*(1.0 + dum2*(dum1 - 1.0));
    hs = std::min(hs, 1.25*p.h);
    hs = std::max(hs, 0.9 *p.h);
    p.h = hs;
  }
};

struct Kick {
  void operator() (FP &p) {
    // save
    p.v_old  = p.v    + 0.5*p.acc*OTOO::dt;
    p.eg_old = p.eg   + 0.5*p.de *OTOO::dt;

    // predict
    p.x  += p.v*OTOO::dt + 0.5*p.acc*OTOO::dt*OTOO::dt;
    p.v  += p.acc*OTOO::dt;
    p.eg += p.de *OTOO::dt;
  }
};

struct Drift {
  void operator() (FP &p) {
    p.v  = p.v_old  + 0.5*p.acc*OTOO::dt;
    p.eg = p.eg_old + 0.5*p.de *OTOO::dt;
  }
};

struct ForceSum {
  void operator() (FP &p) {
    p.acc = p.acc_hydro + p.acc_grav*OTOO::grav;
  }
};

void CalcDT(PS::ParticleSystem<FP>& psys) {
  double dt_min_loc = OTOO::dt_sys;

  Apply(psys, UpdateCS());

  const uint64 nall = psys.getNumberOfParticleLocal();
  for(uint64 i = 0; i < nall; i++) {
    double h   = psys[i].h;
    double div = fabs(psys[i].div);
    double cs  = psys[i].cs;
    double al  = psys[i].al;
    dt_min_loc = std::min(dt_min_loc, ((OTOO::cfl_number*h)/(h*div + cs + 1.2*al*cs)));
  }

  double dt_min = PS::Comm::getMinValue(dt_min_loc);
  
  if (dt_min < 1.0e-5f*OTOO::dt_sys) {
    std::cerr << "Timestep!!!!" << dt_min << "\t" << OTOO::dt_sys << "\n"; 
    //    std::cerr << "cs max " << cs.maxCoeff() << "\n";
    //    std::cerr << "index " << ii << "\n";
    //    std::cerr << "cs " << cs[ii] << " " << cs.sum()/nall << "\n";
    //    std::cerr << "h " << h[ii] << " " << h.sum()/nall << "\n";
    //    std::cerr << "nn " << nn[ii] << "\n";
    //    std::cerr << "d e " << dn[ii]*dunit << " " << eg[ii]*eunit << "\n";
    exit(-1);
  }

  double dum = OTOO::dt_sys;
  while(dum > dt_min) dum *= 0.5;

  if (dum > OTOO::dt) {
    if (fmod(OTOO::t, dum) == 0.0) {
      OTOO::dt = dum;
    }
  } else {
    OTOO::dt = dum;
  }
}

void checkConservativeVariables(const PS::ParticleSystem<FP>& psys)
{
  OTOO::elapsed_time_loop = PS::GetWtime() - OTOO::start_time_loop;

  PS::F64    ekin_loc = 0.0;
  PS::F64    epot_loc = 0.0;
  PS::F64    eth_loc  = 0.0; 
  PS::F64vec mom_loc  = 0.0; 

  for (PS::S32 i = 0; i < psys.getNumberOfParticleLocal(); i++) {
    ekin_loc += 0.5 * psys[i].m * psys[i].v * psys[i].v;
    epot_loc += 0.5 * psys[i].m * (psys[i].pot + psys[i].m / OTOO::eps);
    eth_loc  += psys[i].m * psys[i].eg;
    mom_loc  += psys[i].m * psys[i].v;
  }

  PS::F64 ekin    = PS::Comm::getSum(ekin_loc);
  PS::F64 epot    = PS::Comm::getSum(epot_loc);
  PS::F64 eth     = PS::Comm::getSum(eth_loc);
  PS::F64vec mom  = PS::Comm::getSum(mom_loc);

  static bool is_initialized = false;
  static PS::F64 emech_ini, etot_ini;
  if (is_initialized == false) {
    emech_ini = ekin + epot;
    etot_ini  = ekin + epot + eth;
    is_initialized = true;
  }

  if (PS::Comm::getRank() == 0) {
    const PS::F64 emech = ekin + epot;
    const PS::F64 etot  = ekin + epot + eth;
    const PS::F64 relerr_tot  = (etot  - etot_ini)/etot_ini;

    std::cout << OTOO::t << "\t" << OTOO::dt << "\t"
	      << etot << "\t" << ekin << "\t" << epot << "\t" << eth << "\t"
	      << 100.0*relerr_tot << " %\n";

    double emunit = OTOO::eunit*OTOO::munit;
    
    OTOO::log_energy
      << OTOO::nstep << " " 
      << OTOO::t*OTOO::tunit << " " 
      << OTOO::dt*OTOO::tunit << " " 
      << (double)etot*emunit << " "
      << (double)ekin*emunit << " " 
      << (double)epot*emunit << " " 
      << (double)eth *emunit << " "
      << (OTOO::elapsed_time_loop)/3600.0 << "\n";
    OTOO::log_energy.flush();
  }
}

int main(int argc, char* argv[]){
  // Configure stdout & stderr
  //  std::cout << std::setprecision(15);
  //  std::cerr << std::setprecision(15);

  std::cerr.precision( 4 );
  std::cerr.setf( std::ios_base::scientific, std::ios_base::floatfield );

  // Initialize FDPS
  PS::Initialize(argc, argv);

  // Display # of MPI processes and threads
  PS::S32 nprocs = PS::Comm::getNumberOfProc();
  PS::S32 nthrds = PS::Comm::getNumberOfThread();
  if (PS::Comm::getRank() == 0) {
    std::cout << "===========================================" << std::endl
	      << " # of processes is " << nprocs               << std::endl
	      << " # of thread is    " << nthrds               << std::endl
	      << "===========================================" << std::endl;
  }

  // Make instances of ParticleSystem and initialize them
  PS::ParticleSystem<FP> psys;
  psys.initialize();

  OTOO::InitSPHmodel();
  OTOO::SetupOpenCL(0, PS::Comm::getRank());
    
  uint64 n_tot  = 0;
  uint64 n_loc  = 0;
  if(PS::Comm::getRank() == 0) {
    OTOO::ReadFile(psys, n_tot, n_loc);
  } else {
    psys.setNumberOfParticleLocal(n_loc);
  }
  
  const PS::F32 coef_ema = 0.3;
  PS::DomainInfo dinfo;
  dinfo.initialize(coef_ema);
  dinfo.decomposeDomainAll(psys);
  psys.exchangeParticle(dinfo);
  n_loc = psys.getNumberOfParticleLocal();

  for (uint64 i = 0; i < n_loc; i++) {
    psys[i].al = OTOO::alpha_min;
  }

  OTOO::kernel_time_grav = 0.0;
  OTOO::kernel_time_sph1 = 0.0;
  OTOO::kernel_time_sph2 = 0.0;
  
  // Perform domain decomposition 
  dinfo.collectSampleParticle(psys);
  dinfo.decomposeDomain();
  psys.exchangeParticle(dinfo);

  const PS::F32 theta_grav = 0.5;
  PS::S32 n_leaf_limit = 8;
  PS::S32 n_group_limit = 1000;
  
  PS::TreeForForceLong<G, EP_grav, EP_grav>::Monopole tree_grav;
  tree_grav.initialize(3 * n_loc, theta_grav, n_leaf_limit, n_group_limit);
  
  const PS::S32 n_walk_limit = 100;
  const PS::S32 tag_max = 1;
  tree_grav.calcForceAllAndWriteBackMultiWalk(DispatchKernel,
					      RetrieveKernel,
					      tag_max,
					      psys,
					      dinfo,
					      n_walk_limit);
  
  OTOO::start_time_loop = PS::GetWtime();
  checkConservativeVariables(psys);

  
  PS::TreeForForceShort<SPH1, EP_hydro, EP_hydro>::Symmetry tree_sph1;
  tree_sph1.initialize(3 * n_loc, theta_grav, n_leaf_limit, n_group_limit);

  PS::TreeForForceShort<SPH2, EP_hydro, EP_hydro>::Symmetry tree_sph2;
  tree_sph2.initialize(3 * n_loc, theta_grav, n_leaf_limit, n_group_limit);

  for(int l = 0; l < 70; l++) {
    OTOO::kernel_time_sph1 = 0.0;
    OTOO::kernel_count = 0;
    tree_sph1.calcForceAllAndWriteBackMultiWalk(DispatchKernelSPH1,
						RetrieveKernelSPH1,
						tag_max,
						psys,
						dinfo,
						n_walk_limit);
    Apply(psys, UpdateSmoothingLength());
    psys.exchangeParticle(dinfo);
    if (l % 10 == 0) OutNN(psys);
  }
  
  tree_sph1.calcForceAllAndWriteBackMultiWalk(DispatchKernelSPH1,
					      RetrieveKernelSPH1,
					      tag_max,
					      psys,
					      dinfo,
					      n_walk_limit);

  Apply(psys, UpdatePP());
  Apply(psys, UpdateCS());
  Apply(psys, UpdateVV());

  tree_sph2.calcForceAllAndWriteBackMultiWalk(DispatchKernelSPH2,
					      RetrieveKernelSPH2,
					      tag_max,
					      psys,
					      dinfo,
					      n_walk_limit);
  Apply(psys, ForceSum());

  // Main loop for time integration
  OTOO::start_time_loop = PS::GetWtime();
  while(OTOO::t < OTOO::tend) {
    PS::INTERACTION_LIST_MODE int_mode = PS::REUSE_LIST;
    CalcDT(psys);
    if (PS::Comm::getRank() == 0) {
      std::cout << OTOO::nstep << " : " << OTOO::t << "    " << OTOO::dt << "\n";
      std::cout.flush();
    }
    
    // predict
    Apply(psys, Kick());

    if (OTOO::nstep % 8 == 0) {
      dinfo.collectSampleParticle(psys);
      dinfo.decomposeDomain();
      psys.exchangeParticle(dinfo);
      int_mode = PS::MAKE_LIST_FOR_REUSE;
    }
    
    tree_grav.calcForceAllAndWriteBackMultiWalk(DispatchKernel,
						RetrieveKernel,
						tag_max,
						psys,
						dinfo,
						n_walk_limit,
						true,
						int_mode);

    tree_sph1.calcForceAllAndWriteBackMultiWalk(DispatchKernelSPH1,
						RetrieveKernelSPH1,
						tag_max,
						psys,
						dinfo,
						n_walk_limit,
						true,
						int_mode);

    Apply(psys, UpdateSmoothingLength());

    tree_sph1.calcForceAllAndWriteBackMultiWalk(DispatchKernelSPH1,
						RetrieveKernelSPH1,
						tag_max,
						psys,
						dinfo,
						n_walk_limit,
						true,
						int_mode);

    Apply(psys, UpdatePP());
    Apply(psys, UpdateCS());
    Apply(psys, UpdateVV());
    
    tree_sph2.calcForceAllAndWriteBackMultiWalk(DispatchKernelSPH2,
						RetrieveKernelSPH2,
						tag_max,
						psys,
						dinfo,
						n_walk_limit,
						true,
						int_mode);

    Apply(psys, ForceSum());

    // correct
    Apply(psys, Drift());
    
    OTOO::t += OTOO::dt;
    OTOO::nstep++;

    if (fmod(OTOO::t, OTOO::dt_sys) == 0.0) {
      checkConservativeVariables(psys);
      OutNN(psys);
    }

    if (fmod(OTOO::t, OTOO::dt_dump) == 0.0) {
    }
  }
  
  // Finalize FDPS
  PS::Finalize();
  return 0;
}
