class G {
public:
  PS::F64vec acc;
  PS::F64    pot;
  void clear(){
    acc = 0.0;
    pot = 0.0;
  }
};

class SPH1 { // density, rot, dd, nn
public:
  PS::F64    dn;  // density
  PS::F64vec rot; // rotation
  PS::F64    dd;  // time derivative of density
  PS::F64    nn;  // neighber number

  void clear(){
    dn  = 0.0;
    dd  = 0.0;
    nn  = 0.0;
    rot = 0.0;
  }
};

class SPH2 {
public:
  PS::F64vec acc;
  PS::F64    de;  // dot energy density

  void clear(){
    acc = 0.0;
    de  = 0.0;
  }
};

class SPH0 {
public:
  PS::F64 x;
  PS::F64 y;
  PS::F64 z;
  PS::F64 w;

  void clear(){
    x = 0.0;
    y = 0.0;
    z = 0.0;
    w = 0.0;
  }
};

class FP {
public:
  PS::S64    id;
  PS::F64    m;
  PS::F64vec x;
  PS::F64vec v;
  PS::F64vec acc;
  PS::F64    pot;       // gravitational potential

  PS::F64vec v_old;
  PS::F64    eg_old;

  PS::F64vec acc_grav;  // gravitational acceleration
  PS::F64vec acc_hydro; // acceleration due to pressure-gradient

  PS::F64    h;   // smoothing length
  PS::F64    dn;  // mass density
  PS::F64    eg;  // specific internal energy
  PS::F64    de;
  PS::F64    dd; 
  PS::F64    rot; // rotation of velocity
  PS::F64    div; // divergence of velocity
  PS::F64    nn;
  PS::F64    cs;  // sound speed
  PS::F64    vv;  // vis
  PS::F64    pp;  // pressure
  PS::F64    al;  // vis alpha

  //#define KERNEL_FACTOR M_1_PI
  
  void copyFromForce(const G& f) {
    this->acc_grav = f.acc;
    this->pot      = f.pot;
  }

  void copyFromForce(const SPH1& f){
    this->dn = f.dn*OTOO::KERNEL_FACTOR;
    this->dd = f.dd*OTOO::KERNEL_FACTOR;
    this->nn = f.nn;

    double dn_inv = 1.0/this->dn;
    this->div = -this->dd*dn_inv;

    //    this->rot = sqrt(f.rot*f.rot)*KERNEL_FACTOR*dn_inv;

    this->rot = f.rot.x*OTOO::KERNEL_FACTOR*dn_inv;
  }

  void copyFromForce(const SPH2& f){
    this->acc_hydro = f.acc*OTOO::KERNEL_FACTOR;
    this->de        = f.de *OTOO::KERNEL_FACTOR;
  }
  
  PS::F64 getCharge() const{
    return this->m;
  }
  
  PS::F64vec getPos() const{
    return this->x;
  }

  PS::F64 getRSearch() const{
    return OTOO::hf*this->h;
  }

  void setPos(const PS::F64vec& pos){
    this->x = pos;
  }
};

//** Essential Particle Class
class EP_grav {
public:
  PS::S64 id;
  PS::F64 m;
  PS::F64vec x;

  void copyFromFP(const FP& fp) {
    this->id = fp.id;
    this->m  = fp.m;
    this->x  = fp.x;
  }
  PS::F64 getCharge() const {
    return this->m;
  }
  PS::F64vec getPos() const {
    return this->x;
  }
};

class EP_hydro {
public:
  PS::S64    id;
  PS::F64vec x;
  PS::F64vec v;
  PS::F64    m;
  PS::F64    h;
  PS::F64    dn;
  PS::F64    pp;
  PS::F64    vv;
  PS::F64    cs;
  PS::F64    al;

  void copyFromFP(const FP& fp){
    this->id = fp.id;
    this->x  = fp.x;
    this->v  = fp.v;
    this->m  = fp.m;
    this->h  = fp.h;
    this->dn = fp.dn;
    this->pp = fp.pp;
    this->vv = fp.vv;
    this->cs = fp.cs;
    this->al = fp.al;
  }
  PS::F64vec getPos() const{
    return this->x;
  }
  PS::F64 getRSearch() const{
    return OTOO::hf * this->h;
  }
  void setPos(const PS::F64vec& pos){
    this->x = pos;
  }
};

static double aValue(double s)
{
  if (s < 1.0) {
    return 0.2*(1.0+s*s);
  } else {
    return 0.2*(1.0+1.0/(s*s*s));
  }
}

void CheckNNMAX(const PS::S32 n_walk, const PS::S32 *n_epj, const PS::S32 *n_spj)
{
  int c = 0;
  for(int iw = 0; iw < n_walk; iw++){
    c += n_epj[iw];
    if (n_spj != NULL) c += n_spj[iw];
  }
  if (c >= OTOO::nnmax) {
    OTOO::ReallocateOpenCLMemory(c);
  }
  //  assert( c < OTOO::nnmax );
}

PS::S32 DispatchKernelIndex(
			    const PS::S32   tag,
			    const PS::S32   n_walk,
			    const EP_grav  *epi[],
			    const PS::S32   n_epi[],
			    const PS::S32  *id_epj[],
			    const PS::S32  *n_epj,
			    const PS::S32  *id_spj[],
			    const PS::S32  *n_spj,
			    const EP_grav  *epj,
			    const PS::S32   n_epj_tot,
			    const PS::SPJMonopole  *spj,
			    const PS::S32   n_spj_tot,
			    const bool send_flag)
{
  const PS::F64 eps2 = OTOO::eps*OTOO::eps;
  uint64 c = 0;

  if (n_walk == 0) return 0;

  for(int iw = 0; iw < n_walk; iw++){
    for(int i = 0; i < n_epi[iw]; i++){
      PS::F64vec xi = epi[iw][i].getPos();
      OTOO::xxx[c].x = xi.x;
      OTOO::xxx[c].y = xi.y;
      OTOO::xxx[c].z = xi.z;
      OTOO::xxx[c].w = (double)iw;
      c++;
    }
  }
  int ni = c;

  assert( ni                     < OTOO::nimax );
  assert( ni+n_epj_tot+n_spj_tot < OTOO::njmax );
  CheckNNMAX(n_walk, n_epj, n_spj);
  
  for(int j = 0; j < n_epj_tot; j++){
    PS::F64vec xj = epj[j].getPos();
    OTOO::xxx[c].x = xj.x;
    OTOO::xxx[c].y = xj.y;
    OTOO::xxx[c].z = xj.z;
    OTOO::xxx[c].w = epj[j].getCharge();
    c++;
  }
  for(int j = 0; j < n_spj_tot; j++){
    PS::F64vec xj = spj[j].getPos();
    OTOO::xxx[c].x = xj.x;
    OTOO::xxx[c].y = xj.y;
    OTOO::xxx[c].z = xj.z;
    OTOO::xxx[c].w = spj[j].getCharge();
    c++;
  }
  int nall = c;
  
  c = 0;
  for(int iw = 0; iw < n_walk; iw++){
    OTOO::jjj[iw].x = c;

    int offset = ni;
    for(int j = 0; j < n_epj[iw]; j++){
      int jj = id_epj[iw][j] + offset;
      OTOO::idx[c] = jj;
      c++;
    }

    offset = ni + n_epj_tot;
    for(int j = 0; j < n_spj[iw]; j++){
      int jj = id_spj[iw][j] + offset;
      OTOO::idx[c] = jj;
      c++;
    }
    OTOO::jjj[iw].y = c;
  }
  int nidx = c;
  
  clSetKernelArg(OTOO::ker[0], 4, sizeof(double), (void *)&eps2);

  cl_bool flag = CL_FALSE; 
  
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_xi[0],  flag, 0, nall*sizeof(cl_double4),   OTOO::xxx, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_jj[0],  flag, 0, n_walk*sizeof(cl_int2), OTOO::jjj, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_idx[0], flag, 0, nidx*sizeof(cl_int),       OTOO::idx, 0, NULL, NULL);
  
  OTOO::globalThreads[0] = ni;
  cl_int status = CL_SUCCESS;
  status = clEnqueueNDRangeKernel(OTOO::ov[0]->q, OTOO::ker[0], 1,
				  NULL, OTOO::globalThreads, NULL, 0, NULL,
				  &OTOO::ker_event[0]);

  assert(status == CL_SUCCESS);

  /*
  for(int i = 0; i < ni; i++){
    PS::F64vec xi;
    xi.x = OTOO::xxx[i][0];
    xi.y = OTOO::xxx[i][1];
    xi.z = OTOO::xxx[i][2];
    int iw = (int)OTOO::xxx[i][3];

    PS::F64vec ai = 0.0;
    PS::F64 poti = 0.0;

    for(int j = OTOO::jjj[iw][0]; j < OTOO::jjj[iw][1]; j++){
      int jj = OTOO::idx[j];
      PS::F64vec xj;
      PS::F64    mj;
      xj.x = OTOO::xxx[jj][0];
      xj.y = OTOO::xxx[jj][1];
      xj.z = OTOO::xxx[jj][2];
      mj   = OTOO::xxx[jj][3];

      PS::F64vec rij    = xi - xj;
      PS::F64    r3_inv = rij * rij + eps2;
      PS::F64    r_inv  = 1.0/sqrt(r3_inv);
      r3_inv  = r_inv * r_inv;
      r_inv  *= mj;
      r3_inv *= r_inv;
      ai     -= r3_inv * rij;
      poti   -= r_inv;
    }
    
    OTOO::ggg[i][0] = ai.x;
    OTOO::ggg[i][1] = ai.y;
    OTOO::ggg[i][2] = ai.z;
    OTOO::ggg[i][3] = poti;
  }
  */
  
  return 0;
}

PS::S32 RetrieveKernel(const PS::S32 tag,
                       const PS::S32 n_walk,
                       const PS::S32 ni[],
                       G *force[])
{
  uint64 c = 0;

  if (n_walk == 0) return 0;
  
  c = OTOO::globalThreads[0];
  clFinish(OTOO::ov[0]->q);

  {
    cl_ulong st, en;
    clGetEventProfilingInfo (OTOO::ker_event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &st, NULL);
    clGetEventProfilingInfo (OTOO::ker_event[0], CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &en, NULL);
    OTOO::kernel_time_grav += (en - st)*1.0e-9;
  }
  
  cl_bool flag = CL_TRUE;
  clEnqueueReadBuffer(OTOO::ov[0]->q, OTOO::b_acc[0], flag, 0, c*sizeof(cl_double4), OTOO::ggg, 0, NULL, NULL);
  
  c = 0;
  for(int iw=0; iw<n_walk; iw++){
    for(int i=0; i<ni[iw]; i++){
      force[iw][i].acc.x = OTOO::ggg[c].x;
      force[iw][i].acc.y = OTOO::ggg[c].y;
      force[iw][i].acc.z = OTOO::ggg[c].z;
      force[iw][i].pot   = OTOO::ggg[c].w;
      c++;
    }
  }

  return 0;
}

PS::S32 DispatchKernelSPH1Index(
			   const PS::S32   tag,
			   const PS::S32   n_walk,
			   const EP_hydro *epi[],
			   const PS::S32   n_epi[],
			   const PS::S32  *id_epj[],
			   const PS::S32  *n_epj,
			   const EP_hydro *epj,
			   const PS::S32   n_epj_tot,
			   const bool send_flag)
{
  uint64 c = 0;
  if (n_walk == 0) return 0;
  
  for(int iw = 0; iw < n_walk; iw++){
    for(int i = 0; i < n_epi[iw]; i++){
      PS::F64vec xi = epi[iw][i].getPos();
      OTOO::xxx[c].x = xi.x;
      OTOO::xxx[c].y = xi.y;
      OTOO::xxx[c].z = xi.z;
      OTOO::xxx[c].w = (double)iw;

      PS::F64vec vi = epi[iw][i].v;
      OTOO::vvv[c].x = vi.x;
      OTOO::vvv[c].y = vi.y;
      OTOO::vvv[c].z = vi.z;
      OTOO::vvv[c].w = epi[iw][i].h;
      c++;
    }
  }
  int ni = c;

  assert( ni           < OTOO::nimax );
  assert( ni+n_epj_tot < OTOO::njmax );
  CheckNNMAX(n_walk, n_epj, NULL);
  
  for(int j = 0; j < n_epj_tot; j++){
    PS::F64vec xj = epj[j].getPos();
    OTOO::xxx[c].x = xj.x;
    OTOO::xxx[c].y = xj.y;
    OTOO::xxx[c].z = xj.z;
    OTOO::xxx[c].w = epj[j].m;

    PS::F64vec vi = epj[j].v;
    OTOO::vvv[c].x = vi.x;
    OTOO::vvv[c].y = vi.y;
    OTOO::vvv[c].z = vi.z;
    OTOO::vvv[c].w = epj[j].h;
    c++;
  }
  int nall = c;

  c = 0;  
  for(int iw = 0; iw < n_walk; iw++){
    OTOO::jjj[iw].x = c;

    int offset = ni;
    for(int j = 0; j < n_epj[iw]; j++){
      int jj = id_epj[iw][j] + offset;
      OTOO::idx[c] = jj;
      c++;
    }

    OTOO::jjj[iw].y = c;
  }
  int nidx = c;

  cl_bool flag = CL_FALSE; 
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_xi[0],  flag, 0, nall*sizeof(cl_double4), OTOO::xxx, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_vel[0], flag, 0, nall*sizeof(cl_double4), OTOO::vvv, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_jj[0],  flag, 0, n_walk*sizeof(cl_int2),  OTOO::jjj, 0, NULL, NULL);

  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_idx[0], flag, 0, nidx*sizeof(cl_int),       OTOO::idx, 0, NULL, NULL);
  
  OTOO::globalThreads[0] = ni;
  cl_int status = CL_SUCCESS;
  status = clEnqueueNDRangeKernel(OTOO::ov[0]->q, OTOO::ker_sph1[0],
				  1, NULL, OTOO::globalThreads, NULL, 0, NULL,
				  &OTOO::ker_event[0]);
  assert(status == CL_SUCCESS);

  return 0;
}

PS::S32 RetrieveKernelSPH1(const PS::S32 tag,
                       const PS::S32 n_walk,
                       const PS::S32 ni[],
                       SPH1 *force[])
{
  uint64 c = 0;
  if (n_walk == 0) return 0;

  c = OTOO::globalThreads[0];
  clFinish(OTOO::ov[0]->q);

  {
    cl_ulong st, en;
    clGetEventProfilingInfo (OTOO::ker_event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &st, NULL);
    clGetEventProfilingInfo (OTOO::ker_event[0], CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &en, NULL);
    OTOO::kernel_time_sph1 += (en - st)*1.0e-9;
  }
  
  cl_bool flag = CL_TRUE;
  clEnqueueReadBuffer(OTOO::ov[0]->q, OTOO::b_acc[0], flag, 0, c*sizeof(cl_double4), OTOO::ggg, 0, NULL, NULL);

  c = 0;
  for(int iw=0; iw<n_walk; iw++){
    for(int i=0; i<ni[iw]; i++){
      force[iw][i].dn    = OTOO::ggg[c].w;
      force[iw][i].rot.x = OTOO::ggg[c].x;
      force[iw][i].dd    = OTOO::ggg[c].y;
      force[iw][i].nn    = OTOO::ggg[c].z;
      c++;
    }
  }

  return 0;
}

PS::S32 DispatchKernelSPH2Index(
			   const PS::S32   tag,
			   const PS::S32   n_walk,
			   const EP_hydro *epi[],
			   const PS::S32   n_epi[],
			   const PS::S32  *id_epj[],
			   const PS::S32  *n_epj,
			   const EP_hydro *epj,
			   const PS::S32   n_epj_tot,
			   const bool send_flag)
{
  uint64 c = 0;
  if (n_walk == 0) return 0;
 
  for(int iw = 0; iw < n_walk; iw++){
    for(int i = 0; i < n_epi[iw]; i++){
      PS::F64vec xi = epi[iw][i].getPos();
      OTOO::xxx[c].x = xi.x;
      OTOO::xxx[c].y = xi.y;
      OTOO::xxx[c].z = xi.z;
      OTOO::xxx[c].w = (double)iw;

      PS::F64vec vi = epi[iw][i].v;
      OTOO::vvv[c].x = vi.x;
      OTOO::vvv[c].y = vi.y;
      OTOO::vvv[c].z = vi.z;
      OTOO::vvv[c].w = epi[iw][i].h;

      OTOO::rho[c].x = epi[iw][i].dn;
      OTOO::rho[c].y = epi[iw][i].cs;
      OTOO::rho[c].z = epi[iw][i].vv;
      OTOO::rho[c].w = epi[iw][i].pp;

      OTOO::alp[c]    = epi[iw][i].al;
      c++;
    }
  }
  int ni = c;

  assert( ni           < OTOO::nimax );
  assert( ni+n_epj_tot < OTOO::njmax );
  CheckNNMAX(n_walk, n_epj, NULL);
  
  for(int j = 0; j < n_epj_tot; j++){
    PS::F64vec xj = epj[j].getPos();
    OTOO::xxx[c].x = xj.x;
    OTOO::xxx[c].y = xj.y;
    OTOO::xxx[c].z = xj.z;
    OTOO::xxx[c].w = epj[j].m;

    PS::F64vec vi = epj[j].v;
    OTOO::vvv[c].x = vi.x;
    OTOO::vvv[c].y = vi.y;
    OTOO::vvv[c].z = vi.z;
    OTOO::vvv[c].w = epj[j].h;
    
    OTOO::rho[c].x = epj[j].dn;
    OTOO::rho[c].y = epj[j].cs;
    OTOO::rho[c].z = epj[j].vv;
    OTOO::rho[c].w = epj[j].pp;

    OTOO::alp[c]    = epj[j].al;
    c++;
  }
  int nall = c;

  c = 0;  
  for(int iw = 0; iw < n_walk; iw++){
    OTOO::jjj[iw].x = c;

    int offset = ni;
    for(int j = 0; j < n_epj[iw]; j++){
      int jj = id_epj[iw][j] + offset;
      OTOO::idx[c] = jj;
      c++;
    }

    OTOO::jjj[iw].y = c;
  }
  int nidx = c;
  
  cl_bool flag = CL_FALSE; 
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_xi[0],  flag, 0, nall*sizeof(cl_double4),   OTOO::xxx, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_vel[0], flag, 0, nall*sizeof(cl_double4),   OTOO::vvv, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_rho[0], flag, 0, nall*sizeof(cl_double4),   OTOO::rho, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_alp[0], flag, 0, nall*sizeof(cl_double),    OTOO::alp, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_jj[0],  flag, 0, n_walk*sizeof(cl_int2), OTOO::jjj, 0, NULL, NULL);

  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_idx[0], flag, 0, nidx*sizeof(cl_int),       OTOO::idx, 0, NULL, NULL);
  
  OTOO::globalThreads[0] = ni;
  cl_int status = CL_SUCCESS;
  status = clEnqueueNDRangeKernel(OTOO::ov[0]->q, OTOO::ker_sph2[0],
				  1, NULL, OTOO::globalThreads, NULL, 0, NULL,
				  &OTOO::ker_event[0]);
  assert(status == CL_SUCCESS);

  return 0;
}

PS::S32 RetrieveKernelSPH2(const PS::S32 tag,
                       const PS::S32 n_walk,
                       const PS::S32 ni[],
                       SPH2 *force[])
{
  uint64 c = 0;
  if (n_walk == 0) return 0;

  c = OTOO::globalThreads[0];
  clFinish(OTOO::ov[0]->q);

  {
    cl_ulong st, en;
    clGetEventProfilingInfo (OTOO::ker_event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &st, NULL);
    clGetEventProfilingInfo (OTOO::ker_event[0], CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &en, NULL);
    OTOO::kernel_time_sph2 += (en - st)*1.0e-9;
  }
  
  cl_bool flag = CL_TRUE;
  clEnqueueReadBuffer(OTOO::ov[0]->q, OTOO::b_acc[0], flag, 0, c*sizeof(cl_double4), OTOO::ggg, 0, NULL, NULL);

  c = 0;
  for(int iw=0; iw<n_walk; iw++){
    for(int i=0; i<ni[iw]; i++){
      force[iw][i].acc.x = OTOO::ggg[c].x;
      force[iw][i].acc.y = OTOO::ggg[c].y;
      force[iw][i].acc.z = OTOO::ggg[c].z;
      force[iw][i].de    = OTOO::ggg[c].w;
      c++;
    }
  }

  OTOO::kernel_count++;
  return 0;
}
