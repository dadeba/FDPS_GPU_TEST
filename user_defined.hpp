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

#define KERNEL_FACTOR M_1_PI
  
  void copyFromForce(const G& f) {
    this->acc_grav = f.acc;
    this->pot      = f.pot;
  }

  void copyFromForce(const SPH1& f){
    this->dn = f.dn*KERNEL_FACTOR;
    this->dd = f.dd*KERNEL_FACTOR;
    this->nn = f.nn;
    this->div = -this->dd/this->dn;
    //    this->rot = sqrt(f.rot*f.rot)*KERNEL_FACTOR/this->dn;
    this->rot = f.rot.x*KERNEL_FACTOR/this->dn;
  }

  void copyFromForce(const SPH2& f){
    this->acc_hydro = f.acc*KERNEL_FACTOR;
    this->de        = f.de*KERNEL_FACTOR;
  }
  
  PS::F64 getCharge() const{
    return this->m;
  }
  
  PS::F64vec getPos() const{
    return this->x;
  }

  PS::F64 getRSearch() const{
    return this->h;
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
    return 2.1 * this->h;
  }
  void setPos(const PS::F64vec& pos){
    this->x = pos;
  }
};

/* Interaction functions */
template <class TParticleJ>
void CalcGravity (const EP_grav * ep_i,
                  const PS::S32 n_ip,
                  const TParticleJ * ep_j,
                  const PS::S32 n_jp,
                  G * force) {
  const PS::F64 eps2 = OTOO::eps*OTOO::eps;
  for(PS::S64 i = 0; i < n_ip; i++){
    PS::F64vec xi = ep_i[i].getPos();
    PS::F64vec ai = 0.0;
    PS::F64 poti = 0.0;
    for(PS::S32 j = 0; j < n_jp; j++){
      PS::F64vec rij    = xi - ep_j[j].getPos();
      PS::F64    r3_inv = rij * rij + eps2;
      PS::F64    r_inv  = 1.0/sqrt(r3_inv);
      r3_inv  = r_inv * r_inv;
      r_inv  *= ep_j[j].getCharge();
      r3_inv *= r_inv;
      ai     -= r3_inv * rij;
      poti   -= r_inv;
    }
    force[i].acc += ai;
    force[i].pot += poti;
  }
}

static double sph_kernel(double q)
{
  double dum, dum1, dum2;

  dum1 = 2.0 - q;
  dum1= 0.25f*dum1*dum1*dum1;

  dum2 = q*q;
  dum2 = 1.0 - 1.5*dum2 + 0.75*dum2*q;

  dum = (q >= 2.0) ? 0.0f : dum1;
  dum = (q >= 1.0) ? dum  : dum2;

  return dum;
}

static double my_recip(double q) {
  return 1.0/q;
}

static double sph_dkernel(double q)
{
  double dum;
  double dum1, dum2, dum3;
  double q_i;

  q_i = 1.0;
  if (q != 0.0) q_i = my_recip(q);
  
  dum1 = 2.0 - q;
  dum1 = -0.75*dum1*dum1*q_i;

  dum2 = -0.75*(4.0-3.0*q);

  dum3 = -q_i;

  dum = (q > 2.0 || q == 0.0f) ? 0.0f : dum1;
  dum = (q > 1.0)              ? dum  : dum2;
  dum = (q > 2.0/3.0)          ? dum  : dum3;

  return dum;
}

static double aValue(double s)
{
  if (s < 1.0) {
    return 0.2*(1.0+s*s);
  } else {
    return 0.2*(1.0+1.0/(s*s*s));
  }
}

class CalcSPH1 {
public:
  void operator () (const EP_hydro * ep_i,
		    const PS::S32 n_ip,
		    const EP_hydro * ep_j,
		    const PS::S32 n_jp,
		    SPH1 * force)
  {
    for (uint64 i = 0; i < n_ip ; i++){
      const double hi = ep_i[i].h;

      double den  = 0.0;
      double rotx = 0.0;
      double roty = 0.0;
      double rotz = 0.0;
      double dd   = 0.0;
      double nn   = 0.0;
      
      // Compute density
      for (uint64 j = 0; j < n_jp; j++){
	const PS::F64vec dx = ep_i[i].x - ep_j[j].x;
	const PS::F64vec dv = ep_i[i].v - ep_j[j].v;
	const PS::F64    r2 = dx*dx;

	double hj = ep_j[j].h;
	double mj = ep_j[j].m;
	
	double q, h1_i, ker, dker, dum1, dum2, xv;
	double h2_i, h3_i, h5_i;

	h1_i = 2.0/(hi + hj);
	q = sqrt(r2)*h1_i;

	h2_i = h1_i*h1_i;
	h3_i = h2_i*h1_i;
	h5_i = h2_i*h3_i;
	ker  = sph_kernel(q)*h3_i;
	dker = sph_dkernel(q)*h5_i;
  
	// density
	den += mj*ker;
  
	// roration
	dum1 = mj*dker;
	rotx += (dv.y*dx.z - dv.z*dx.y)*dum1;
	roty += (dv.z*dx.x - dv.x*dx.z)*dum1;
	rotz += (dv.x*dx.y - dv.y*dx.x)*dum1;
  
	// time derivative of density
	xv = dx.x*dv.x + dx.y*dv.y + dx.z*dv.z;
	dum2 = xv*dum1;
	dd += dum2;

	// NN
	if (q < 1.5) {
	  nn += 1.0;
	} else {
	  nn += sph_kernel(4.0*(q-1.5));
	}
      }
      
      force[i].dn    = den;
      force[i].rot.x = rotx;
      force[i].rot.y = roty;
      force[i].rot.z = rotz;
      force[i].dd    = dd;
      force[i].nn    = nn;
    }
  }
};

class CalcSPH2 {
public:
  void operator () (const EP_hydro * ep_i,
		    const PS::S32 n_ip,
		    const EP_hydro * ep_j,
		    const PS::S32 n_jp,
		    SPH2 * force)
  {
    for (uint64 i = 0; i < n_ip ; i++){
      const double hi = ep_i[i].h;

      double ax = 0.0;
      double ay = 0.0;
      double az = 0.0;
      double de = 0.0;
      
      for (uint64 j = 0; j < n_jp; j++){
	const PS::F64vec dx = ep_i[i].x - ep_j[j].x;
	const PS::F64vec dv = ep_i[i].v - ep_j[j].v;
	const PS::F64    r2 = dx*dx;

	double hj = ep_j[j].h;
	double mj = ep_j[j].m;

	double q, h1, h1_i, dker, eta2, dum1, dum2, dum3, xv, vis;
	double h2_i, h3_i, h5_i;
	double alph, beta;
	
	h1 = (hi + hj)/2.0;
	h1_i = 1.0/h1;
	q = sqrt(r2)*h1_i;

	h2_i = h1_i*h1_i;
	h3_i = h2_i*h1_i;
	h5_i = h2_i*h3_i;
	dker = sph_dkernel(q)*h5_i;

	xv = dx.x*dv.x + dx.y*dv.y + dx.z*dv.z;
	if (xv < 0.0) {
	  dum1 = 0.5*(ep_i[i].dn + ep_j[j].dn); // rho
	  dum2 = 0.5*(ep_i[i].cs + ep_j[j].cs); // cs
    
	  eta2 = 0.01*h1*h1;
	  dum3 = (h1*xv)/(r2 + eta2);

	  alph = std::max(ep_i[i].al, ep_j[j].al);
	  beta = 2.0*alph;

	  vis = (-alph*dum2*dum3 + beta*dum3*dum3)/dum1;
	  vis = 0.5*(ep_i[i].vv + ep_j[j].vv)*vis; // vis
	} else {
	  vis = 0.0;
	}

	dum1 = mj*dker;
	dum2 = (ep_i[i].pp + ep_j[j].pp + vis)*dum1; // pp
	dum3 = xv*dum1;
  
	// accelaration
	ax += -dx.x*dum2;
	ay += -dx.y*dum2;
	az += -dx.z*dum2;
  
	// time derivative of internal energy
	de += (ep_i[i].pp + 0.5*vis)*dum3;
      }
      
      force[i].acc.x = ax;
      force[i].acc.y = ay;
      force[i].acc.z = az;
      force[i].de    = de;
    }
  }
};

PS::S32 DispatchKernel(
		       const PS::S32          tag,
		       const PS::S32          n_walk,
		       const EP_grav          *epi[],
		       const PS::S32          n_epi[],
		       const EP_grav          *epj[],
		       const PS::S32          n_epj[],
		       const PS::SPJMonopole  *spj[],
		       const PS::S32           n_spj[])
{
  const PS::F64 eps2 = OTOO::eps*OTOO::eps;
  uint64 c = 0;

  for(int iw = 0; iw < n_walk; iw++){
    for(int i = 0; i < n_epi[iw]; i++){
      PS::F64vec xi = epi[iw][i].getPos();
      OTOO::xxx[c][0] = xi.x;
      OTOO::xxx[c][1] = xi.y;
      OTOO::xxx[c][2] = xi.z;
      OTOO::xxx[c][3] = (double)iw;
      c++;
    }
  }
  int ni = c;
  
  for(int iw = 0; iw < n_walk; iw++){
    OTOO::jjj[iw][0] = c;

    for(int j = 0; j < n_epj[iw]; j++){
      PS::F64vec xj = epj[iw][j].getPos();
      OTOO::xxx[c][0] = xj.x;
      OTOO::xxx[c][1] = xj.y;
      OTOO::xxx[c][2] = xj.z;
      OTOO::xxx[c][3] = epj[iw][j].getCharge();
      c++;
    }
    for(int j = 0; j < n_spj[iw]; j++){
      PS::F64vec xj = spj[iw][j].getPos();
      OTOO::xxx[c][0] = xj.x;
      OTOO::xxx[c][1] = xj.y;
      OTOO::xxx[c][2] = xj.z;
      OTOO::xxx[c][3] = spj[iw][j].getCharge();
      c++;
    }

    OTOO::jjj[iw][1] = c;
  }

  clSetKernelArg(OTOO::ker[0], 3, sizeof(double), (void *)&eps2);

  cl_bool flag = CL_FALSE; 
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_xi[0], flag, 0, c*sizeof(cl_double4),   OTOO::xxx, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_jj[0], flag, 0, n_walk*sizeof(cl_int2), OTOO::jjj, 0, NULL, NULL);

  OTOO::globalThreads[0] = ni;
  cl_int status = CL_SUCCESS;
  status = clEnqueueNDRangeKernel(OTOO::ov[0]->q, OTOO::ker[0], 1, NULL, OTOO::globalThreads, NULL, 0, NULL,
				  &OTOO::ker_event[0]);
  assert(status == CL_SUCCESS);

  return 0;
}

PS::S32 RetrieveKernel(const PS::S32 tag,
                       const PS::S32 n_walk,
                       const PS::S32 ni[],
                       G *force[])
{
  uint64 c = 0;
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
      force[iw][i].acc.x = OTOO::ggg[c][0];
      force[iw][i].acc.y = OTOO::ggg[c][1];
      force[iw][i].acc.z = OTOO::ggg[c][2];
      force[iw][i].pot   = OTOO::ggg[c][3];
      c++;
    }
  }

  return 0;
}

PS::S32 DispatchKernelSPH1(
			   const PS::S32          tag,
			   const PS::S32          n_walk,
			   const EP_hydro         *epi[],
			   const PS::S32          n_epi[],
			   const EP_hydro         *epj[],
			   const PS::S32          n_epj[])
{
  uint64 c = 0;

  for(int iw = 0; iw < n_walk; iw++){
    for(int i = 0; i < n_epi[iw]; i++){
      PS::F64vec xi = epi[iw][i].getPos();
      OTOO::xxx[c][0] = xi.x;
      OTOO::xxx[c][1] = xi.y;
      OTOO::xxx[c][2] = xi.z;
      OTOO::xxx[c][3] = (double)iw;

      PS::F64vec vi = epi[iw][i].v;
      OTOO::vvv[c][0] = vi.x;
      OTOO::vvv[c][1] = vi.y;
      OTOO::vvv[c][2] = vi.z;
      OTOO::vvv[c][3] = epi[iw][i].h;
      OTOO::mas[c]    = epi[iw][i].m;
      c++;
    }
  }
  int ni = c;
  
  for(int iw = 0; iw < n_walk; iw++){
    OTOO::jjj[iw][0] = c;

    for(int j = 0; j < n_epj[iw]; j++){
      PS::F64vec xj = epj[iw][j].getPos();
      OTOO::xxx[c][0] = xj.x;
      OTOO::xxx[c][1] = xj.y;
      OTOO::xxx[c][2] = xj.z;
      OTOO::xxx[c][3] = epj[iw][j].m;

      PS::F64vec vi = epj[iw][j].v;
      OTOO::vvv[c][0] = vi.x;
      OTOO::vvv[c][1] = vi.y;
      OTOO::vvv[c][2] = vi.z;
      OTOO::vvv[c][3] = epj[iw][j].h;
      OTOO::mas[c]    = epj[iw][j].m;
      c++;
    }

    OTOO::jjj[iw][1] = c;
  }

  cl_bool flag = CL_FALSE; 
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_xi[0],  flag, 0, c*sizeof(cl_double4),   OTOO::xxx, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_vel[0], flag, 0, c*sizeof(cl_double4),   OTOO::vvv, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_mas[0], flag, 0, c*sizeof(cl_double),    OTOO::mas, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_jj[0],  flag, 0, n_walk*sizeof(cl_int2), OTOO::jjj, 0, NULL, NULL);

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
      force[iw][i].dn    = OTOO::ggg[c][3];
      force[iw][i].rot.x = OTOO::ggg[c][0];
      force[iw][i].dd    = OTOO::ggg[c][1];
      force[iw][i].nn    = OTOO::ggg[c][2];
      c++;
    }
  }

  OTOO::kernel_count++;
  return 0;
}

PS::S32 DispatchKernelSPH2(const PS::S32          tag,
			   const PS::S32          n_walk,
			   const EP_hydro         *epi[],
			   const PS::S32          n_epi[],
			   const EP_hydro         *epj[],
			   const PS::S32          n_epj[])
{
  uint64 c = 0;

  for(int iw = 0; iw < n_walk; iw++){
    for(int i = 0; i < n_epi[iw]; i++){
      PS::F64vec xi = epi[iw][i].getPos();
      OTOO::xxx[c][0] = xi.x;
      OTOO::xxx[c][1] = xi.y;
      OTOO::xxx[c][2] = xi.z;
      OTOO::xxx[c][3] = (double)iw;

      PS::F64vec vi = epi[iw][i].v;
      OTOO::vvv[c][0] = vi.x;
      OTOO::vvv[c][1] = vi.y;
      OTOO::vvv[c][2] = vi.z;
      OTOO::vvv[c][3] = epi[iw][i].h;

      OTOO::rho[c][0] = epi[iw][i].dn;
      OTOO::rho[c][1] = epi[iw][i].cs;
      OTOO::rho[c][2] = epi[iw][i].vv;
      OTOO::rho[c][3] = epi[iw][i].pp;

      OTOO::mas[c]    = epi[iw][i].m;
      OTOO::alp[c]    = epi[iw][i].al;
      c++;
    }
  }
  int ni = c;
  
  for(int iw = 0; iw < n_walk; iw++){
    OTOO::jjj[iw][0] = c;

    for(int j = 0; j < n_epj[iw]; j++){
      PS::F64vec xj = epj[iw][j].getPos();
      OTOO::xxx[c][0] = xj.x;
      OTOO::xxx[c][1] = xj.y;
      OTOO::xxx[c][2] = xj.z;
      OTOO::xxx[c][3] = epj[iw][j].m;

      PS::F64vec vi = epj[iw][j].v;
      OTOO::vvv[c][0] = vi.x;
      OTOO::vvv[c][1] = vi.y;
      OTOO::vvv[c][2] = vi.z;
      OTOO::vvv[c][3] = epj[iw][j].h;

      OTOO::rho[c][0] = epj[iw][j].dn;
      OTOO::rho[c][1] = epj[iw][j].cs;
      OTOO::rho[c][2] = epj[iw][j].vv;
      OTOO::rho[c][3] = epj[iw][j].pp;

      OTOO::mas[c]    = epj[iw][j].m;
      OTOO::alp[c]    = epj[iw][j].al;

      OTOO::mas[c]    = epj[iw][j].m;
      c++;
    }

    OTOO::jjj[iw][1] = c;
  }

  cl_bool flag = CL_FALSE; 
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_xi[0],  flag, 0, c*sizeof(cl_double4),   OTOO::xxx, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_vel[0], flag, 0, c*sizeof(cl_double4),   OTOO::vvv, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_rho[0], flag, 0, c*sizeof(cl_double4),   OTOO::rho, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_mas[0], flag, 0, c*sizeof(cl_double),    OTOO::mas, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_alp[0], flag, 0, c*sizeof(cl_double),    OTOO::alp, 0, NULL, NULL);
  clEnqueueWriteBuffer(OTOO::ov[0]->q, OTOO::b_jj[0],  flag, 0, n_walk*sizeof(cl_int2), OTOO::jjj, 0, NULL, NULL);

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
  c = OTOO::globalThreads[0];
  clFinish(OTOO::ov[0]->q);
  
  cl_bool flag = CL_TRUE;
  clEnqueueReadBuffer(OTOO::ov[0]->q, OTOO::b_acc[0], flag, 0, c*sizeof(cl_double4), OTOO::ggg, 0, NULL, NULL);

  c = 0;
  for(int iw=0; iw<n_walk; iw++){
    for(int i=0; i<ni[iw]; i++){
      force[iw][i].acc.x = OTOO::ggg[c][0];
      force[iw][i].acc.y = OTOO::ggg[c][1];
      force[iw][i].acc.z = OTOO::ggg[c][2];
      force[iw][i].de    = OTOO::ggg[c][3];
      c++;
    }
  }
  return 0;
}
