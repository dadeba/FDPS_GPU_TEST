#define READONLY_P const * restrict 

__kernel
void 
grav(
     __global double4 READONLY_P xi_global,
     __global int2 READONLY_P jj_global, 
     __global double4 *acc,
     const double e2
     )
{
  unsigned int g_xid = get_global_id(0);
  unsigned int g_yid = get_global_id(1);
  unsigned int g_w   = get_global_size(0);
  unsigned int gid   = g_yid*g_w + g_xid;
  unsigned int i = gid;

  double4 xi = xi_global[i];

  int iw = (int)xi.w;
  int2 jj = jj_global[iw];
  int j0 = jj.x;
  int j1 = jj.y;

  double a_x, a_y, a_z, p_t;
  a_x = a_y = a_z = p_t = 0.0;

  for(int j = j0; j < j1; j++) {
    double4 xj = xi_global[j];
    double  mj = xj.w;
    
    double dx, dy, dz;
    dx = xj.x - xi.x;
    dy = xj.y - xi.y;
    dz = xj.z - xi.z;

    double r2 = dx*dx + dy*dy + dz*dz + e2;
    double r1i = native_rsqrt(r2);
    double r2i = r1i*r1i;
    double r1im = mj*r1i;
    double r3im = r1im*r2i;

    a_x += dx*r3im;
    a_y += dy*r3im;
    a_z += dz*r3im;
    p_t += -r1im;
  }
  
  acc[i].x = a_x;
  acc[i].y = a_y;
  acc[i].z = a_z;
  acc[i].w = p_t;
}

double sph_kernel(double q)
{
  double dum, dum1, dum2;

  dum1 = 2.0 - q;
  dum1= 0.25f*dum1*dum1*dum1;

  dum2 = q*q;
  dum2 = 1.0 - 1.5*dum2 + 0.75*dum2*q;

  dum = (q >= 2.0) ? 0.0 : dum1;
  dum = (q >= 1.0) ? dum : dum2;

  return dum;
}

double sph_dkernel(double q)
{
  double dum;
  double dum1, dum2, dum3;
  double q_i;

  q_i = 1.0;
  if (q != 0.0) q_i = native_recip(q);
  
  dum1 = 2.0 - q;
  dum1 = -0.75*dum1*dum1*q_i;

  dum2 = -0.75*(4.0-3.0*q);

  dum3 = -q_i;

  dum = (q > 2.0 || q == 0.0f) ? 0.0 : dum1;
  dum = (q > 1.0)              ? dum : dum2;
  dum = (q > 2.0/3.0)          ? dum : dum3;

  return dum;
}

/*
__kernel
void 
ker_nn(
       __global double4 READONLY_P xi_global,
       __global double  READONLY_P hh_global,
       __global int2 READONLY_P jj_global, 
       __global double *nn,
       )
{
  unsigned int g_xid = get_global_id(0);
  unsigned int g_yid = get_global_id(1);
  unsigned int g_w   = get_global_size(0);
  unsigned int gid   = g_yid*g_w + g_xid;
  unsigned int i = gid;

  double4 xi = xi_global[i];
  double  hi = hh_global[i];
  
  int iw = (int)xi.w;
  int2 jj = jj_global[iw];
  int j0 = jj.x;
  int j1 = jj.y;

  double nn0 = 0.0;
  for(int j = j0; j < j1; j++) {
    double4 xj = xi_global[j];
    double  hj = hh_global[j];

    h1_i = 2.0/(hi + hj);
    q = native_sqrt(r2)*h1_i;

    if (q < 1.5) {
      nn0 += 1.0;
    } else {
      nn0 += sph_kernel(4.0*(q-1.5));
    }
  }

  nn[i] = nn0;
}
*/

double R2(double4 p)
{
  return p.x*p.x + p.y*p.y + p.z*p.z;
}

__kernel
void ker_sph1(
	      __global double4 READONLY_P pos_global, // x, y, z, iw
	      __global double4 READONLY_P vel_global, // vx, vy, vz, h
	      __global double  READONLY_P mas_global, // mj
	      __global int2 READONLY_P jj_global, 
	      __global double4 *res
	      )
{
  unsigned int g_xid = get_global_id(0);
  unsigned int g_yid = get_global_id(1);
  unsigned int g_w   = get_global_size(0);
  unsigned int gid   = g_yid*g_w + g_xid;
  unsigned int i = gid;

  double4 xi = pos_global[i]; // x, y, z
  double4 vi = vel_global[i]; // vx, vy, vz, h
  double  hi = vi.w;
  
  int iw = (int)xi.w;
  int2 jj = jj_global[iw];
  int j0 = jj.x;
  int j1 = jj.y;

  // initialize force
  double4 den = (double4)(0.0, 0.0, 0.0, 0.0);
  double2 dd  = (double2)(0.0, 0.0);
  for(int j = j0; j < j1; j++) {
    double4 xj = pos_global[j]; // x, y, z
    double4 vj = vel_global[j]; // vx, vy, vz, h
    double  mj = mas_global[j];
    double  hj = vj.w;
      
    double4 dx = xi - xj;
    double  r2 = R2(dx);
    double4 dv = vi - vj;

    double q, h1_i, ker, dker, dum1, dum2, xv;
    double h2_i, h3_i, h5_i;

    h1_i = 2.0/(hi + hj);
    q = native_sqrt(r2)*h1_i;

    h2_i = h1_i*h1_i;
    h3_i = h2_i*h1_i;
    h5_i = h2_i*h3_i;
    ker  = sph_kernel(q)*h3_i;
    dker = sph_dkernel(q)*h5_i;
  
    // density
    den.w += mj*ker;
  
    // roration
    dum1 = mj*dker;
    den.x += (dv.y*dx.z - dv.z*dx.y)*dum1;
    den.y += (dv.z*dx.x - dv.x*dx.z)*dum1;
    den.z += (dv.x*dx.y - dv.y*dx.x)*dum1;
    
    // time derivative of density
    xv = dx.x*dv.x + dx.y*dv.y + dx.z*dv.z;
    dum2 = xv*dum1;
    dd.x += dum2;

    // NN
    if (q < 1.5) {
      dd.y += 1.0;
    } else {
      dd.y += sph_kernel(4.0*(q-1.5));
    }
  }

  double rot = sqrt(den.x*den.x + den.y*den.y + den.z*den.z);
  den.x = rot;
  den.y = dd.x;
  den.z = dd.y;

  res[i] = den;
}

__kernel
void ker_sph2(
	      __global double4 READONLY_P pos_global,
	      __global double4 READONLY_P vel_global,
	      __global double4 READONLY_P rho_global,
	      __global double  READONLY_P mas_global,
	      __global double  READONLY_P alp_global,
	      __global int2 READONLY_P jj_global, 
	      __global double4 *res
	      )
{
  unsigned int g_xid = get_global_id(0);
  unsigned int g_yid = get_global_id(1);
  unsigned int g_w   = get_global_size(0);
  unsigned int gid   = g_yid*g_w + g_xid;
  unsigned int i = gid;

  double4 xi = pos_global[i]; // x, y, z
  double4 vi = vel_global[i]; // vx, vy, vz, h
  double4 ri = rho_global[i]; // rho, cs, vvi, ppi
  double  hi = vi.w;
  double  alphi = alp_global[i];
    
  int iw = (int)xi.w;
  int2 jj = jj_global[iw];
  int j0 = jj.x;
  int j1 = jj.y;

  // initialize force
  double4 a = (double4)(0.0, 0.0, 0.0, 0.0);

  for(int j = j0; j < j1; j++) {
    double4 xj = pos_global[j]; // x, y, z
    double4 vj = vel_global[j]; // vx, vy, vz, h
    double4 rj = rho_global[j]; // rho, cs, vvi, ppi
    double  mj = mas_global[j];
    double  hj = vj.w;
    double  alphj = alp_global[j];
    
    double4 dx = xi - xj;
    double  r2 = R2(dx);
    double4 dv = vi - vj;

    double q, h1, h1_i, dker, eta2, dum1, dum2, dum3, xv, vis;
    double h2_i, h3_i, h5_i;
    double alph, beta;
  
    double4 a0;

    h1 = (hi + hj)/2.0;
    h1_i = 1.0/h1;
    q = native_sqrt(r2)*h1_i;

    h2_i = h1_i*h1_i;
    h3_i = h2_i*h1_i;
    h5_i = h2_i*h3_i;
    dker = sph_dkernel(q)*h5_i;

    xv = dx.x*dv.x + dx.y*dv.y + dx.z*dv.z;
    if (xv < 0.0) {
      dum1 = 0.5*(ri.x + rj.x); // rho
      dum2 = 0.5*(ri.y + rj.y); // cs
    
      eta2 = 0.01*h1*h1;
      dum3 = (h1*xv)/(r2 + eta2);

      alph = max(alphi, alphj);
      beta = 2.0*alph;

      vis = (-alph*dum2*dum3 + beta*dum3*dum3)/dum1;
      vis = 0.5*(ri.z + rj.z)*vis; // vis
    } else {
      vis = 0.0;
    }

    dum1 = mj*dker;
    dum2 = (ri.w + rj.w + vis)*dum1; // pp
    dum3 = xv*dum1;
  
    // accelaration
    a.x += -dx.x*dum2;
    a.y += -dx.y*dum2;
    a.z += -dx.z*dum2;
  
    // time derivative of internal energy
    a.w += (ri.w + 0.5*vis)*dum3;
  }

  res[i] = a;
}


