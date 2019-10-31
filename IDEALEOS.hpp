#ifndef OTOO_IDEOS_H
#define OTOO_IDEOS_H
#include "EOS.hpp"

namespace OTOO {
  class IDEOS : public EOS {
  public:
    IDEOS(double);
    ~IDEOS() {};
    double GetS(double d, double e);
    double GetP(double d, double e);
    double GetE(double d, double e);
    double GetT(double d, double e);
    void TestEOS(uint64 n, double *d, double *e = NULL) {};
    double GetEmin(double d) { return 0.0f; };
  protected:
    double gam;
    double gam1;
  };

  IDEOS::IDEOS(double gamgam = 5.0/3.0) : gam(gamgam)
  {
    gam1 = gam - 1.0;
  }

  double IDEOS::GetP(double d, double e)
  {
    return gam1*d*e;
  }

  double IDEOS::GetS(double d, double e)
  {
    return sqrt(gam*gam1*e);
  }

  double IDEOS::GetE(double d, double e)
  {
    // dummy
    return e;
  }

  double IDEOS::GetT(double d, double e)
  {
    // dummy
    return e;
  }
}
#endif
