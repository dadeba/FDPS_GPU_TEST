#ifndef OTOO_EOS_H
#define OTOO_EOS_H

namespace OTOO {
  class EOS {
  public:
    EOS() {};
    ~EOS() {};

    virtual double GetS(double d, double e) = 0;
    virtual double GetP(double d, double e) = 0;
    virtual double GetE(double d, double e) = 0;
    virtual double GetT(double d, double e) = 0;
    virtual void TestEOS(uint64 n, double *d, double *e= NULL) = 0;
    virtual double GetEmin(double d) = 0; 
  };
}
#endif
