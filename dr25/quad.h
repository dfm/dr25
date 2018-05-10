#ifndef _DR25_QUAD_H_
#define _DR25_QUAD_H_

#include <cmath>
#include <limits>
#include <algorithm>
#include "ellint.h"

namespace batman {

  using std::abs;
  using std::max;
  using std::min;

  template <typename T>
  T quad (const T& c1, const T& c2, const T& p, const T& d0) {
    const T omega = 1.0 - c1/3.0 - c2/6.0;
    const T tol = std::numeric_limits<T>::epsilon();

    T kap0 = T(0.0), kap1 = T(0.0);
    T lambdad = T(0.0), lambdae = T(0.0), etad = T(0.0);

    // allow for negative impact parameters
    T d = abs(d0);

    // check the corner cases
    if (abs(p - d) < tol) d = p;
    if (abs(p - 1.0 - d) < tol) d = p - 1.0;
    if (abs(1.0 - p - d) < tol) d = 1.0 - p;
    if (d < tol) d = T(0.0);

    //source is unocculted:
    if (d >= 1.0 + p) return T(1.0);

    //source is completely occulted:
    if (p >= 1.0 && d <= p - 1.0) {
      lambdad = T(0.0);
      etad = T(0.5);        //error in Fortran code corrected here, following Jason Eastman's python code
      lambdae = T(1.0);
      return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*(lambdad + 2.0/3.0) + c2*etad)/omega;
    }

    T x1 = pow((p - d), 2.0);
    T x2 = pow((p + d), 2.0);
    T x3 = p*p - d*d;

    //source is partly occulted and occulting object crosses the limb:
    if (d >= abs(1.0 - p) && d <= 1.0 + p) {
      kap1 = acos(min((1.0 - p*p + d*d)/2.0/d, 1.0));
      kap0 = acos(min((p*p + d*d - 1.0)/2.0/p/d, 1.0));
      lambdae = p*p*kap0 + kap1;
      lambdae = (lambdae - 0.5*sqrt(max(4.0*d*d - pow((1.0 + d*d - p*p), 2.0), 0.0)))/M_PI;
    }

    //edge of the occulting star lies at the origin
    if(d == p) {
      if(d < 0.5) {
        T q = 2.0*p;
        T Kk = ellint_1(q);
        T Ek = ellint_2(q);
        lambdad = 1.0/3.0 + 2.0/9.0/M_PI*(4.0*(2.0*p*p - 1.0)*Ek + (1.0 - 4.0*p*p)*Kk);
        etad = p*p/2.0*(p*p + 2.0*d*d);
        return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega;
      } else if(d > 0.5) {
        T q = 0.5/p;
        T Kk = ellint_1(q);
        T Ek = ellint_2(q);
        lambdad = 1.0/3.0 + 16.0*p/9.0/M_PI*(2.0*p*p - 1.0)*Ek -  \
                  (32.0*pow(p, 4.0) - 20.0*p*p + 3.0)/9.0/M_PI/p*Kk;
        etad = 1.0/2.0/M_PI*(kap1 + p*p*(p*p + 2.0*d*d)*kap0 -  \
            (1.0 + 5.0*p*p + d*d)/4.0*sqrt((1.0 - x1)*(x2 - 1.0)));
      } else {
        lambdad = T(1.0/3.0 - 4.0/M_PI/9.0);
        etad = T(3.0/32.0);
        return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega;
      }

      return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega;
    }

    //occulting star partly occults the source and crosses the limb:
    //if((d > 0.5 + abs(p  - 0.5) && d < 1.0 + p) || (p > 0.5 && d > abs(1.0 - p)*1.0001 \
    //&& d < p))  //the factor of 1.0001 is from the Mandel/Agol Fortran routine, but gave bad output for d near abs(1-p)
    if ((d > 0.5 + abs(p  - 0.5) && d < 1.0 + p) || (p > 0.5 && d > abs(1.0 - p) && d < p)) {
      T q = sqrt((1.0 - x1)/4.0/d/p);
      T Kk = ellint_1(q);
      T Ek = ellint_2(q);
      T n = 1.0/x1 - 1.0;
      T Pk = ellint_3(T(-n), q);
      lambdad = 1.0/9.0/M_PI/sqrt(p*d)*(((1.0 - x2)*(2.0*x2 + x1 - 3.0) - 3.0*x3*(x2 - 2.0))*Kk + 4.0*p*d*(d*d + 7.0*p*p - 4.0)*Ek - 3.0*x3/x1*Pk);
      if(d < p) lambdad += T(2.0/3.0);
      etad = 1.0/2.0/M_PI*(kap1 + p*p*(p*p + 2.0*d*d)*kap0 - (1.0 + 5.0*p*p + d*d)/4.0*sqrt((1.0 - x1)*(x2 - 1.0)));
      return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega;
    }

    //occulting star transits the source:
    if (p <= 1.0  && d <= (1.0 - p)) {
      etad = p*p/2.0*(p*p + 2.0*d*d);
      lambdae = p*p;

      T q = sqrt((x2 - x1)/(1.0 - x1));
      T Kk = ellint_1(q);
      T Ek = ellint_2(q);
      T n = x2/x1 - 1.0;
      T Pk = ellint_3(T(-n), q);

      lambdad = 2.0/9.0/M_PI/sqrt(1.0 - x1)*((1.0 - 5.0*d*d + p*p + x3*x3)*Kk + (1.0 - x1)*(d*d + 7.0*p*p - 4.0)*Ek - 3.0*x3/x1*Pk);

      // edge of planet hits edge of star
      if(abs(p + d - 1.0) <= tol) {
        lambdad = 2.0/3.0/M_PI*acos(1.0 - 2.0*p) - 4.0/9.0/M_PI*sqrt(p*(1.0 - p))*(3.0 + 2.0*p - 8.0*p*p);
      }
      if(d < p) lambdad += T(2.0/3.0);
    }

    return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega;
  }

}

#endif
