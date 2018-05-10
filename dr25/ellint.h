#ifndef _DR25_ELLINT_H_
#define _DR25_ELLINT_H_

#include <cmath>

namespace batman {

  using std::abs;

#define ELLINT_CONV_TOL 1.0e-8
#define ELLINT_MAX_ITER 200

  // K: 1.0 - k^2 >= 0.0
  template <typename T>
  T ellint_1 (const T& k) {
    T kc = sqrt(1.0 - k * k), m = T(1.0), h;
    for (int i = 0; i < ELLINT_MAX_ITER; ++i) {
      h = m;
      m += kc;
      if (abs(h - kc) / h <= ELLINT_CONV_TOL) break;
      kc = sqrt(h * kc);
      m *= 0.5;
    }
    return M_PI / m;
  }

  // E: 1.0 - k^2 >= 0.0
  template <typename T>
  T ellint_2 (const T& k) {
    T b = 1.0 - k * k, kc = sqrt(b), m = T(1.0), c = T(1.0), a = b + 1.0, m0;
    for (int i = 0; i < ELLINT_MAX_ITER; ++i) {
      b = 2.0 * (c * kc + b);
      c = a;
      m0 = m;
      m += kc;
      a += b / m;
      if (abs(m0 - kc) / m0 <= ELLINT_CONV_TOL) break;
      kc = 2.0 * sqrt(kc * m0);
    }
    return M_PI_4 * a / m;
  }

  // Pi: 1.0 - k^2 >= 0.0 & 0.0 <= n < 1.0 (doesn't seem consistent for n < 0.0)
  template <typename T>
  T ellint_3 (const T& n, const T& k) {
    T kc = sqrt(1.0 - k * k), p = sqrt(1.0 - n), m0 = 1.0, c = 1.0, d = 1.0 / p, e = kc, f, g;
    for (int i = 0; i < ELLINT_MAX_ITER; ++i) {
      f = c;
      c += d / p;
      g = e / p;
      d = 2.0 * (f * g + d);
      p = g + p;
      g = m0;
      m0 = kc + m0;
      if (abs(1.0 - kc / g) <= ELLINT_CONV_TOL) break;
      kc = 2.0 * sqrt(e);
      e = kc * m0;
    }
    return M_PI_2 * (c * m0 + d) / (m0 * (m0 + p));
  }

#undef ELLINT_CONV_TOL
#undef ELLINT_MAX_ITER

}

#endif
