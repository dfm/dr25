#ifndef _DR25_ELLINT_GRAD_H_
#define _DR25_ELLINT_GRAD_H_

#include <cmath>
#include <Eigen/Core>
#include <AutoDiffScalar.h>

namespace batman {

  using std::abs;

  // Gradients.
  template <typename T>
  Eigen::AutoDiffScalar<T> ellint_1 (const Eigen::AutoDiffScalar<T>& z)
  {
    typename T::Scalar value = z.value(),
              Kz = ellint_1(value),
              Ez = ellint_2(value),
              z2 = value * value;
    return Eigen::AutoDiffScalar<T>(
      Kz,
      z.derivatives() * (Ez / (1.0 - z2) - Kz) / value
    );
  }

  template <typename T>
  Eigen::AutoDiffScalar<T> ellint_2 (const Eigen::AutoDiffScalar<T>& z)
  {
    typename T::Scalar value = z.value(),
              Kz = ellint_1(value),
              Ez = ellint_2(value);
    return Eigen::AutoDiffScalar<T>(
      Ez,
      z.derivatives() * (Ez - Kz) / value
    );
  }

  template <typename T>
  Eigen::AutoDiffScalar<T> ellint_3 (const Eigen::AutoDiffScalar<T>& n,
                                     const Eigen::AutoDiffScalar<T>& k)
  {
    typename T::Scalar k_value = k.value(),
                       n_value = n.value(),
                       Kk = ellint_1(k_value),
                       Ek = ellint_2(k_value),
                       Pnk = ellint_3(n_value, k_value),
                       k2 = k_value * k_value,
                       n2 = n_value * n_value;
    return Eigen::AutoDiffScalar<T>(
      Pnk,
      (n.derivatives() * 0.5*(Ek + (Kk*(k2-n_value) + Pnk*(n2-k2))/n_value) / (n_value-1.0) -
      k.derivatives() * k_value * (Ek / (k2 - 1.0) + Pnk)) / (k2-n_value)
    );
  }

}

#endif
