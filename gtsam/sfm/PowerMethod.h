/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   PowerMethod.h
 * @date   Sept 2020
 * @author Jing Wu
 * @brief  accelerated power method for fast eigenvalue and eigenvector
 * computation
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/dllexport.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace gtsam {

using Sparse = Eigen::SparseMatrix<double>;

/* ************************************************************************* */
/// MINIMUM EIGENVALUE COMPUTATIONS

// Template argument Operator just needs multiplication operator
template <class Operator> struct PowerMethod {
  /**
   * \brief Compute i-th Eigenpair with power method
   *
   * References :
   * 1) Rosen, D. and Carlone, L., 2017, September. Computational
   * enhancements for certifiably correct SLAM. In Proceedings of the
   * International Conference on Intelligent Robots and Systems.
   * 2) Yulun Tian and Kasra Khosoussi and David M. Rosen and Jonathan P. How,
   * 2020, Aug, Distributed Certifiably Correct Pose-Graph Optimization, Arxiv
   * 3) C. de Sa, B. He, I. Mitliagkas, C. Ré, and P. Xu, “Accelerated
   * stochastic power iteration,” in Proc. Mach. Learn. Res., no. 84, 2018, pp.
   * 58–67
   *
   * It performs the following iteration: \f$ x_{k+1} = A * x_k + \beta *
   * x_{k-1} \f$ where A is the certificate matrix, x is the Ritz vector
   *
   */

  // Const reference to an externally-held matrix whose minimum-eigenvalue we
  // want to compute
  const Operator &A_;

  const int dim_;        // dimension of Matrix A
  const int nrRequired_; // number of eigenvalues required

  // flag for running power method or accelerated power method. If false, the
  // former, vice versa.
  bool accelerated_;

  // a Polyak momentum term
  double beta_;

  // const int m_ncv_; // dimention of orthonormal basis subspace
  size_t nrIterations_; // number of iterations

private:
  Vector ritzValues_;       // all Ritz eigenvalues
  Matrix ritzVectors_;      // all Ritz eigenvectors
  Vector ritzConverged_;    // store whether the Ritz eigenpair converged
  Vector sortedRitzValues_; // sorted converged eigenvalue
  Matrix sortedRizVectors_; // sorted converged eigenvectors

public:
  // Constructor
  explicit PowerMethod(const Operator &A, const Matrix &S, int nrRequired = 1,
                       bool accelerated = false, double beta = 0)
      : A_(A), dim_(A.rows()), nrRequired_(nrRequired),
        accelerated_(accelerated), beta_(beta), nrIterations_(0) {
    Vector x0;
    Vector x00;
    if (!S.isZero(0)) {
      x0 = S.row(1);
      x00 = S.row(0);
    } else {
      x0 = Vector::Random(dim_);
      x00 = Vector::Random(dim_);
    }

    // initialize Ritz eigen values
    ritzValues_.resize(dim_);
    ritzValues_.setZero();

    // initialize the Ritz converged vector
    ritzConverged_.resize(dim_);
    ritzConverged_.setZero();

    // initialize Ritz eigen vectors
    ritzVectors_.resize(dim_, nrRequired);
    ritzVectors_.setZero();
    if (accelerated_) {
      ritzVectors_.col(0) = update(x0, x00, beta_);
      ritzVectors_.col(1) = update(ritzVectors_.col(0), x0, beta_);
    } else {
      ritzVectors_.col(0) = update(x0);
      perturb(0);
    }

    // setting beta
    if (accelerated_) {
      Vector init_resid = ritzVectors_.col(0);
      const double up = init_resid.transpose() * A_ * init_resid;
      const double down = init_resid.transpose().dot(init_resid);
      const double mu = up / down;
      beta_ = mu * mu / 4;
      setBeta();
    }
  }

  Vector update(const Vector &x1, const Vector &x0, const double beta) const {
    Vector y = A_ * x1 - beta * x0;
    y.normalize();
    return y;
  }

  Vector update(const Vector &x) const {
    Vector y = A_ * x;
    y.normalize();
    return y;
  }

  Vector update(int i) const {
    if (accelerated_) {
      return update(ritzVectors_.col(i - 1), ritzVectors_.col(i - 2), beta_);
    } else
      return update(ritzVectors_.col(i - 1));
  }

  /// Tuning the momentum beta using the Best Heavy Ball algorithm in Ref(3)
  void setBeta() {
    if (dim_ < 10)
      return;
    double maxBeta = beta_;
    size_t maxIndex;
    std::vector<double> betas = {2 / 3 * maxBeta, 0.99 * maxBeta, maxBeta,
                                 1.01 * maxBeta, 1.5 * maxBeta};

    Matrix ritzVectors;
    ritzVectors.resize(dim_, 10);
    ritzVectors.setZero();
    for (size_t i = 0; i < 10; i++) {
      for (size_t k = 0; k < betas.size(); ++k) {
        for (size_t j = 1; j < 10; j++) {
          if (j < 2) {
            Vector x0 = Vector::Random(dim_);
            Vector x00 = Vector::Random(dim_);
            ritzVectors.col(0) = update(x0, x00, betas[k]);
            ritzVectors.col(1) = update(ritzVectors.col(0), x0, betas[k]);
          } else {
            ritzVectors.col(j) = update(ritzVectors.col(j - 1),
                                        ritzVectors.col(j - 2), betas[k]);
          }
          const Vector x = ritzVectors.col(j);
          const double up = x.transpose() * A_ * x;
          const double down = x.transpose().dot(x);
          const double mu = up / down;
          if (mu * mu / 4 > maxBeta) {
            maxIndex = k;
            maxBeta = mu * mu / 4;
            break;
          }
        }
      }
    }

    beta_ = betas[maxIndex];
  }

  void perturb(int i) {
    // generate a 0.03*||x_0||_2 as stated in David's paper
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    int n = dim_;
    Vector disturb;
    disturb.resize(n);
    disturb.setZero();
    for (int i = 0; i < n; ++i) {
      disturb(i) = uniform01(generator);
    }
    disturb.normalize();

    Vector x0 = ritzVectors_.col(i);
    double magnitude = x0.norm();
    ritzVectors_.col(i) = x0 + 0.03 * magnitude * disturb;
  }

  // Perform power iteration on a single Ritz value
  // Updates ritzValues_ and ritzConverged_
  bool iterateOne(double tol, int i) {
    const Vector x = ritzVectors_.col(i);
    double theta = x.transpose() * A_ * x;

    // store the Ritz eigen value
    ritzValues_(i) = theta;

    // update beta
    if (accelerated_)
      beta_ = std::max(beta_, theta * theta / 4);

    const Vector diff = A_ * x - theta * x;
    double error = diff.norm();
    if (error < tol)
      ritzConverged_(i) = 1;
    return error < tol;
  }

  // Perform power iteration on all Ritz values
  // Updates ritzValues_, ritzVectors_, and ritzConverged_
  int iterate(double tol) {
    int nrConverged = 0;
    for (int i = 0; i < nrRequired_; i++) {
      if (iterateOne(tol, i)) {
        nrConverged += 1;
      }
      if (!accelerated_ && i < nrRequired_ - 1) {
        ritzVectors_.col(i + 1) = update(i + 1);
      } else if (accelerated_ && i > 0 && i < nrRequired_ - 1) {
        ritzVectors_.col(i + 1) = update(i + 1);
      }
    }
    return nrConverged;
  }

  size_t nrIterations() { return nrIterations_; }

  void sortEigenpairs() {
    std::vector<std::pair<double, int>> pairs;
    for (int i = 0; i < ritzConverged_.size(); ++i) {
      if (ritzConverged_(i))
        pairs.push_back({ritzValues_(i), i});
    }

    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<double, int> &left,
                 const std::pair<double, int> &right) {
                return left.first < right.first;
              });

    // initialize sorted Ritz eigenvalues and eigenvectors
    size_t nrConverged = pairs.size();
    sortedRitzValues_.resize(nrConverged);
    sortedRitzValues_.setZero();
    sortedRizVectors_.resize(dim_, nrConverged);
    sortedRizVectors_.setZero();

    // fill sorted Ritz eigenvalues and eigenvectors with sorted index
    for (size_t j = 0; j < nrConverged; ++j) {
      sortedRitzValues_(j) = pairs[j].first;
      sortedRizVectors_.col(j) = ritzVectors_.col(pairs[j].second);
    }
  }

  void reset() {
    if (accelerated_) {
      ritzVectors_.col(0) = update(ritzVectors_.col(dim_ - 1 - 1),
                                   ritzVectors_.col(dim_ - 1 - 2), beta_);
      ritzVectors_.col(1) =
          update(ritzVectors_.col(0), ritzVectors_.col(dim_ - 1 - 1), beta_);
    } else {
      ritzVectors_.col(0) = update(ritzVectors_.col(dim_ - 1));
    }
  }

  int compute(int maxit, double tol) {
    // Starting
    int i = 0;
    int nrConverged = 0;
    for (; i < maxit; i++) {
      nrIterations_ += 1;
      nrConverged = iterate(tol);
      if (nrConverged >= nrRequired_)
        break;
      else
        reset();
    }

    // sort the result
    sortEigenpairs();

    return std::min(nrRequired_, nrConverged);
  }

  Vector eigenvalues() { return sortedRitzValues_; }

  Matrix eigenvectors() { return sortedRizVectors_; }
};

} // namespace gtsam
