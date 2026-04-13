#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include "Highs.h" 
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> 

using namespace Eigen;
using namespace std;
namespace py = pybind11;

// Pure C++ implementation of the spherical spread check.
// This calculates the absolute cosine similarity between all pairs of columns.
// It short-circuits and returns false the moment it finds a pair > threshold.
bool is_spherically_spread(const MatrixXd& G, double threshold) {
    int n = G.cols();
    for (int i = 0; i < n; ++i) {
        double norm_i = G.col(i).norm();
        if (norm_i == 0.0) continue; 

        for (int j = i + 1; j < n; ++j) {
            double norm_j = G.col(j).norm();
            if (norm_j == 0.0) continue;

            double dot_prod = G.col(i).dot(G.col(j));
            double cos_sim = std::abs(dot_prod / (norm_i * norm_j));

            if (cos_sim > threshold) {
                return false; 
            }
        }
    }
    return true;
}

double calc_mHeight_efficient(int n, int k, int m, const MatrixXd& P, double threshold = 0.95) {
    // 1. Construct G = [I | P]
    MatrixXd G(k, n);
    G.leftCols(k) = MatrixXd::Identity(k, k);
    G.rightCols(n - k) = P;

    // 2. Check spherical spread strictly in C++
    if (!is_spherically_spread(G, threshold)) {
        return 6969.0;
    }

    double h_max = 0.0;
    
    // Lexicographically largest array ensures prev_permutation gets all combinations
    vector<bool> v(n);
    fill(v.begin(), v.begin() + m, true);

    do {
        vector<int> S, S_bar;
        for (int i = 0; i < n; ++i) {
            if (v[i]) S.push_back(i);
            else S_bar.push_back(i);
        }

        // Bug-for-bug compatibility with SciPy: Explicitly stack A_all and -A_all
        int num_constraints = 2 * (n - m);

        for (int i : S) {
            // Instantiate a completely fresh solver per LP to match SciPy's zero-state-pollution
            Highs highs;
            highs.setOptionValue("output_flag", false);
            highs.setOptionValue("presolve", "on"); 

            HighsModel model;
            model.lp_.num_col_ = k;
            model.lp_.num_row_ = num_constraints;
            model.lp_.sense_ = ObjSense::kMinimize;

            // bounds=(None, None) means variables are entirely free
            model.lp_.col_lower_.assign(k, -kHighsInf);
            model.lp_.col_upper_.assign(k, kHighsInf);

            // A_ub * x <= b_ub (b_ub = np.ones)
            model.lp_.row_lower_.assign(num_constraints, -kHighsInf); 
            model.lp_.row_upper_.assign(num_constraints, 1.0);        

            model.lp_.a_matrix_.num_col_ = k;
            model.lp_.a_matrix_.num_row_ = num_constraints;
            model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
            model.lp_.a_matrix_.start_.push_back(0);

            for (int col = 0; col < k; ++col) {
                // Block 1: Positive constraints (corresponds to A_all)
                for (size_t row_idx = 0; row_idx < S_bar.size(); ++row_idx) {
                    int orig_col = S_bar[row_idx];
                    double val = G(col, orig_col);
                    if (val != 0.0) {
                        model.lp_.a_matrix_.index_.push_back(row_idx);
                        model.lp_.a_matrix_.value_.push_back(val);
                    }
                }
                
                // Block 2: Negative constraints (corresponds to -A_all)
                for (size_t row_idx = 0; row_idx < S_bar.size(); ++row_idx) {
                    int orig_col = S_bar[row_idx];
                    double val = -G(col, orig_col);
                    if (val != 0.0) {
                        // Offset by S_bar.size() so rows perfectly align linearly
                        model.lp_.a_matrix_.index_.push_back(row_idx + S_bar.size());
                        model.lp_.a_matrix_.value_.push_back(val);
                    }
                }
                model.lp_.a_matrix_.start_.push_back(model.lp_.a_matrix_.index_.size());
            }

            // Set objective exactly like `c = (-1.0 * G[:, i]).flatten()`
            vector<double> c(k);
            for (int col = 0; col < k; ++col) {
                c[col] = -1.0 * G(col, i);
            }
            model.lp_.col_cost_ = c;

            highs.passModel(model);
            highs.run();
            HighsModelStatus status = highs.getModelStatus();

            if (status == HighsModelStatus::kOptimal) {
                double z = -highs.getInfo().objective_function_value;
                if (z > h_max) h_max = z;
            } else if (status == HighsModelStatus::kUnbounded) {
                return INFINITY; 
            }
        }
    } while (prev_permutation(v.begin(), v.end()));

    return h_max;
}

PYBIND11_MODULE(mheight_ext, m) {
    m.def("calc_mHeight_efficient", &calc_mHeight_efficient, 
          "Calculate mHeight efficiently using pure C++",
          py::arg("n"), py::arg("k"), py::arg("m"), py::arg("P"), 
          py::arg("threshold") = 0.95);
}