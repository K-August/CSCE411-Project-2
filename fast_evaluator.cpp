#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Highs.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace py = pybind11;
using namespace Eigen;

// Helper to iterate combinations exactly like Python's itertools.combinations
bool next_combination(std::vector<int>& comb, int n, int m) {
    int i = m - 1;
    while (i >= 0 && comb[i] == n - m + i) {
        --i;
    }
    if (i < 0) return false;
    ++comb[i];
    for (int j = i + 1; j < m; ++j) comb[j] = comb[j - 1] + 1;
    return true;
}

// Geometric check
bool is_spherically_spread(const MatrixXd& G, double threshold) {
    int cols = G.cols();
    for (int i = 0; i < cols; ++i) {
        for (int j = i + 1; j < cols; ++j) {
            double cos_sim = std::abs(G.col(i).dot(G.col(j))) / (G.col(i).norm() * G.col(j).norm());
            if (cos_sim > threshold) return false;
        }
    }
    return true;
}

// Main Evaluator (P passed by const reference to stop Pybind11 memory duplication)
double calc_mHeight_efficient(int n, int k, int m, const MatrixXd& P, double threshold) {
    // G = [I | P]
    MatrixXd G(k, n);
    G.leftCols(k) = MatrixXd::Identity(k, k);
    G.rightCols(n - k) = P;

    if (!is_spherically_spread(G, threshold)) {
        return 6969.0;
    }

    double h_max = 0.0;
    
    // Prepare initial combination S = [0, 1, ..., m-1]
    std::vector<int> S(m);
    for (int i = 0; i < m; ++i) S[i] = i;

    // ==============================================================
    // PERFORMANCE UPGRADE: Pre-allocate all memory outside the loop!
    // ==============================================================
    std::vector<int> S_bar;
    S_bar.reserve(n - m);
    std::vector<double> c(k);

    HighsModel model;
    model.lp_.num_col_ = k;
    model.lp_.num_row_ = n - m;
    model.lp_.sense_ = ObjSense::kMaximize; 
    model.lp_.col_lower_.assign(k, -kHighsInf);
    model.lp_.col_upper_.assign(k, kHighsInf);
    model.lp_.row_lower_.assign(n - m, -1.0);
    model.lp_.row_upper_.assign(n - m, 1.0);
    model.lp_.col_cost_.assign(k, 0.0);

    // Prepare Compressed Column format vectors
    int num_nz_max = (n - m) * k;
    model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    model.lp_.a_matrix_.start_.resize(k + 1, 0);
    model.lp_.a_matrix_.index_.reserve(num_nz_max);
    model.lp_.a_matrix_.value_.reserve(num_nz_max);

    do {
        // Fast clear (resets size to 0 without deleting the underlying memory capacity)
        S_bar.clear(); 
        for (int i = 0; i < n; ++i) {
            if (std::find(S.begin(), S.end(), i) == S.end()) {
                S_bar.push_back(i);
            }
        }
        
        model.lp_.a_matrix_.index_.clear();
        model.lp_.a_matrix_.value_.clear();

        for (int col = 0; col < k; ++col) {
            model.lp_.a_matrix_.start_[col] = model.lp_.a_matrix_.index_.size();
            for (int row = 0; row < n - m; ++row) {
                double val = G(col, S_bar[row]);
                if (val != 0.0) {
                    model.lp_.a_matrix_.index_.push_back(row);
                    model.lp_.a_matrix_.value_.push_back(val);
                }
            }
        }
        model.lp_.a_matrix_.start_[k] = model.lp_.a_matrix_.index_.size();

        // ==============================================================
        // INSTANTIATE HIGHS ENGINE INSIDE THE LOOP
        // Because `model` is already pre-allocated, this is incredibly fast
        // and 100% guarantees no solver memory corruption.
        // ==============================================================
        Highs highs;
        highs.setOptionValue("output_flag", false); // Keep it silent!
        highs.passModel(model);

        // Solve for each vector in S
        for (int i : S) {
            for(int j = 0; j < k; ++j) {
                c[j] = G(j, i);
            }
            
            highs.changeColsCost(0, k-1, c.data());
            highs.run();
            
            HighsModelStatus status = highs.getModelStatus();
            if (status == HighsModelStatus::kOptimal) {
                double z = highs.getInfo().objective_function_value;
                if (z > h_max) h_max = z;
            } else if (status == HighsModelStatus::kUnbounded) {
                return std::numeric_limits<double>::infinity(); 
            }
        }
    } while (next_combination(S, n, m));

    return h_max;
}

// Pybind11 binding
PYBIND11_MODULE(fast_evaluator, m) {
    m.def("calc_mHeight_efficient", &calc_mHeight_efficient, "Calculate mHeight with HiGHS");
}