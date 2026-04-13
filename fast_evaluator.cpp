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
        // Prevent division by zero if GA generates a zero-column
        double norm_i = G.col(i).norm();
        if (norm_i == 0.0) continue; 

        for (int j = i + 1; j < cols; ++j) {
            double norm_j = G.col(j).norm();
            if (norm_j == 0.0) continue;

            double cos_sim = std::abs(G.col(i).dot(G.col(j))) / (norm_i * norm_j);
            if (cos_sim > threshold) return false;
        }
    }
    return true;
}

// Main Evaluator
// FIX 1: Use Eigen::Ref to safely borrow memory from Python without taking ownership
double calc_mHeight_efficient(int n, int k, int m, const Eigen::Ref<const MatrixXd>& P, double threshold) {
    
    // Safety guard: If the genetic algorithm mutates into invalid bounds, reject safely.
    if (m >= n || k > n || m <= 0) return 0.0;

    // G = [I | P]
    MatrixXd G(k, n);
    G.leftCols(k) = MatrixXd::Identity(k, k);
    G.rightCols(n - k) = P;

    if (!is_spherically_spread(G, threshold)) {
        return 6969.0;
    }

    double h_max = 0.0;
    
    std::vector<int> S(m);
    for (int i = 0; i < m; ++i) S[i] = i;

    // FIX 2: Release Python GIL so your Genetic Algorithm can multi-thread this function!
    py::gil_scoped_release release; 

    do {
        std::vector<int> S_bar;
        S_bar.reserve(n - m);
        for (int i = 0; i < n; ++i) {
            if (std::find(S.begin(), S.end(), i) == S.end()) {
                S_bar.push_back(i);
            }
        }
        
        // FIX 3: Localized scope. C++ automatically destroys and cleans up memory
        // exactly at the end of the loop. 100% safe from double-free corruption.
        Highs highs;
        highs.setOptionValue("output_flag", false);
        
        HighsModel model;
        model.lp_.num_col_ = k;
        model.lp_.num_row_ = n - m;
        model.lp_.sense_ = ObjSense::kMaximize; 
        
        model.lp_.col_lower_.assign(k, -kHighsInf);
        model.lp_.col_upper_.assign(k, kHighsInf);

        model.lp_.row_lower_.assign(n - m, -1.0);
        model.lp_.row_upper_.assign(n - m, 1.0);

        int num_nz = (n - m) * k;
        model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
        model.lp_.a_matrix_.start_.resize(k + 1, 0);
        model.lp_.a_matrix_.index_.reserve(num_nz);
        model.lp_.a_matrix_.value_.reserve(num_nz);

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

        model.lp_.col_cost_.assign(k, 0.0);
        highs.passModel(model);

        for (int i : S) {
            std::vector<double> c(k);
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

PYBIND11_MODULE(fast_evaluator, m) {
    m.def("calc_mHeight_efficient", &calc_mHeight_efficient, "Calculate mHeight with HiGHS");
}