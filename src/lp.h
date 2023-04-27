#ifndef LP_H_
#define LP_H_

// #include <iostream>
#include <vector>
#include <assert.h>
#include "ortools/linear_solver/linear_solver.h"
#include "utils.h"

using namespace std;

namespace operations_research {
    vector<double> solve(vector<vector<double>> q_values, vector<double> p) 
    {   
        // q_values = {{1e+10, 1e+9, 1e+9}, {1e+9, 1e+9, 1e+9}, {1e+9, 1e+9, 1e+9}, {1e+9, 1e+9, 1e+9}};
        // p = {1.0/3, 1.0/3, 1.0/3};
        // cout << "Weight p: ";
        // for (auto a: p) {
        //     cout << a << ", ";
        // }
        // cout << endl;
        // cout << "Q values: ";
        // for (auto v: q_values) {
        //     for (auto vv: v) {
        //         cout << vv << ", ";
        //     }
        //     cout << endl;
        // }

        assert(q_values.size() > 0 && q_values[0].size() > 0);
        int num_actions = q_values.size();
        int n_reward = q_values[0].size();

        // create solver
        unique_ptr<MPSolver> solver(MPSolver::CreateSolver("GLOP"));
        const double ub = solver->infinity();
        const double lb = -1 * ub;
        const int num_vars = n_reward + n_reward * n_reward + num_actions;
        vector<const MPVariable*> vars;
        vars.reserve(num_vars);
        // next we define vars
        // x_1, ..., x_n
        for (int i = 0; i < n_reward; i++) {
            vars.push_back(solver->MakeNumVar(lb, ub, ""));
        }
        // d_11, ..., d_nn
        for (int i = 0; i < n_reward * n_reward; i++) {
            vars.push_back(solver->MakeNumVar(0.0, ub, ""));
        }
        // pi, distribution over actions
        for (int i = 0; i < num_actions; i++) {
            vars.push_back(solver->MakeNumVar(0.0, 1.0, ""));
        }
        const int num_ineq_constraits = n_reward * n_reward, num_eq_constraits = 1;
        // create inequal constraits
        // cout << "Inequal constraits: " << endl;
        for (int k = 0; k < n_reward; k++) {
            for (int i = 0; i < n_reward; i++) {
                MPConstraint* ineq_constraint = solver->MakeRowConstraint(lb, 0.0, "");
                ineq_constraint->SetCoefficient(vars[k], 1.0);
                for (int k_ = 0; k_ < n_reward; k_++) {
                    if (k_ != k) ineq_constraint->SetCoefficient(vars[k_], 0.0);
                }
                ineq_constraint->SetCoefficient(vars[(i+1)*n_reward + k], -1.0);
                for (int ik = n_reward; ik < num_vars - num_actions; ik++) {
                    if (ik != (i+1)*n_reward + k) ineq_constraint->SetCoefficient(vars[ik], 0.0);
                }
                for (int a = num_vars - num_actions; a < num_vars; a++) {
                    ineq_constraint->SetCoefficient(vars[a], -1 * q_values[a + num_actions - num_vars][i]);
                }
                // for (int i = 0; i < num_vars; i++) {
                //     cout << ineq_constraint->GetCoefficient(vars[i]) << ", ";
                // }
                // cout << endl;
            }
        }
        // create equal constraits
        MPConstraint* eq_constraint = solver->MakeRowConstraint(1.0, 1.0, "");
        for (int i = 0; i < num_vars - num_actions; i++) {
            eq_constraint->SetCoefficient(vars[i], 0.0);
        }
        for (int a = num_vars - num_actions; a < num_vars; a++) {
            eq_constraint->SetCoefficient(vars[a], 1.0);
        }
        // cout << "Equal constraits: " << endl;
        // for (int i = 0; i < num_vars; i++) {
        //     cout << eq_constraint->GetCoefficient(vars[i]) << ", ";
        // }
        // cout << endl;
        // define the weight w
        vector<double> w, w_prime;
        w.reserve(n_reward);
        w_prime.reserve(n_reward);
        for (int i = 0; i < n_reward; i++) {
            w[i] = 1.0 / pow(2, i);
        }
        for (int i = 0; i < n_reward; i++) {
            double next = (i == n_reward-1 ? 0.0 : w[i+1]);
            w_prime[i] = n_reward * (w[i] - next);
        }
        // create objectives
        MPObjective* const objective = solver->MutableObjective();
        for (int k = 0; k < n_reward; k++) {
            objective->SetCoefficient(vars[k], (k+1) * w_prime[k] / n_reward);
        }
        for (int i = 0; i < n_reward; i++) {
            for (int k = 0; k < n_reward; k++) {
                objective->SetCoefficient(vars[(i+1)*n_reward + k], -1 * w_prime[k] * p[i]);
            }
        }
        for (int i = num_vars - num_actions; i < num_vars; i++) {
            objective->SetCoefficient(vars[i], 0.0);
        }
        // cout << "Obj coeffs: " << endl;
        // for (int i = 0; i < num_vars; i++) {
        //     cout << objective->GetCoefficient(vars[i]) << ", ";
        // }
        // cout << endl;
        objective->SetMaximization();
        solver->Solve();
        
        vector<double> pi(num_actions, 0.0);
        for (int a = num_vars - num_actions; a < num_vars; a++) {
            pi[a + num_actions - num_vars] = vars[a]->solution_value();
        }
        // cout << "Solution: " << endl;
        // for (int i = 0; i < num_vars; i++) {
        //     cout << vars[i]->solution_value() << ", ";
        // }
        // cout << endl;
        // cout << "Optimal value = " << objective->Value() << endl;
        // exit(-1);
        return pi;
    }
}

#endif // LP_H_