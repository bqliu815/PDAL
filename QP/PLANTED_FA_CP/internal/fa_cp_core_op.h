/*
Copyright 2025 Haihao Lu
Copyright 2026 Hongpei Li
Modified for the RHR-FA-CP release by Benqi Liu, 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include "internal_types.h"
#include "fa_cp.h"
#include "fa_cp_types.h"
#include "preconditioner.h"
#include "solver.h"
#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif
__global__ void compute_lp_next_fa_cp_primal_solution_kernel(
    const double *current_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size);
__global__ void compute_lp_next_fa_cp_primal_solution_major_kernel(
    const double *current_primal, double *fa_cp_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size, double *dual_slack);
__global__ void compute_diagonal_q_next_fa_cp_primal_solution_kernel(
    const double *current_primal, double *reflected_primal,
    double *objective_product, const double *dual_product,
    const double *objective, const double *var_lb, const double *var_ub, int n,
    double step_size);
__global__ void compute_diagonal_q_next_fa_cp_primal_solution_major_kernel(
    const double *current_primal, double *fa_cp_primal, double *reflected_primal,
    double *objective_product, const double *dual_product,
    const double *objective, const double *var_lb, const double *var_ub, int n,
    double step_size);
__global__ void compute_next_fa_cp_dual_solution_kernel(
    const double *current_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb,
    const double *const_ub, int n, double step_size);
__global__ void compute_next_fa_cp_dual_solution_major_kernel(
    const double *current_dual, double *fa_cp_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb,
    const double *const_ub, int n, double step_size);
__global__ void compute_delta_solution_kernel(
    const double *initial_primal, const double *fa_cp_primal,
    double *delta_primal, const double *initial_dual, const double *fa_cp_dual,
    double *delta_dual, int n_vars, int n_cons);
__global__ void compute_and_rescale_reduced_cost_kernel(
    double *reduced_cost, const double *objective, const double *dual_product,
    const double *variable_rescaling, const double objective_vector_rescaling,
    const double constraint_bound_rescaling, int n_vars);
void fa_cp_update(fa_cp_solver_state_t *state);
void halpern_update(fa_cp_solver_state_t *state, double reflection_coefficient);

void rescale_solution(fa_cp_solver_state_t *state);

fa_cp_result_t *create_result_from_state(fa_cp_solver_state_t *state,
                                         const qp_problem_t *original_problem);

void perform_restart(fa_cp_solver_state_t *state,
                     const fa_cp_parameters_t *params);

void initialize_step_size_and_primal_weight(fa_cp_solver_state_t *state,
                                            const fa_cp_parameters_t *params);

fa_cp_solver_state_t *
initialize_solver_state(const fa_cp_parameters_t *params,
                        const qp_problem_t *original_problem,
                        const rescale_info_t *rescale_info);

void compute_fixed_point_error(fa_cp_solver_state_t *state);

void perform_primal_restart(fa_cp_solver_state_t *state);
void perform_dual_restart(fa_cp_solver_state_t *state);

void primal_feasibility_polish(const fa_cp_parameters_t *params,
                               fa_cp_solver_state_t *state,
                               const fa_cp_solver_state_t *ori_state);
void dual_feasibility_polish(const fa_cp_parameters_t *params,
                             fa_cp_solver_state_t *state,
                             const fa_cp_solver_state_t *ori_state);

void primal_feas_polish_state_free(fa_cp_solver_state_t *state);
void dual_feas_polish_state_free(fa_cp_solver_state_t *state);

void feasibility_polish(const fa_cp_parameters_t *params,
                        fa_cp_solver_state_t *state);

void compute_primal_fixed_point_error(fa_cp_solver_state_t *state);
void compute_dual_fixed_point_error(fa_cp_solver_state_t *state);

fa_cp_solver_state_t *
initialize_primal_feas_polish_state(const fa_cp_solver_state_t *original_state);
fa_cp_solver_state_t *
initialize_dual_feas_polish_state(const fa_cp_solver_state_t *original_state);

#ifdef __cplusplus
}
#endif
