/*
Copyright 2025 Haihao Lu
Copyright 2026 Hongpei Li
Modified for the RHR-Lin-CP-AL release by Benqi Liu, 2026.

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
#include "cp_al.h"
#include "cp_al_types.h"
#include "preconditioner.h"
#include "solver.h"
#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif
__global__ void compute_lp_next_cp_al_primal_solution_kernel(
    const double *current_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size);
__global__ void compute_lp_next_cp_al_primal_solution_major_kernel(
    const double *current_primal, double *cp_al_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size, double *dual_slack);
__global__ void compute_diagonal_q_next_cp_al_primal_solution_kernel(
    const double *current_primal, double *reflected_primal,
    double *objective_product, const double *dual_product,
    const double *objective, const double *var_lb, const double *var_ub, int n,
    double step_size);
__global__ void compute_diagonal_q_next_cp_al_primal_solution_major_kernel(
    const double *current_primal, double *cp_al_primal, double *reflected_primal,
    double *objective_product, const double *dual_product,
    const double *objective, const double *var_lb, const double *var_ub, int n,
    double step_size);
__global__ void compute_next_cp_al_dual_solution_kernel(
    const double *current_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb,
    const double *const_ub, int n, double step_size);
__global__ void compute_next_cp_al_dual_solution_major_kernel(
    const double *current_dual, double *cp_al_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb,
    const double *const_ub, int n, double step_size);
__global__ void compute_delta_solution_kernel(
    const double *initial_primal, const double *cp_al_primal,
    double *delta_primal, const double *initial_dual, const double *cp_al_dual,
    double *delta_dual, int n_vars, int n_cons);
__global__ void compute_and_rescale_reduced_cost_kernel(
    double *reduced_cost, const double *objective, const double *dual_product,
    const double *variable_rescaling, const double objective_vector_rescaling,
    const double constraint_bound_rescaling, int n_vars);
void cp_al_update(cp_al_solver_state_t *state);
void halpern_update(cp_al_solver_state_t *state, double reflection_coefficient);

void rescale_solution(cp_al_solver_state_t *state);

cp_al_result_t *create_result_from_state(cp_al_solver_state_t *state,
                                         const qp_problem_t *original_problem);

void perform_restart(cp_al_solver_state_t *state,
                     const cp_al_parameters_t *params);

void initialize_step_size_and_primal_weight(cp_al_solver_state_t *state,
                                            const cp_al_parameters_t *params);

cp_al_solver_state_t *
initialize_solver_state(const cp_al_parameters_t *params,
                        const qp_problem_t *original_problem,
                        const rescale_info_t *rescale_info);

void compute_fixed_point_error(cp_al_solver_state_t *state);

void perform_primal_restart(cp_al_solver_state_t *state);
void perform_dual_restart(cp_al_solver_state_t *state);

void primal_feasibility_polish(const cp_al_parameters_t *params,
                               cp_al_solver_state_t *state,
                               const cp_al_solver_state_t *ori_state);
void dual_feasibility_polish(const cp_al_parameters_t *params,
                             cp_al_solver_state_t *state,
                             const cp_al_solver_state_t *ori_state);

void primal_feas_polish_state_free(cp_al_solver_state_t *state);
void dual_feas_polish_state_free(cp_al_solver_state_t *state);

void feasibility_polish(const cp_al_parameters_t *params,
                        cp_al_solver_state_t *state);

void compute_primal_fixed_point_error(cp_al_solver_state_t *state);
void compute_dual_fixed_point_error(cp_al_solver_state_t *state);

cp_al_solver_state_t *
initialize_primal_feas_polish_state(const cp_al_solver_state_t *original_state);
cp_al_solver_state_t *
initialize_dual_feas_polish_state(const cp_al_solver_state_t *original_state);

#ifdef __cplusplus
}
#endif