/*
Copyright 2025 Haihao Lu
Modified for CP_AL by Benqi Liu, 2026.

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

#include "cp_al_types.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include <stdbool.h>

#define CP_AL_MAX_LAMBDA_TIERS 16

typedef struct
{
    int num_rows;
    int num_cols;
    int num_nonzeros;
    int *row_ptr;
    int *col_ind;
    int *row_ind;
    double *val;
    int *transpose_map;
} cu_sparse_matrix_csr_t;

typedef struct
{
    int num_variables;
    int num_constraints;
    double *variable_lower_bound;
    double *variable_upper_bound;
    double *objective_vector;
    double objective_constant;
    cu_sparse_matrix_csr_t *constraint_matrix;
    cu_sparse_matrix_csr_t *constraint_matrix_t;
    double *constraint_lower_bound;
    double *constraint_upper_bound;
    int num_blocks_primal;
    int num_blocks_dual;
    int num_blocks_primal_dual;
    int num_blocks_nnz;
    double objective_vector_norm;
    double constraint_bound_norm;
    double *constraint_lower_bound_finite_val;
    double *constraint_upper_bound_finite_val;
    double *variable_lower_bound_finite_val;
    double *variable_upper_bound_finite_val;

    double *initial_primal_solution;
    double *current_primal_solution;
    double *cp_al_primal_solution;
    double *halpern_primal_solution;
    double *reflected_primal_solution;
    double *dual_product;
    double *initial_dual_solution;
    double *current_dual_solution;
    double *cp_al_dual_solution;
    double *halpern_dual_solution;
    double *reflected_dual_solution;
    double *primal_product;
    double step_size;
    double *d_primal_step_size;
    double *d_dual_step_size;
    double *d_al_penalty;
    double primal_weight;
    double al_lambda;
    double al_penalty;
    bool al_sigma_mode;
    bool al_sigma_adaptive;
    double al_sigma;
    double al_sigma_base;
    double al_sigma_min;
    double al_sigma_max;
    double al_sigma_best;
    double al_sigma_error_sum;
    double al_sigma_last_error;
    double al_lambda_param;
    double lambda_est;
    int total_count;
    bool is_this_major_iteration;
    double primal_weight_error_sum;
    double primal_weight_last_error;
    double best_primal_weight;
    double best_primal_dual_residual_gap;
    bool adaptive_lambda;
    bool adaptive_lambda_metrics_ready;
    int lambda_num_tiers;
    double lambda_values[CP_AL_MAX_LAMBDA_TIERS];
    int lambda_tier;
    int lambda_hold_remaining;
    int lambda_hold_epochs;
    int lambda_restart_count;
    int lambda_min_restarts;
    int lambda_min_iterations;
    int lambda_force_restart_iters;
    int lambda_last_restart_iter;
    bool lambda_race;
    int lambda_race_next_tier;
    int lambda_race_best_tier;
    int lambda_race_exploit_remaining;
    int lambda_race_exploit_epochs;
    int lambda_race_cooldown_remaining;
    int lambda_race_cooldown_epochs;
    double lambda_race_best_score;
    int lambda_race_cycle_best_tier;
    double lambda_race_cycle_best_score;
    double lambda_race_bad_ratio;
    double lambda_base;
    double lambda_low;
    double lambda_mid;
    double lambda_high;
    double lambda_min_time_sec;
    double lambda_feas_plateau_tol;
    double lambda_gap_plateau_tol;
    double lambda_fp_plateau_tol;
    double lambda_feas_active_tol;
    double lambda_gap_active_tol;
    double lambda_fp_active_tol;
    double lambda_gap_dominance;
    double lambda_epoch_feas_start;
    double lambda_epoch_gap_start;
    double lambda_epoch_fp_start;
    double lambda_epoch_time_start;

    double *constraint_rescaling;
    double *variable_rescaling;
    double constraint_bound_rescaling;
    double objective_vector_rescaling;
    bool cr_early_stopped;
    double *primal_slack;
    double *dual_slack;
    double rescaling_time_sec;
    clock_t start_time;
    double cumulative_time_sec;

    double *primal_residual;
    double absolute_primal_residual;
    double relative_primal_residual;
    double *dual_residual;
    double absolute_dual_residual;
    double relative_dual_residual;
    double primal_objective_value;
    double dual_objective_value;
    double objective_gap;
    double relative_objective_gap;
    double max_primal_ray_infeasibility;
    double max_dual_ray_infeasibility;
    double primal_ray_linear_objective;
    double dual_ray_objective;
    termination_reason_t termination_reason;

    double *delta_primal_solution;
    double *delta_dual_solution;
    double fixed_point_error;
    double initial_fixed_point_error;
    double last_trial_fixed_point_error;
    int restart_target;
    double restart_score_current;
    double restart_score_average;
    double restart_score_halpern;
    int inner_count;
    int *d_inner_count;

    cusparseHandle_t sparse_handle;
    cublasHandle_t blas_handle;
    size_t spmv_buffer_size;
    size_t primal_spmv_buffer_size;
    size_t dual_spmv_buffer_size;
    void *primal_spmv_buffer;
    void *dual_spmv_buffer;
    void *spmv_buffer;

    cusparseSpMatDescr_t matA;
    cusparseSpMatDescr_t matAt;
    cusparseDnVecDescr_t vec_primal_sol;
    cusparseDnVecDescr_t vec_dual_sol;
    cusparseDnVecDescr_t vec_primal_prod;
    cusparseDnVecDescr_t vec_dual_prod;

    double *ones_primal_d;
    double *ones_dual_d;

    double feasibility_polishing_time;
    int feasibility_iteration;

    cudaStream_t stream;
} cp_al_solver_state_t;

typedef struct
{
    double *con_rescale;
    double *var_rescale;
    double con_bound_rescale;
    double obj_vec_rescale;
    double rescaling_time_sec;
} rescale_info_t;
