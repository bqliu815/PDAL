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

#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  TERMINATION_REASON_UNSPECIFIED,
  TERMINATION_REASON_OPTIMAL,
  TERMINATION_REASON_PRIMAL_INFEASIBLE,
  TERMINATION_REASON_DUAL_INFEASIBLE,
  TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED,
  TERMINATION_REASON_TIME_LIMIT,
  TERMINATION_REASON_ITERATION_LIMIT,
  TERMINATION_REASON_FEAS_POLISH_SUCCESS
} termination_reason_t;

typedef enum { NORM_TYPE_L2 = 0, NORM_TYPE_L_INF = 1 } norm_type_t;

typedef struct {
  int *row_ptr;
  int *col_ind;
  double *val;
} CsrComponent;

typedef enum {
  FA_CP_SPARSE_Q,
  FA_CP_DIAG_Q,
  FA_CP_LOW_RANK_PLUS_SPARSE_Q,
  FA_CP_LOW_RANK_Q,
  FA_CP_NON_Q
} quad_obj_type_t;

typedef struct {
  int num_variables;
  int num_constraints;
  int num_rank_lowrank_obj;
  double *variable_lower_bound;
  double *variable_upper_bound;
  double *objective_vector;
  double objective_constant;

  CsrComponent *constraint_matrix;
  int constraint_matrix_num_nonzeros;

  CsrComponent *objective_sparse_matrix;
  int objective_sparse_matrix_num_nonzeros;

  CsrComponent *objective_lowrank_matrix;
  int objective_lowrank_matrix_num_nonzeros;

  double *constraint_lower_bound;
  double *constraint_upper_bound;

  double *primal_start;
  double *dual_start;

} qp_problem_t;

typedef struct {
  double artificial_restart_threshold;
  double sufficient_reduction_for_restart;
  double necessary_reduction_for_restart;
  double k_p;
  double k_i;
  double k_d;
  double i_smooth;
} restart_parameters_t;

typedef struct {
  double eps_optimal_relative;
  double eps_feasible_relative;
  double eps_feas_polish_relative;
  double eps_infeasible;
  double time_sec_limit;
  int iteration_limit;
} termination_criteria_t;

typedef enum {
  ADAPTIVE_AL_SIGMA_MODE_BIDIRECTIONAL = 0,
  ADAPTIVE_AL_SIGMA_MODE_UP_ONLY = 1,
  ADAPTIVE_AL_SIGMA_MODE_DOWN_ONLY = 2
} adaptive_al_sigma_mode_t;

typedef enum {
  ADAPTIVE_AL_SIGMA_RULE_TRIGGER = 0,
  ADAPTIVE_AL_SIGMA_RULE_SQRT = 1,
  ADAPTIVE_AL_SIGMA_RULE_BANGBANG = 2
} adaptive_al_sigma_rule_t;

typedef struct {
  int l_inf_ruiz_iterations;
  bool has_pock_chambolle_alpha;
  double pock_chambolle_alpha;
  bool bound_objective_rescaling;
  bool use_al_qp;
  double al_sigma;
  bool adaptive_al_sigma;
  adaptive_al_sigma_mode_t adaptive_al_sigma_mode;
  adaptive_al_sigma_rule_t adaptive_al_sigma_rule;
  double al_sigma_min;
  double al_sigma_max;
  double al_sigma_increase_factor;
  double al_sigma_decrease_factor;
  double al_sigma_primal_trigger;
  double al_sigma_gap_trigger;
  const char *trajectory_csv;
  int verbose;
  int termination_evaluation_frequency;
  int sv_max_iter;
  double sv_tol;
  termination_criteria_t termination_criteria;
  restart_parameters_t restart_params;
  double reflection_coefficient;
  bool feasibility_polishing;
  norm_type_t optimality_norm;
} fa_cp_parameters_t;

typedef struct {
  int num_variables;
  int num_constraints;
  int num_nonzeros;

  int num_reduced_variables;
  int num_reduced_constraints;
  int num_reduced_nonzeros;

  double *primal_solution;
  double *dual_solution;
  double *reduced_cost;

  int total_count;
  int total_inner_count;
  double rescaling_time_sec;
  double cumulative_time_sec;

  double absolute_primal_residual;
  double relative_primal_residual;
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
  double feasibility_polishing_time;
  int feasibility_iteration;
} fa_cp_result_t;

// matrix formats
typedef enum {
  matrix_dense = 0,
  matrix_csr = 1,
  matrix_csc = 2,
  matrix_coo = 3
} matrix_format_t;

// matrix descriptor
typedef struct {
  int m; // num_constraints
  int n; // num_variables
  matrix_format_t fmt;

  // treat abs(x) < zero_tolerance as zero
  double zero_tolerance;

  union MatrixData {
    struct MatrixDense { // Dense (row-major)
      const double *A;   // m*n
    } dense;

    struct MatrixCSR { // CSR
      int nnz;
      const int *row_ptr;
      const int *col_ind;
      const double *vals;
    } csr;

    struct MatrixCSC { // CSC
      int nnz;
      const int *col_ptr;
      const int *row_ind;
      const double *vals;
    } csc;

    struct MatrixCOO { // COO
      int nnz;
      const int *row_ind;
      const int *col_ind;
      const double *vals;
    } coo;
  } data;
} matrix_desc_t;

#ifdef __cplusplus
} // extern "C"
#endif
