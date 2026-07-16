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

#include "internal_types.h"
#include "fa_cp.h"
#include "fa_cp_core_op.h"
#include "preconditioner.h"
#include "solver.h"
#include "solver_state.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void initialize_trajectory_csv(const char *trajectory_csv) {
  if (trajectory_csv == NULL || trajectory_csv[0] == '\0') {
    return;
  }
  FILE *outfile = fopen(trajectory_csv, "w");
  if (outfile == NULL) {
    perror("Error opening trajectory CSV");
    exit(EXIT_FAILURE);
  }
  fprintf(outfile,
          "iteration,time_sec,rel_primal,rel_dual,rel_gap,kkt_residual,"
          "fixed_point_residual,restart,sigma,primal_weight,al_sigma,"
          "lambda_est,primal_step,dual_step\n");
  fclose(outfile);
}

static void compute_trajectory_diagnostics(const fa_cp_solver_state_t *state,
                                           double *sigma,
                                           double *primal_weight,
                                           double *al_sigma,
                                           double *lambda_est,
                                           double *primal_step,
                                           double *dual_step) {
  *sigma = NAN;
  *primal_weight = state->primal_weight;
  *al_sigma = state->use_al_qp ? state->al_sigma : NAN;
  *lambda_est = NAN;
  *primal_step = NAN;
  *dual_step = NAN;

  if (state->primal_weight > 0.0) {
    *primal_step = state->step_size / state->primal_weight;
    *dual_step = state->step_size * state->primal_weight;
  }
}

static void append_trajectory_row(const char *trajectory_csv,
                                  const fa_cp_solver_state_t *state,
                                  bool restart_flag) {
  if (trajectory_csv == NULL || trajectory_csv[0] == '\0') {
    return;
  }
  FILE *outfile = fopen(trajectory_csv, "a");
  if (outfile == NULL) {
    perror("Error appending trajectory CSV");
    exit(EXIT_FAILURE);
  }
  const double kkt_residual =
      fmax(state->relative_primal_residual,
           fmax(state->relative_dual_residual, state->relative_objective_gap));
  const double fixed_point_residual =
      state->total_count == 0 ? NAN : state->fixed_point_error;
  double sigma = NAN;
  double primal_weight = NAN;
  double al_sigma = NAN;
  double lambda_est = NAN;
  double primal_step = NAN;
  double dual_step = NAN;
  compute_trajectory_diagnostics(state, &sigma, &primal_weight, &al_sigma,
                                 &lambda_est, &primal_step, &dual_step);
  fprintf(outfile,
          "%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%d,%.17e,%.17e,%.17e,"
          "%.17e,%.17e,%.17e\n",
          state->total_count, state->cumulative_time_sec,
          state->relative_primal_residual, state->relative_dual_residual,
          state->relative_objective_gap, kkt_residual, fixed_point_residual,
          restart_flag ? 1 : 0, sigma, primal_weight, al_sigma, lambda_est,
          primal_step, dual_step);
  fclose(outfile);
}

fa_cp_result_t *optimize(const fa_cp_parameters_t *input_params,
                         const qp_problem_t *original_problem) {
  qp_problem_t *working_problem = deepcopy_problem(original_problem);
  fa_cp_parameters_t copyed_params = *input_params;
  fa_cp_parameters_t *params = &copyed_params;
  print_initial_info(input_params, original_problem);

  rescale_info_t *rescale_info = rescale_problem(params, working_problem);
  fa_cp_solver_state_t *state =
      initialize_solver_state(params, working_problem, rescale_info);

  if (state->quadratic_objective_term->nonconvexity < 0) {
    state->inner_solver->iteration_limit = 1;
  }

  rescale_info_free(rescale_info);
  initialize_step_size_and_primal_weight(state, params);
  initialize_trajectory_csv(params->trajectory_csv);
  clock_t start_time = clock();
  bool do_restart = false;
  while (state->total_count < params->termination_criteria.iteration_limit) {
    bool should_consider_restart =
        state->is_this_major_iteration || state->total_count == 0;
    bool planned_restart = false;
    if ((state->is_this_major_iteration || state->total_count == 0) ||
        (state->total_count % get_print_frequency(state->total_count) == 0)) {
      compute_residual(state, params->optimality_norm);
      if (state->is_this_major_iteration &&
          state->total_count < 3 * params->termination_evaluation_frequency) {
        compute_infeasibility_information(state);
      }

      state->cumulative_time_sec =
          (double)(clock() - start_time) / CLOCKS_PER_SEC;

      check_termination_criteria(state, &params->termination_criteria);
      if (should_consider_restart &&
          state->termination_reason == TERMINATION_REASON_UNSPECIFIED) {
        planned_restart =
            should_do_adaptive_restart(state, &params->restart_params,
                                       params->termination_evaluation_frequency);
      }
      append_trajectory_row(params->trajectory_csv, state, planned_restart);
      display_iteration_stats(state, params->verbose);
      if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED) {
        break;
      }
    }

    if (should_consider_restart) {
      do_restart = planned_restart;
      if (do_restart)
        perform_restart(state, params);
    }

    state->is_this_major_iteration =
        ((state->total_count + 1) % params->termination_evaluation_frequency) ==
        0;

    fa_cp_update(state);

    if (state->is_this_major_iteration || do_restart) {
      compute_fixed_point_error(state);
      if (do_restart) {
        state->initial_fixed_point_error = state->fixed_point_error;
        do_restart = false;
      }
    }
    halpern_update(state, params->reflection_coefficient);

    state->inner_count++;
    state->total_count++;
  }

  if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED) {
    state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
    compute_residual(state, params->optimality_norm);
    append_trajectory_row(params->trajectory_csv, state, false);
    display_iteration_stats(state, params->verbose);
  }

  if (params->feasibility_polishing &&
      state->termination_reason != TERMINATION_REASON_DUAL_INFEASIBLE &&
      state->termination_reason != TERMINATION_REASON_PRIMAL_INFEASIBLE) {
    feasibility_polish(params, state);
  }

  fa_cp_result_t *result = create_result_from_state(state, original_problem);

  fa_cp_final_log(result, params);
  fa_cp_solver_state_free(state);
  qp_problem_free(working_problem);
  return result;
}
