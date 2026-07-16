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

#include "internal_types.h"
#include "cp_al.h"
#include "cp_al_core_op.h"
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
#include <time.h>

cp_al_result_t *optimize(const cp_al_parameters_t *input_params,
                         const qp_problem_t *original_problem) {
  qp_problem_t *working_problem = deepcopy_problem(original_problem);
  cp_al_parameters_t copyed_params = *input_params;
  cp_al_parameters_t *params = &copyed_params;
  print_initial_info(input_params, original_problem);

  rescale_info_t *rescale_info = rescale_problem(params, working_problem);
  cp_al_solver_state_t *state =
      initialize_solver_state(params, working_problem, rescale_info);

  if (state->quadratic_objective_term->nonconvexity < 0) {
    state->inner_solver->iteration_limit = 1;
  }

  rescale_info_free(rescale_info);
  initialize_step_size_and_primal_weight(state, params);
  clock_t start_time = clock();
  while (state->total_count < params->termination_criteria.iteration_limit) {
    bool evaluate_now =
        (state->total_count == 0) ||
        (state->total_count % params->termination_evaluation_frequency) == 0;
    state->is_this_major_iteration = evaluate_now;

    if (evaluate_now ||
        (state->total_count % get_print_frequency(state->total_count) == 0)) {
      compute_residual(state, params->optimality_norm);
      if (evaluate_now &&
          state->total_count < 3 * params->termination_evaluation_frequency) {
        compute_infeasibility_information(state);
      }

      state->cumulative_time_sec =
          (double)(clock() - start_time) / CLOCKS_PER_SEC;

      check_termination_criteria(state, &params->termination_criteria);
      display_iteration_stats(state, params->verbose);
      if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED) {
        break;
      }
    }

    cp_al_update(state);
    compute_fixed_point_error(state);

    if (!isfinite(state->initial_fixed_point_error)) {
      state->initial_fixed_point_error = state->fixed_point_error;
    }

    if (evaluate_now &&
        should_do_adaptive_restart(state, &params->restart_params,
                                   params->termination_evaluation_frequency)) {
      perform_restart(state, params);
      state->initial_fixed_point_error = state->fixed_point_error;
    }

    halpern_update(state, params->reflection_coefficient);

    state->inner_count++;
    state->total_count++;
  }

  if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED) {
    state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
    compute_residual(state, params->optimality_norm);
    display_iteration_stats(state, params->verbose);
  }

  if (params->feasibility_polishing &&
      state->termination_reason != TERMINATION_REASON_DUAL_INFEASIBLE &&
      state->termination_reason != TERMINATION_REASON_PRIMAL_INFEASIBLE) {
    feasibility_polish(params, state);
  }

  cp_al_result_t *result = create_result_from_state(state, original_problem);

  cp_al_final_log(result, params);
  cp_al_solver_state_free(state);
  qp_problem_free(working_problem);
  return result;
}
