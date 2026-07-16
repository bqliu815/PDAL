/*
Copyright 2025 Haihao Lu
Copyright 2026 Hongpei Li
Modifications Copyright 2026 Benqi Liu

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

static const char *get_env_value(const char *primary_name,
                                 const char *fallback_name) {
  const char *value = getenv(primary_name);
  if ((value == NULL || value[0] == '\0') && fallback_name != NULL) {
    value = getenv(fallback_name);
  }
  return value;
}

static int get_restart_env_int(const char *primary_name,
                               const char *fallback_name, int default_value) {
  const char *value = get_env_value(primary_name, fallback_name);
  if (value == NULL || value[0] == '\0') {
    return default_value;
  }
  return atoi(value);
}

static double get_restart_env_double(const char *primary_name,
                                     const char *fallback_name,
                                     double default_value) {
  const char *value = get_env_value(primary_name, fallback_name);
  if (value == NULL || value[0] == '\0') {
    return default_value;
  }
  char *endptr = NULL;
  double parsed = strtod(value, &endptr);
  if (endptr == value || !isfinite(parsed)) {
    return default_value;
  }
  return parsed;
}

__global__ void
halpern_update_kernel(const double *initial_primal, double *current_primal,
                      const double *reflected_primal,
                      const double *initial_dual, double *current_dual,
                      const double *reflected_dual, int n_vars, int n_cons,
                      double weight, double reflection_coeff);
__global__ void rescale_solution_kernel(double *primal_solution,
                                        double *dual_solution,
                                        const double *variable_rescaling,
                                        const double *constraint_rescaling,
                                        const double objective_vector_rescaling,
                                        const double constraint_bound_rescaling,
                                        int n_vars, int n_cons);

__global__ void compute_and_rescale_reduced_cost_kernel(
    double *reduced_cost, const double *objective, const double *dual_product,
    const double *variable_rescaling, const double objective_vector_rescaling,
    const double constraint_bound_rescaling, int n_vars) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vars) {
    reduced_cost[i] = (objective[i] - dual_product[i]) * variable_rescaling[i] /
                      objective_vector_rescaling;
  }
}

__global__ void compute_lp_next_fa_cp_primal_solution_kernel(
    const double *current_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double current_primal_i = current_primal[i];
    double temp =
        current_primal_i - step_size * (objective[i] - dual_product[i]);
    double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
    reflected_primal[i] = 2.0 * temp_proj - current_primal_i;
  }
}

__global__ void compute_lp_next_fa_cp_primal_solution_major_kernel(
    const double *current_primal, double *fa_cp_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size, double *dual_slack) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double current_primal_i = current_primal[i];
    double temp =
        current_primal_i - step_size * (objective[i] - dual_product[i]);
    double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
    reflected_primal[i] = 2.0 * temp_proj - current_primal_i;
    fa_cp_primal[i] = temp_proj;
    dual_slack[i] = (temp_proj - temp) / step_size;
  }
}

__global__ void compute_diagonal_q_next_fa_cp_primal_solution_major_kernel(
    const double *current_primal, double *fa_cp_primal, double *reflected_primal,
    double *objective_product, const double *dual_product,
    const double *objective, const double *var_lb, const double *var_ub, int n,
    double step_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double current_primal_i = current_primal[i];
    double temp =
        (current_primal_i - step_size * (objective[i] - dual_product[i])) /
        (1.0 + step_size * objective_product[i]);
    double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
    reflected_primal[i] = 2.0 * temp_proj - current_primal_i;
    fa_cp_primal[i] = temp_proj;
  }
}

__global__ void compute_diagonal_q_next_fa_cp_primal_solution_kernel(
    const double *current_primal, double *reflected_primal,
    double *objective_product, const double *dual_product,
    const double *objective, const double *var_lb, const double *var_ub, int n,
    double step_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double current_primal_i = current_primal[i];
    double temp =
        (current_primal_i - step_size * (objective[i] - dual_product[i])) /
        (1.0 + step_size * objective_product[i]);
    double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
    reflected_primal[i] = 2.0 * temp_proj - current_primal_i;
  }
}

__global__ void compute_next_fa_cp_dual_solution_kernel(
    const double *current_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb,
    const double *const_ub, int n, double step_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double temp = current_dual[i] / step_size - primal_product[i];
    double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
    reflected_dual[i] = 2.0 * (temp - temp_proj) * step_size - current_dual[i];
  }
}

__global__ void compute_next_fa_cp_dual_solution_major_kernel(
    const double *current_dual, double *fa_cp_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb,
    const double *const_ub, int n, double step_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double temp = current_dual[i] / step_size - primal_product[i];
    double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
    fa_cp_dual[i] = (temp - temp_proj) * step_size;
    reflected_dual[i] = 2.0 * fa_cp_dual[i] - current_dual[i];
  }
}

__global__ void compute_al_effective_dual_kernel(
    const double *primal_product, const double *current_dual,
    double *effective_dual, const double *const_lb, const double *const_ub,
    int n, double sigma) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double shifted = primal_product[i] - current_dual[i] / sigma;
    double projected = fmax(const_lb[i], fmin(shifted, const_ub[i]));
    effective_dual[i] =
        current_dual[i] - sigma * (primal_product[i] - projected);
  }
}

__global__ void compute_next_fa_cp_al_dual_solution_kernel(
    const double *current_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb,
    const double *const_ub, int n, double dual_step_size, double sigma) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double denom = sigma + dual_step_size;
    double shifted = primal_product[i] - current_dual[i] / denom;
    double projected = fmax(const_lb[i], fmin(shifted, const_ub[i]));
    double next_dual =
        current_dual[i] + dual_step_size * (projected - primal_product[i]);
    reflected_dual[i] = 2.0 * next_dual - current_dual[i];
  }
}

__global__ void compute_next_fa_cp_al_dual_solution_major_kernel(
    const double *current_dual, double *fa_cp_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb,
    const double *const_ub, int n, double dual_step_size, double sigma) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double denom = sigma + dual_step_size;
    double shifted = primal_product[i] - current_dual[i] / denom;
    double projected = fmax(const_lb[i], fmin(shifted, const_ub[i]));
    double next_dual =
        current_dual[i] + dual_step_size * (projected - primal_product[i]);
    fa_cp_dual[i] = next_dual;
    reflected_dual[i] = 2.0 * next_dual - current_dual[i];
  }
}

__global__ void
halpern_update_kernel(const double *initial_primal, double *current_primal,
                      const double *reflected_primal,
                      const double *initial_dual, double *current_dual,
                      const double *reflected_dual, int n_vars, int n_cons,
                      double weight, double reflection_coeff) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vars) {
    double reflected = reflection_coeff * reflected_primal[i] +
                       (1.0 - reflection_coeff) * current_primal[i];
    current_primal[i] = weight * reflected + (1.0 - weight) * initial_primal[i];
  } else if (i < n_vars + n_cons) {
    int idx = i - n_vars;
    double reflected = reflection_coeff * reflected_dual[idx] +
                       (1.0 - reflection_coeff) * current_dual[idx];
    current_dual[idx] = weight * reflected + (1.0 - weight) * initial_dual[idx];
  }
}

__global__ void rescale_solution_kernel(double *primal_solution,
                                        double *dual_solution,
                                        const double *variable_rescaling,
                                        const double *constraint_rescaling,
                                        const double objective_vector_rescaling,
                                        const double constraint_bound_rescaling,
                                        int n_vars, int n_cons) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vars) {
    primal_solution[i] =
        primal_solution[i] / variable_rescaling[i] / constraint_bound_rescaling;
  } else if (i < n_vars + n_cons) {
    int idx = i - n_vars;
    dual_solution[idx] = dual_solution[idx] / constraint_rescaling[idx] /
                         objective_vector_rescaling;
  }
}

__global__ void compute_delta_solution_kernel(
    const double *initial_primal, const double *fa_cp_primal,
    double *delta_primal, const double *initial_dual, const double *fa_cp_dual,
    double *delta_dual, int n_vars, int n_cons) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vars) {
    delta_primal[i] = fa_cp_primal[i] - initial_primal[i];
  } else if (i < n_vars + n_cons) {
    int idx = i - n_vars;
    delta_dual[idx] = fa_cp_dual[idx] - initial_dual[idx];
  }
}

static void compute_al_effective_dual_product(fa_cp_solver_state_t *state,
                                              double *primal_solution,
                                              double sigma) {
  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_primal_sol, primal_solution));
  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
      state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

  compute_al_effective_dual_kernel<<<state->num_blocks_dual,
                                     THREADS_PER_BLOCK>>>(
      state->primal_product, state->current_dual_solution,
      state->al_dual_interaction, state->constraint_lower_bound,
      state->constraint_upper_bound, state->num_constraints, sigma);

  CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol,
                                        state->al_dual_interaction));
  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
      state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));
}

void lp_primal_update(fa_cp_solver_state_t *state, double step_size) {
  if (state->is_this_major_iteration ||
      ((state->total_count + 2) %
       get_print_frequency(state->total_count + 2)) == 0) {
    compute_lp_next_fa_cp_primal_solution_major_kernel<<<
        state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->current_primal_solution, state->fa_cp_primal_solution,
        state->reflected_primal_solution, state->dual_product,
        state->objective_vector, state->variable_lower_bound,
        state->variable_upper_bound, state->num_variables, step_size,
        state->dual_slack);
  } else {
    compute_lp_next_fa_cp_primal_solution_kernel<<<state->num_blocks_primal,
                                                  THREADS_PER_BLOCK>>>(
        state->current_primal_solution, state->reflected_primal_solution,
        state->dual_product, state->objective_vector,
        state->variable_lower_bound, state->variable_upper_bound,
        state->num_variables, step_size);
  }
}

void diag_q_primal_update(fa_cp_solver_state_t *state, double step_size) {
  if (state->is_this_major_iteration ||
      ((state->total_count + 2) %
       get_print_frequency(state->total_count + 2)) == 0) {
    compute_diagonal_q_next_fa_cp_primal_solution_major_kernel<<<
        state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->current_primal_solution, state->fa_cp_primal_solution,
        state->reflected_primal_solution,
        state->quadratic_objective_term->diagonal_objective_matrix,
        state->dual_product, state->objective_vector,
        state->variable_lower_bound, state->variable_upper_bound,
        state->num_variables, step_size);
  } else {
    compute_diagonal_q_next_fa_cp_primal_solution_kernel<<<
        state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->current_primal_solution, state->reflected_primal_solution,
        state->quadratic_objective_term->diagonal_objective_matrix,
        state->dual_product, state->objective_vector,
        state->variable_lower_bound, state->variable_upper_bound,
        state->num_variables, step_size);
  }
}

__global__ void primal_gradient_descent_kernel(
    const double *dual_product, const double *current_primal_solution,
    double *reflected_primal, const double *objective_vector,
    const double *objective_product, const double *var_lb, const double *var_ub,
    const double stepsize, const int n_vars) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vars) {
    double current_grad =
        objective_product[i] + objective_vector[i] - dual_product[i];
    double current_primal_sol = current_primal_solution[i];
    double next_primal_sol = current_primal_sol - stepsize * current_grad;
    next_primal_sol = fmax(var_lb[i], fmin(next_primal_sol, var_ub[i]));
    reflected_primal[i] = 2.0 * next_primal_sol - current_primal_sol;
  }
}

__global__ void primal_gradient_descent_kernel_major(
    const double *dual_product, const double *current_primal_solution,
    double *reflected_primal, double *fa_cp_primal_solution,
    const double *objective_vector, const double *objective_product,
    const double *var_lb, const double *var_ub, const double stepsize,
    const int n_vars) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vars) {
    double current_grad =
        objective_product[i] + objective_vector[i] - dual_product[i];
    double current_primal_sol = current_primal_solution[i];
    double next_primal_sol = current_primal_sol - stepsize * current_grad;
    next_primal_sol = fmax(var_lb[i], fmin(next_primal_sol, var_ub[i]));
    fa_cp_primal_solution[i] = next_primal_sol;
    reflected_primal[i] = 2.0 * next_primal_sol - current_primal_sol;
  }
}

__global__ void primal_gradient_descent_kernel_bb_init(
    const double *dual_product, double *gradient, double *direction,
    const double *current_primal_solution, double *fa_cp_primal_solution,
    const double *objective_vector, const double *objective_product,
    const double *var_lb, const double *var_ub, const double stepsize,
    const int n_vars) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vars) {
    double current_grad =
        objective_product[i] + objective_vector[i] - dual_product[i];
    double current_primal_sol = current_primal_solution[i];
    double next_primal_sol = current_primal_sol - stepsize * current_grad;
    next_primal_sol = fmax(var_lb[i], fmin(next_primal_sol, var_ub[i]));
    fa_cp_primal_solution[i] = next_primal_sol;
    gradient[i] = current_grad;
    direction[i] = next_primal_sol - current_primal_sol;
  }
}

__global__ void primal_bb_update_gradient_kernel(
    double *fa_cp_primal_solution, const double *current_primal_solution,
    const double *objective_vector, const double *dual_product,
    const double *objective_product, double *gradient, double *delta_gradient,
    const double inv_step_size, const int n_vars) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vars) {
    double last_gradient = gradient[i];
    double current_gradient =
        objective_product[i] + objective_vector[i] - dual_product[i] +
        inv_step_size * (fa_cp_primal_solution[i] - current_primal_solution[i]);
    delta_gradient[i] = current_gradient - last_gradient;
    gradient[i] = current_gradient;
  }
}

__global__ void
primal_bb_update_direction_kernel(double *fa_cp_primal_solution,
                                  const double *gradient, double *direction,
                                  const double *var_lb, const double *var_ub,
                                  const double alpha, const int n_vars) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vars) {
    double cur_sol = fa_cp_primal_solution[i];
    double next_sol = cur_sol - alpha * gradient[i];
    next_sol = fmax(var_lb[i], fmin(next_sol, var_ub[i]));
    direction[i] = next_sol - cur_sol;
    fa_cp_primal_solution[i] = next_sol;
  }
}

__global__ void primal_bb_final_kernel(const double *current_primal_solution,
                                       const double *fa_cp_primal_solution,
                                       double *reflected_primal_solution,
                                       const int n_vars) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vars) {
    double cur_sol = fa_cp_primal_solution[i];
    double last_sol = current_primal_solution[i];
    reflected_primal_solution[i] = 2.0 * cur_sol - last_sol;
  }
}

void primal_BB_step_size_update(fa_cp_solver_state_t *state, double step_size,
                                double al_sigma) {
  double inv_step_size = 1.0 / step_size;
  int inner_solver_iter = 1;
  double norm_gtg;
  double tmp;
  double alpha = 1.0 / inv_step_size;
  if (state->use_al_qp) {
    compute_al_effective_dual_product(state, state->current_primal_solution,
                                      al_sigma);
  }
  update_obj_product(state, state->current_primal_solution);
  primal_gradient_descent_kernel_bb_init<<<state->num_blocks_primal,
                                           THREADS_PER_BLOCK>>>(
      state->dual_product, state->inner_solver->bb_step_size->gradient,
      state->inner_solver->bb_step_size->direction,
      state->current_primal_solution, state->fa_cp_primal_solution,
      state->objective_vector,
      state->quadratic_objective_term->primal_obj_product,
      state->variable_lower_bound, state->variable_upper_bound, alpha,
      state->num_variables);
  while (inner_solver_iter < state->inner_solver->iteration_limit) {

    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables,
                                   state->inner_solver->bb_step_size->direction,
                                   1, &norm_gtg));
    if (norm_gtg <= state->inner_solver->tol)
      break;
    if (state->use_al_qp) {
      compute_al_effective_dual_product(state, state->fa_cp_primal_solution,
                                        al_sigma);
    }
    update_obj_product(state, state->fa_cp_primal_solution);
    primal_bb_update_gradient_kernel<<<state->num_blocks_primal,
                                       THREADS_PER_BLOCK>>>(
        state->fa_cp_primal_solution, state->current_primal_solution,
        state->objective_vector, state->dual_product,
        state->quadratic_objective_term->primal_obj_product,
        state->inner_solver->bb_step_size->gradient,
        state->inner_solver->primal_buffer, inv_step_size,
        state->num_variables);
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables,
                            state->inner_solver->bb_step_size->direction, 1,
                            state->inner_solver->primal_buffer, 1, &tmp));
    alpha = norm_gtg * norm_gtg / tmp;
    primal_bb_update_direction_kernel<<<state->num_blocks_primal,
                                        THREADS_PER_BLOCK>>>(
        state->fa_cp_primal_solution,
        state->inner_solver->bb_step_size->gradient,
        state->inner_solver->bb_step_size->direction,
        state->variable_lower_bound, state->variable_upper_bound, alpha,
        state->num_variables);
    inner_solver_iter++;
  }
  primal_bb_final_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
      state->current_primal_solution, state->fa_cp_primal_solution,
      state->reflected_primal_solution, state->num_variables);
  state->inner_solver->total_count += (inner_solver_iter - 1);
}

void primal_gradient_update(fa_cp_solver_state_t *state, double step_size) {
  double inv_step_size = 1.0 / step_size;
  double alpha = 1.0 / (state->quadratic_objective_term->norm + inv_step_size);
  update_obj_product(state, state->current_primal_solution);
  if (state->is_this_major_iteration ||
      ((state->total_count + 2) %
       get_print_frequency(state->total_count + 2)) == 0) {
    primal_gradient_descent_kernel_major<<<state->num_blocks_primal,
                                           THREADS_PER_BLOCK>>>(
        state->dual_product, state->current_primal_solution,
        state->reflected_primal_solution, state->fa_cp_primal_solution,
        state->objective_vector,
        state->quadratic_objective_term->primal_obj_product,
        state->variable_lower_bound, state->variable_upper_bound, alpha,
        state->num_variables);
  } else {
    primal_gradient_descent_kernel<<<state->num_blocks_primal,
                                     THREADS_PER_BLOCK>>>(
        state->dual_product, state->current_primal_solution,
        state->reflected_primal_solution, state->objective_vector,
        state->quadratic_objective_term->primal_obj_product,
        state->variable_lower_bound, state->variable_upper_bound, alpha,
        state->num_variables);
  }
}

void fa_cp_update(fa_cp_solver_state_t *state) {
  double primal_step_size = state->step_size / state->primal_weight;
  if (state->quadratic_objective_term->nonconvexity < 0) {
    primal_step_size =
        fmax(primal_step_size,
             -1.01 * fmin(0.0, state->quadratic_objective_term->nonconvexity));
    primal_step_size /= 2;
  }
  double dual_step_size = state->step_size * state->primal_weight;
  double al_sigma = state->use_al_qp ? state->al_sigma : 0.0;

  // Primal Update
  if (state->use_al_qp) {
    // The full AL branch recomputes the interaction inside the primal inner
    // solve, so we do not freeze A^T u^k here.
  } else {
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol,
                                          state->current_dual_solution));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));
  }

  if (state->use_al_qp) {
    primal_BB_step_size_update(state, primal_step_size, al_sigma);
  } else {
    switch (state->quadratic_objective_term->quad_obj_type) {
    case FA_CP_NON_Q: {
      lp_primal_update(state, primal_step_size);
      break;
    }
    case FA_CP_DIAG_Q: {
      diag_q_primal_update(state, primal_step_size);
      break;
    }
    case FA_CP_SPARSE_Q:
    case FA_CP_LOW_RANK_Q:
    case FA_CP_LOW_RANK_PLUS_SPARSE_Q: {
      primal_BB_step_size_update(state, primal_step_size, 0.0);
      break;
    }
    default:
      fprintf(stderr, "Error: Unknown Quadratic Objective Type detected.\n");
      exit(EXIT_FAILURE);
    }
  }

  // Dual Update
  CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol,
                                        state->reflected_primal_solution));
  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
      state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

  if (state->use_al_qp) {
    if (state->is_this_major_iteration ||
        ((state->total_count + 2) %
         get_print_frequency(state->total_count + 2)) == 0) {
      compute_next_fa_cp_al_dual_solution_major_kernel<<<
          state->num_blocks_dual, THREADS_PER_BLOCK>>>(
          state->current_dual_solution, state->fa_cp_dual_solution,
          state->reflected_dual_solution, state->primal_product,
          state->constraint_lower_bound, state->constraint_upper_bound,
          state->num_constraints, dual_step_size, al_sigma);
    } else {
      compute_next_fa_cp_al_dual_solution_kernel<<<state->num_blocks_dual,
                                                  THREADS_PER_BLOCK>>>(
          state->current_dual_solution, state->reflected_dual_solution,
          state->primal_product, state->constraint_lower_bound,
          state->constraint_upper_bound, state->num_constraints,
          dual_step_size, al_sigma);
    }
  } else {
    if (state->is_this_major_iteration ||
        ((state->total_count + 2) %
         get_print_frequency(state->total_count + 2)) == 0) {
      compute_next_fa_cp_dual_solution_major_kernel<<<state->num_blocks_dual,
                                                     THREADS_PER_BLOCK>>>(
          state->current_dual_solution, state->fa_cp_dual_solution,
          state->reflected_dual_solution, state->primal_product,
          state->constraint_lower_bound, state->constraint_upper_bound,
          state->num_constraints, dual_step_size);
    } else {
      compute_next_fa_cp_dual_solution_kernel<<<state->num_blocks_dual,
                                               THREADS_PER_BLOCK>>>(
          state->current_dual_solution, state->reflected_dual_solution,
          state->primal_product, state->constraint_lower_bound,
          state->constraint_upper_bound, state->num_constraints,
          dual_step_size);
    }
  }
}

void halpern_update(fa_cp_solver_state_t *state, double reflection_coefficient) {
  double weight = (double)(state->inner_count + 1) / (state->inner_count + 2);
  halpern_update_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
      state->initial_primal_solution, state->current_primal_solution,
      state->reflected_primal_solution, state->initial_dual_solution,
      state->current_dual_solution, state->reflected_dual_solution,
      state->num_variables, state->num_constraints, weight,
      reflection_coefficient);
}

void rescale_solution(fa_cp_solver_state_t *state) {
  rescale_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
      state->fa_cp_primal_solution, state->fa_cp_dual_solution,
      state->variable_rescaling, state->constraint_rescaling,
      state->objective_vector_rescaling, state->constraint_bound_rescaling,
      state->num_variables, state->num_constraints);
}

static double sane_positive(double value, double fallback) {
  return (isfinite(value) && value > 0.0) ? value : fallback;
}

static double score_component(double value) {
  return isfinite(value) ? fmax(value, 1e-300) : INFINITY;
}

static double clamp_double(double value, double lo, double hi) {
  return fmin(fmax(value, lo), hi);
}

static double sigma_reference_scale(const fa_cp_solver_state_t *state) {
  double ref = state->step_size * state->primal_weight;
  return sane_positive(ref, 1.0);
}

static void set_al_sigma_active_state(fa_cp_solver_state_t *state, double sigma,
                                      double zero_threshold) {
  if (!isfinite(sigma) || sigma <= zero_threshold) {
    state->al_sigma = 0.0;
    state->use_al_qp = false;
  } else {
    state->al_sigma = sigma;
    state->use_al_qp = state->al_qp_available;
  }
}

static bool guarded_normal_motion_ok(double normal_motion,
                                     const fa_cp_parameters_t *params) {
  double threshold =
      sane_positive(params->sigma_normal_motion_threshold, 1e-10);
  return !isfinite(normal_motion) || normal_motion > threshold;
}

static bool guarded_zero_activation_motion_ok(
    double normal_motion, const fa_cp_parameters_t *params) {
  double threshold =
      fmax(sane_positive(params->sigma_normal_motion_threshold, 1e-10), 2e-1);
  return !isfinite(normal_motion) || normal_motion > threshold;
}

static bool guarded_primal_bottleneck(double rp, double rd, double rg,
                                      const fa_cp_parameters_t *params) {
  double threshold = sane_positive(params->sigma_activation_threshold, 2.0);
  double other = fmax(rd, rg);
  return rp >= threshold * sane_positive(other, 1e-300);
}

static double clipped_multiplicative_update(double current, double ratio,
                                            const fa_cp_parameters_t *params) {
  double gain = sane_positive(params->sigma_update_gain, 0.5);
  double clip_low = sane_positive(params->sigma_clip_low, 0.6);
  double clip_high = sane_positive(params->sigma_clip_high, 1.7);
  if (clip_low > clip_high) {
    double tmp = clip_low;
    clip_low = clip_high;
    clip_high = tmp;
  }
  ratio = sane_positive(ratio, 1.0);
  double factor = pow(ratio, gain);
  factor = clamp_double(factor, clip_low, clip_high);
  return current * factor;
}

static double restart_score(const fa_cp_solver_state_t *state) {
  double rp = score_component(state->relative_primal_residual);
  double rd = score_component(state->relative_dual_residual);
  double rg = score_component(state->relative_objective_gap);
  return fmax(rp, fmax(rd, rg));
}

static void update_al_sigma_at_restart(fa_cp_solver_state_t *state,
                                       const fa_cp_parameters_t *params,
                                       double primal_dist, double dual_dist,
                                       double normal_motion) {
  bool balanced_guarded =
      params->sigma_update_rule == FA_CP_SIGMA_UPDATE_BALANCED_GUARDED;
  bool guarded =
      params->sigma_update_rule == FA_CP_SIGMA_UPDATE_GUARDED || balanced_guarded;
  if (!state->al_qp_available ||
      params->sigma_update_rule == FA_CP_SIGMA_UPDATE_FIXED) {
    return;
  }
  if (!guarded && !state->use_al_qp) {
    return;
  }

  double sigma_ref = balanced_guarded ? sigma_reference_scale(state) : 1.0;
  double sigma_min = sane_positive(params->sigma_min, 1e-8) * sigma_ref;
  double sigma_max = sane_positive(params->sigma_max, 10.0) * sigma_ref;
  double zero_threshold =
      sane_positive(params->sigma_zero_threshold,
                    sane_positive(params->sigma_min, 1e-8)) *
      sigma_ref;
  if (sigma_min > sigma_max) {
    double tmp = sigma_min;
    sigma_min = sigma_max;
    sigma_max = tmp;
  }
  if (zero_threshold > sigma_min) {
    sigma_min = zero_threshold;
  }

  double current_sigma =
      state->use_al_qp ? clamp_double(sane_positive(state->al_sigma, sigma_min),
                                      sigma_min, sigma_max)
                       : 0.0;
  set_al_sigma_active_state(state, current_sigma, zero_threshold);

  const double eps = 1e-300;
  double rp = sane_positive(state->relative_primal_residual, eps);
  double rd = sane_positive(state->relative_dual_residual, eps);
  double rg = sane_positive(state->relative_objective_gap, eps);
  double score = restart_score(state);
  if (!isfinite(score)) {
    return;
  }

  if (isfinite(score) &&
      (!isfinite(state->best_sigma_score) || score < state->best_sigma_score)) {
    state->best_sigma_score = score;
    state->best_al_sigma = current_sigma;
  }

  if (params->sigma_rollback_tol > 0.0 && isfinite(score) &&
      (!guarded || current_sigma > zero_threshold) &&
      isfinite(state->best_sigma_score) && state->best_sigma_score > 0.0 &&
      score > params->sigma_rollback_tol * state->best_sigma_score &&
      isfinite(state->best_al_sigma) &&
      (guarded ? state->best_al_sigma >= 0.0 : state->best_al_sigma > 0.0)) {
    double best_sigma = guarded && state->best_al_sigma <= zero_threshold
                            ? 0.0
                            : clamp_double(state->best_al_sigma, sigma_min,
                                           sigma_max);
    set_al_sigma_active_state(state, best_sigma, zero_threshold);
    state->last_sigma_score = score;
    state->last_sigma_primal_residual = rp;
    state->last_sigma_dual_residual = rd;
    state->last_sigma_gap = rg;
    state->sigma_update_count++;
    if (params->verbose >= 2) {
      printf("  sigma restart update: rollback to %.3e (score %.3e)\n",
             state->al_sigma, score);
    }
    return;
  }

  double proposal = current_sigma;
  switch (params->sigma_update_rule) {
  case FA_CP_SIGMA_UPDATE_RESIDUAL_BALANCE:
    proposal = clipped_multiplicative_update(current_sigma, rp / rd, params);
    break;
  case FA_CP_SIGMA_UPDATE_KKT_BALANCE:
    proposal =
        clipped_multiplicative_update(current_sigma, rp / fmax(rd, rg), params);
    break;
  case FA_CP_SIGMA_UPDATE_DISPLACEMENT:
    if (isfinite(primal_dist) && isfinite(dual_dist) && primal_dist > 1e-16 &&
        dual_dist > 1e-16) {
      double target = clamp_double(dual_dist / primal_dist, sigma_min, sigma_max);
      proposal =
          clipped_multiplicative_update(current_sigma, target / current_sigma,
                                        params);
    }
    break;
  case FA_CP_SIGMA_UPDATE_PROGRESS: {
    double last_other =
        fmax(state->last_sigma_dual_residual, state->last_sigma_gap);
    double other = fmax(rd, rg);
    if (isfinite(state->last_sigma_primal_residual) && isfinite(last_other) &&
        state->last_sigma_primal_residual > 0.0 && last_other > 0.0) {
      double primal_progress = state->last_sigma_primal_residual / rp;
      double other_progress = last_other / other;
      proposal = clipped_multiplicative_update(current_sigma,
                                               other_progress / primal_progress,
                                               params);
    } else {
      proposal =
          clipped_multiplicative_update(current_sigma, rp / fmax(rd, rg), params);
    }
    break;
  }
  case FA_CP_SIGMA_UPDATE_GREEDY_SCORE:
    if (state->sigma_update_count == 0) {
      state->sigma_search_direction = (rp >= fmax(rd, rg)) ? 1 : -1;
    } else if (isfinite(score) && isfinite(state->last_sigma_score) &&
               score > state->last_sigma_score) {
      state->sigma_search_direction *= -1;
    }
    proposal = clipped_multiplicative_update(
        current_sigma, state->sigma_search_direction > 0 ? params->sigma_clip_high
                                                         : params->sigma_clip_low,
        params);
    break;
  case FA_CP_SIGMA_UPDATE_GUARDED:
  case FA_CP_SIGMA_UPDATE_BALANCED_GUARDED: {
    bool normal_ok = guarded_normal_motion_ok(normal_motion, params);
    bool zero_motion_ok = guarded_zero_activation_motion_ok(normal_motion, params);
    bool primal_bottleneck = guarded_primal_bottleneck(rp, rd, rg, params);
    bool zero_stalled =
        current_sigma <= zero_threshold && isfinite(state->last_sigma_score) &&
        score > 0.9 * state->last_sigma_score;
    bool stalled_has_primal_signal = rp >= 0.5 * sane_positive(rd, 1e-300);
    double activation_sigma =
        sane_positive(params->sigma_activation_value, 0.1) * sigma_ref;
    bool activation_scale_ok =
        !balanced_guarded || activation_sigma >= 1e-6;
    bool activate =
        current_sigma <= zero_threshold
            ? zero_motion_ok && zero_stalled && stalled_has_primal_signal &&
                  activation_scale_ok
            : normal_ok && primal_bottleneck;
    if (params->verbose >= 2) {
      printf("  guarded sigma check: rp %.3e, rd %.3e, gap %.3e, "
             "normal %.3e, ref %.3e, zero-motion %d, bottleneck %d, "
             "stalled %d, primal-stall %d, scale-ok %d, activate %d\n",
             rp, rd, rg, normal_motion, sigma_ref, zero_motion_ok ? 1 : 0,
             primal_bottleneck ? 1 : 0, zero_stalled ? 1 : 0,
             stalled_has_primal_signal ? 1 : 0, activation_scale_ok ? 1 : 0,
             activate ? 1 : 0);
    }
    if (current_sigma <= zero_threshold) {
      proposal = activate ? activation_sigma : 0.0;
    } else if (!normal_ok) {
      proposal = clipped_multiplicative_update(current_sigma,
                                               params->sigma_clip_low, params);
    } else if (activate) {
      proposal =
          clipped_multiplicative_update(current_sigma, rp / fmax(rd, rg), params);
    } else {
      proposal = clipped_multiplicative_update(current_sigma,
                                               params->sigma_clip_low, params);
    }
    break;
  }
  default:
    break;
  }

  if (proposal > zero_threshold) {
    proposal = clamp_double(proposal, sigma_min, sigma_max);
  } else {
    proposal = 0.0;
  }
  set_al_sigma_active_state(state, proposal, zero_threshold);
  state->last_sigma_score = score;
  state->last_sigma_primal_residual = rp;
  state->last_sigma_dual_residual = rd;
  state->last_sigma_gap = rg;
  state->sigma_update_count++;
  if (params->verbose >= 2) {
    printf("  sigma restart update: %.3e -> %.3e (%s, score %.3e)\n",
           current_sigma, state->al_sigma,
           sigma_update_rule_to_string(params->sigma_update_rule), score);
  }
}

void perform_restart(fa_cp_solver_state_t *state,
                     const fa_cp_parameters_t *params) {
  const int restart_select_mode =
      get_restart_env_int("FA_CP_RESTART_SELECT", "FA_CP_RESTART_SELECT_LEGACY", 0);
  const int restart_select_trace =
      get_restart_env_int("FA_CP_RESTART_SELECT_TRACE",
                          "FA_CP_RESTART_SELECT_TRACE", 0);
  const bool restart_to_halpern =
      restart_select_mode != 0 && state->restart_target == 1;
  const double *restart_primal_solution =
      restart_to_halpern ? state->current_primal_solution
                         : state->fa_cp_primal_solution;
  const double *restart_dual_solution =
      restart_to_halpern ? state->current_dual_solution
                         : state->fa_cp_dual_solution;

  compute_delta_solution_kernel<<<state->num_blocks_primal_dual,
                                  THREADS_PER_BLOCK>>>(
      state->initial_primal_solution, restart_primal_solution,
      state->delta_primal_solution, state->initial_dual_solution,
      restart_dual_solution, state->delta_dual_solution,
      state->num_variables, state->num_constraints);

  double primal_dist, dual_dist;
  CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables,
                                 state->delta_primal_solution, 1,
                                 &primal_dist));
  CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints,
                                 state->delta_dual_solution, 1, &dual_dist));

  double normal_motion = INFINITY;
  if (primal_dist > 1e-16 && isfinite(primal_dist)) {
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol,
                                          state->delta_primal_solution));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));
    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    double normal_dist = 0.0;
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints,
                                   state->primal_product, 1, &normal_dist));
    normal_motion = normal_dist / primal_dist;
  }

  double ratio_infeas =
      state->relative_dual_residual / state->relative_primal_residual;

  if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 &&
      dual_dist < 1e12 && ratio_infeas > 1e-8 && ratio_infeas < 1e8) {
    double error =
        log(dual_dist) - log(primal_dist) - log(state->primal_weight);
    state->primal_weight_error_sum *= params->restart_params.i_smooth;
    state->primal_weight_error_sum += error;
    double delta_error = error - state->primal_weight_last_error;
    state->primal_weight *=
        exp(params->restart_params.k_p * error +
            params->restart_params.k_i * state->primal_weight_error_sum +
            params->restart_params.k_d * delta_error);
    state->primal_weight_last_error = error;
  } else {
    state->primal_weight = state->best_primal_weight;
    state->primal_weight_error_sum = 0.0;
    state->primal_weight_last_error = 0.0;
  }

  double primal_dual_residual_gap = abs(
      log10(state->relative_dual_residual / state->relative_primal_residual));
  if (primal_dual_residual_gap < state->best_primal_dual_residual_gap) {
    state->best_primal_dual_residual_gap = primal_dual_residual_gap;
    state->best_primal_weight = state->primal_weight;
  }

  update_al_sigma_at_restart(state, params, primal_dist, dual_dist,
                             normal_motion);

  CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, restart_primal_solution,
                        state->num_variables * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, restart_dual_solution,
                        state->num_constraints * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  if (restart_to_halpern) {
    CUDA_CHECK(cudaMemcpy(state->fa_cp_primal_solution, restart_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->fa_cp_dual_solution, restart_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    state->restart_halpern_count++;
  } else {
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution, restart_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution, restart_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    state->restart_current_count++;
  }

  if (restart_select_trace != 0) {
    printf("  restart target: %s (current_score %.3e, halpern_score %.3e, "
           "current_count %d, halpern_count %d)\n",
           restart_to_halpern ? "halpern" : "current",
           state->restart_score_current, state->restart_score_halpern,
           state->restart_current_count, state->restart_halpern_count);
  }

  state->inner_count = 0;
  state->last_trial_fixed_point_error = INFINITY;
}

void initialize_step_size_and_primal_weight(fa_cp_solver_state_t *state,
                                            const fa_cp_parameters_t *params) {
  if (state->constraint_matrix->num_nonzeros == 0) {
    state->step_size = 1.0;
  } else {
    double max_sv = estimate_maximum_singular_value(
        state->sparse_handle, state->blas_handle, state->constraint_matrix,
        state->constraint_matrix_t, params->sv_max_iter, params->sv_tol);
    state->step_size = 0.998 / max_sv;
  }

  if (params->bound_objective_rescaling) {
    state->primal_weight = 1.0;
  } else {
    state->primal_weight = (state->objective_vector_norm + 1.0) /
                           (state->constraint_bound_norm + 1.0);
  }
  state->best_primal_weight = state->primal_weight;
}

void compute_fixed_point_error(fa_cp_solver_state_t *state) {
  compute_delta_solution_kernel<<<state->num_blocks_primal_dual,
                                  THREADS_PER_BLOCK>>>(
      state->current_primal_solution, state->reflected_primal_solution,
      state->delta_primal_solution, state->current_dual_solution,
      state->reflected_dual_solution, state->delta_dual_solution,
      state->num_variables, state->num_constraints);

  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_dual_sol, state->delta_dual_solution));
  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
      state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

  double interaction, movement;

  double primal_norm = 0.0;
  double dual_norm = 0.0;
  double cross_term = 0.0;

  CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints,
                                 state->delta_dual_solution, 1, &dual_norm));
  CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables,
                                 state->delta_primal_solution, 1,
                                 &primal_norm));
  movement = primal_norm * primal_norm * state->primal_weight +
             dual_norm * dual_norm / state->primal_weight;

  CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables,
                          state->dual_product, 1, state->delta_primal_solution,
                          1, &cross_term));
  interaction = 2 * state->step_size * cross_term;

  state->fixed_point_error = sqrt(fmax(movement + interaction, 0.0));

  if (state->inner_solver->has_inner_loop) {
    state->inner_solver->tol = fmin(
        state->inner_solver->tol,
        fmax(0.0005 * primal_norm / state->step_size * state->primal_weight,
             1e-9));
    // printf("Update Inner Solver Tolerence to:  %.3e, primal movement is:
    // %.3e\n",state->inner_solver->tol, primal_norm);
  }
}

static double compute_restart_delta_score(fa_cp_solver_state_t *state,
                                          const double *candidate_primal,
                                          const double *reference_primal,
                                          const double *candidate_dual,
                                          const double *reference_dual) {
  compute_delta_solution_kernel<<<state->num_blocks_primal_dual,
                                  THREADS_PER_BLOCK>>>(
      reference_primal, candidate_primal, state->delta_primal_solution,
      reference_dual, candidate_dual, state->delta_dual_solution,
      state->num_variables, state->num_constraints);

  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_dual_sol, state->delta_dual_solution));
  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
      state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

  double primal_norm = 0.0;
  double dual_norm = 0.0;
  double cross_term = 0.0;
  CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables,
                                 state->delta_primal_solution, 1,
                                 &primal_norm));
  CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints,
                                 state->delta_dual_solution, 1, &dual_norm));
  CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables,
                          state->dual_product, 1, state->delta_primal_solution,
                          1, &cross_term));

  const double movement = primal_norm * primal_norm * state->primal_weight +
                          dual_norm * dual_norm / state->primal_weight;
  const double interaction = 2.0 * state->step_size * cross_term;
  return sqrt(fmax(movement + interaction, 0.0));
}

void choose_restart_candidate(fa_cp_solver_state_t *state) {
  const int restart_select_mode =
      get_restart_env_int("FA_CP_RESTART_SELECT", "FA_CP_RESTART_SELECT_LEGACY", 0);
  state->restart_target = 0;
  state->restart_score_current =
      isfinite(state->fixed_point_error) ? state->fixed_point_error : INFINITY;
  state->restart_score_halpern = INFINITY;

  if (restart_select_mode == 0 || state->total_count == 0 ||
      !isfinite(state->restart_score_current)) {
    return;
  }

  const int min_iters = get_restart_env_int(
      "FA_CP_RESTART_SELECT_MIN_ITERS", "FA_CP_RESTART_SELECT_MIN_ITERS", 0);
  const double min_time =
      get_restart_env_double("FA_CP_RESTART_SELECT_MIN_TIME",
                             "FA_CP_RESTART_SELECT_MIN_TIME", 0.0);
  if (state->total_count < min_iters || state->cumulative_time_sec < min_time) {
    return;
  }

  state->restart_score_halpern = compute_restart_delta_score(
      state, state->fa_cp_primal_solution, state->current_primal_solution,
      state->fa_cp_dual_solution, state->current_dual_solution);

  if (!(state->restart_score_current > 0.0)) {
    return;
  }

  const double margin =
      get_restart_env_double("FA_CP_RESTART_SELECT_MARGIN",
                             "FA_CP_RESTART_SELECT_MARGIN", 1.0);
  if (restart_select_mode == 2 ||
      (isfinite(state->restart_score_halpern) &&
       state->restart_score_halpern < margin * state->restart_score_current)) {
    state->restart_target = 1;
    state->fixed_point_error = state->restart_score_halpern;
  }
}

fa_cp_result_t *create_result_from_state(fa_cp_solver_state_t *state,
                                         const qp_problem_t *original_problem) {
  fa_cp_result_t *results =
      (fa_cp_result_t *)safe_calloc(1, sizeof(fa_cp_result_t));

  // Compute reduced cost
  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_dual_sol, state->fa_cp_dual_solution));
  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
      state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

  compute_and_rescale_reduced_cost_kernel<<<state->num_blocks_primal,
                                            THREADS_PER_BLOCK>>>(
      state->dual_slack, state->objective_vector, state->dual_product,
      state->variable_rescaling, state->objective_vector_rescaling,
      state->constraint_bound_rescaling, state->num_variables);

  rescale_solution(state);

  results->primal_solution =
      (double *)safe_malloc(state->num_variables * sizeof(double));
  results->dual_solution =
      (double *)safe_malloc(state->num_constraints * sizeof(double));
  results->reduced_cost =
      (double *)safe_malloc(state->num_variables * sizeof(double));

  CUDA_CHECK(cudaMemcpy(results->primal_solution, state->fa_cp_primal_solution,
                        state->num_variables * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(results->dual_solution, state->fa_cp_dual_solution,
                        state->num_constraints * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(results->reduced_cost, state->dual_slack,
                        state->num_variables * sizeof(double),
                        cudaMemcpyDeviceToHost));

  results->num_variables = original_problem->num_variables;
  results->num_constraints = original_problem->num_constraints;
  results->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;
  results->total_count = state->total_count;
  results->rescaling_time_sec = state->rescaling_time_sec;
  results->cumulative_time_sec = state->cumulative_time_sec;
  results->relative_primal_residual = state->relative_primal_residual;
  results->relative_dual_residual = state->relative_dual_residual;
  results->absolute_primal_residual = state->absolute_primal_residual;
  results->absolute_dual_residual = state->absolute_dual_residual;
  results->primal_objective_value = state->primal_objective_value;
  results->dual_objective_value = state->dual_objective_value;
  results->objective_gap = state->objective_gap;
  results->relative_objective_gap = state->relative_objective_gap;
  results->max_primal_ray_infeasibility = state->max_primal_ray_infeasibility;
  results->max_dual_ray_infeasibility = state->max_dual_ray_infeasibility;
  results->primal_ray_linear_objective = state->primal_ray_linear_objective;
  results->dual_ray_objective = state->dual_ray_objective;
  results->final_al_sigma = state->al_sigma;
  results->sigma_update_count = state->sigma_update_count;
  results->sigma_update_rule =
      state->al_qp_available ? state->sigma_update_rule : FA_CP_SIGMA_UPDATE_FIXED;
  results->termination_reason = state->termination_reason;
  results->feasibility_polishing_time = state->feasibility_polishing_time;
  results->feasibility_iteration = state->feasibility_iteration;
  results->total_inner_count = state->inner_solver->total_count;
  return results;
}

// Feasibility Polishing
void feasibility_polish(const fa_cp_parameters_t *params,
                        fa_cp_solver_state_t *state) {
  clock_t start_time = clock();
  if (state->relative_primal_residual <
          params->termination_criteria.eps_feas_polish_relative &&
      state->relative_dual_residual <
          params->termination_criteria.eps_feas_polish_relative) {
    printf("Skipping feasibility polishing as the solution is already "
           "sufficiently feasible.\n");
    return;
  }
  double original_primal_weight = 0.0;
  if (params->bound_objective_rescaling) {
    original_primal_weight = 1.0;
  } else {
    original_primal_weight = (state->objective_vector_norm + 1.0) /
                             (state->constraint_bound_norm + 1.0);
  }

  // PRIMAL FEASIBILITY POLISHING
  fa_cp_solver_state_t *primal_state =
      initialize_primal_feas_polish_state(state);
  primal_state->primal_weight = original_primal_weight;
  primal_state->best_primal_weight = original_primal_weight;
  primal_feasibility_polish(params, primal_state, state);

  if (primal_state->termination_reason ==
      TERMINATION_REASON_FEAS_POLISH_SUCCESS) {
    CUDA_CHECK(cudaMemcpy(
        state->fa_cp_primal_solution, primal_state->fa_cp_primal_solution,
        state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    state->absolute_primal_residual = primal_state->absolute_primal_residual;
    state->relative_primal_residual = primal_state->relative_primal_residual;
    state->primal_objective_value = primal_state->primal_objective_value;
  }
  state->feasibility_iteration += primal_state->total_count - 1;

  // DUAL FEASIBILITY POLISHING
  fa_cp_solver_state_t *dual_state = initialize_dual_feas_polish_state(state);
  dual_state->primal_weight = original_primal_weight;
  dual_state->best_primal_weight = original_primal_weight;
  dual_feasibility_polish(params, dual_state, state);

  if (dual_state->termination_reason ==
      TERMINATION_REASON_FEAS_POLISH_SUCCESS) {
    CUDA_CHECK(cudaMemcpy(
        state->fa_cp_dual_solution, dual_state->fa_cp_dual_solution,
        state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    state->absolute_dual_residual = dual_state->absolute_dual_residual;
    state->relative_dual_residual = dual_state->relative_dual_residual;
    state->dual_objective_value = dual_state->dual_objective_value;
  }
  state->feasibility_iteration += dual_state->total_count - 1;

  state->objective_gap =
      fabs(state->primal_objective_value - state->dual_objective_value);
  state->relative_objective_gap =
      state->objective_gap / (1.0 + fabs(state->primal_objective_value) +
                              fabs(state->dual_objective_value));

  // FINAL LOGGING
  fa_cp_feas_polish_final_log(primal_state, dual_state, params->verbose);
  primal_feas_polish_state_free(primal_state);
  dual_feas_polish_state_free(dual_state);

  state->feasibility_polishing_time =
      (double)(clock() - start_time) / CLOCKS_PER_SEC;
  return;
}

void primal_feasibility_polish(const fa_cp_parameters_t *params,
                               fa_cp_solver_state_t *state,
                               const fa_cp_solver_state_t *ori_state) {
  print_initial_feas_polish_info(true, params);
  clock_t start_time = clock();
  bool do_restart = false;
  while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED) {
    if ((state->is_this_major_iteration || state->total_count == 0) ||
        (state->total_count % get_print_frequency(state->total_count) == 0)) {
      compute_primal_feas_polish_residual(state, ori_state,
                                          params->optimality_norm);

      state->cumulative_time_sec =
          (double)(clock() - start_time) / CLOCKS_PER_SEC;

      check_feas_polishing_termination_criteria(
          state, &params->termination_criteria, true);
      display_feas_polish_iteration_stats(state, params->verbose, true);
    }

    if ((state->is_this_major_iteration || state->total_count == 0)) {
      do_restart =
          should_do_adaptive_restart(state, &params->restart_params,
                                     params->termination_evaluation_frequency);
      if (do_restart)
        perform_primal_restart(state);
    }

    state->is_this_major_iteration =
        ((state->total_count + 1) % params->termination_evaluation_frequency) ==
        0;

    fa_cp_update(state);

    if (state->is_this_major_iteration || do_restart) {
      compute_primal_fixed_point_error(state);
      if (do_restart) {
        state->initial_fixed_point_error = state->fixed_point_error;
        do_restart = false;
      }
    }
    halpern_update(state, params->reflection_coefficient);

    state->inner_count++;
    state->total_count++;
  }
  return;
}

void dual_feasibility_polish(const fa_cp_parameters_t *params,
                             fa_cp_solver_state_t *state,
                             const fa_cp_solver_state_t *ori_state) {
  print_initial_feas_polish_info(false, params);
  clock_t start_time = clock();
  bool do_restart = false;
  while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED) {
    if ((state->is_this_major_iteration || state->total_count == 0) ||
        (state->total_count % get_print_frequency(state->total_count) == 0)) {
      compute_dual_feas_polish_residual(state, ori_state,
                                        params->optimality_norm);

      state->cumulative_time_sec =
          (double)(clock() - start_time) / CLOCKS_PER_SEC;

      check_feas_polishing_termination_criteria(
          state, &params->termination_criteria, false);
      display_feas_polish_iteration_stats(state, params->verbose, false);
    }

    if ((state->is_this_major_iteration || state->total_count == 0)) {
      do_restart =
          should_do_adaptive_restart(state, &params->restart_params,
                                     params->termination_evaluation_frequency);
      if (do_restart)
        perform_dual_restart(state);
    }

    state->is_this_major_iteration =
        ((state->total_count + 1) % params->termination_evaluation_frequency) ==
        0;

    fa_cp_update(state);

    if (state->is_this_major_iteration || do_restart) {
      compute_dual_fixed_point_error(state);
      if (do_restart) {
        state->initial_fixed_point_error = state->fixed_point_error;
        do_restart = false;
      }
    }
    halpern_update(state, params->reflection_coefficient);

    state->inner_count++;
    state->total_count++;
  }
  return;
}

fa_cp_solver_state_t *
initialize_primal_feas_polish_state(const fa_cp_solver_state_t *original_state) {
  fa_cp_solver_state_t *primal_state =
      (fa_cp_solver_state_t *)safe_malloc(sizeof(fa_cp_solver_state_t));
  *primal_state = *original_state;
  int num_var = original_state->num_variables;
  int num_cons = original_state->num_constraints;

#define ALLOC_ZERO(dest, bytes)                                                \
  CUDA_CHECK(cudaMalloc(&dest, bytes));                                        \
  CUDA_CHECK(cudaMemset(dest, 0, bytes));

  // RESET PROBLEM TO FEASIBILITY PROBLEM
  ALLOC_ZERO(primal_state->objective_vector, num_var * sizeof(double));
  primal_state->objective_constant = 0.0;

#define ALLOC_AND_COPY_DEV(dest, src, bytes)                                   \
  CUDA_CHECK(cudaMalloc(&dest, bytes));                                        \
  CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice));

  // ALLOCATE AND COPY SOLUTION VECTORS
  ALLOC_AND_COPY_DEV(primal_state->initial_primal_solution,
                     original_state->initial_primal_solution,
                     num_var * sizeof(double));
  ALLOC_AND_COPY_DEV(primal_state->current_primal_solution,
                     original_state->current_primal_solution,
                     num_var * sizeof(double));
  ALLOC_AND_COPY_DEV(primal_state->fa_cp_primal_solution,
                     original_state->fa_cp_primal_solution,
                     num_var * sizeof(double));
  ALLOC_AND_COPY_DEV(primal_state->reflected_primal_solution,
                     original_state->reflected_primal_solution,
                     num_var * sizeof(double));
  ALLOC_AND_COPY_DEV(primal_state->primal_product,
                     original_state->primal_product, num_cons * sizeof(double));
  ALLOC_ZERO(primal_state->al_dual_interaction, num_cons * sizeof(double));

  // ALLOC ZERO FOR OTHERS
  ALLOC_ZERO(primal_state->initial_dual_solution, num_cons * sizeof(double));
  ALLOC_ZERO(primal_state->current_dual_solution, num_cons * sizeof(double));
  ALLOC_ZERO(primal_state->fa_cp_dual_solution, num_cons * sizeof(double));
  ALLOC_ZERO(primal_state->reflected_dual_solution, num_cons * sizeof(double));
  ALLOC_ZERO(primal_state->dual_product, num_var * sizeof(double));

  ALLOC_ZERO(primal_state->dual_slack, num_var * sizeof(double));
  ALLOC_ZERO(primal_state->primal_slack, num_cons * sizeof(double));
  ALLOC_ZERO(primal_state->dual_residual, num_var * sizeof(double));
  ALLOC_ZERO(primal_state->primal_residual, num_cons * sizeof(double));
  ALLOC_ZERO(primal_state->delta_primal_solution, num_var * sizeof(double));
  ALLOC_ZERO(primal_state->delta_dual_solution, num_cons * sizeof(double));

  // RESET SCALAR
  primal_state->primal_weight_error_sum = 0.0;
  primal_state->primal_weight_last_error = 0.0;
  primal_state->best_primal_weight = 0.0;
  primal_state->fixed_point_error = INFINITY;
  primal_state->initial_fixed_point_error = INFINITY;
  primal_state->last_trial_fixed_point_error = INFINITY;
  primal_state->step_size = original_state->step_size;
  primal_state->primal_weight = original_state->primal_weight;
  primal_state->is_this_major_iteration = false;
  primal_state->total_count = 0;
  primal_state->inner_count = 0;
  primal_state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
  primal_state->cumulative_time_sec = 0.0;
  primal_state->best_primal_dual_residual_gap = INFINITY;

  // IGNORE DUAL RESIDUAL AND OBJECTIVE GAP
  primal_state->relative_dual_residual = 0.0;
  primal_state->absolute_dual_residual = 0.0;
  primal_state->relative_objective_gap = 0.0;
  primal_state->objective_gap = 0.0;

  return primal_state;
}

void primal_feas_polish_state_free(fa_cp_solver_state_t *state) {
#define SAFE_CUDA_FREE(p)                                                      \
  if ((p) != NULL) {                                                           \
    CUDA_CHECK(cudaFree(p));                                                   \
    (p) = NULL;                                                                \
  }

  if (!state)
    return;
  SAFE_CUDA_FREE(state->objective_vector);
  SAFE_CUDA_FREE(state->initial_primal_solution);
  SAFE_CUDA_FREE(state->current_primal_solution);
  SAFE_CUDA_FREE(state->fa_cp_primal_solution);
  SAFE_CUDA_FREE(state->reflected_primal_solution);
  SAFE_CUDA_FREE(state->dual_product);
  SAFE_CUDA_FREE(state->initial_dual_solution);
  SAFE_CUDA_FREE(state->current_dual_solution);
  SAFE_CUDA_FREE(state->fa_cp_dual_solution);
  SAFE_CUDA_FREE(state->reflected_dual_solution);
  SAFE_CUDA_FREE(state->primal_product);
  SAFE_CUDA_FREE(state->al_dual_interaction);
  SAFE_CUDA_FREE(state->primal_slack);
  SAFE_CUDA_FREE(state->dual_slack);
  SAFE_CUDA_FREE(state->primal_residual);
  SAFE_CUDA_FREE(state->dual_residual);
  SAFE_CUDA_FREE(state->delta_primal_solution);
  SAFE_CUDA_FREE(state->delta_dual_solution);
  free(state);
}

__global__ void zero_finite_value_vectors_kernel(double *__restrict__ vec,
                                                 int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (isfinite(vec[idx]))
      vec[idx] = 0.0;
  }
}

fa_cp_solver_state_t *
initialize_dual_feas_polish_state(const fa_cp_solver_state_t *original_state) {
  fa_cp_solver_state_t *dual_state =
      (fa_cp_solver_state_t *)safe_malloc(sizeof(fa_cp_solver_state_t));
  *dual_state = *original_state;
  int num_var = original_state->num_variables;
  int num_cons = original_state->num_constraints;

#define ALLOC_AND_COPY_DEV(dest, src, bytes)                                   \
  CUDA_CHECK(cudaMalloc(&dest, bytes));                                        \
  CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice));

// RESET PROBLEM TO DUAL FEASIBILITY PROBLEM
#define SET_FINITE_TO_ZERO(vec, n)                                             \
  {                                                                            \
    int threads = 256;                                                         \
    int blocks = (n + threads - 1) / threads;                                  \
    zero_finite_value_vectors_kernel<<<blocks, threads>>>(vec, n);             \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  }

  ALLOC_AND_COPY_DEV(dual_state->constraint_lower_bound,
                     original_state->constraint_lower_bound,
                     num_cons * sizeof(double));
  ALLOC_AND_COPY_DEV(dual_state->constraint_upper_bound,
                     original_state->constraint_upper_bound,
                     num_cons * sizeof(double));
  ALLOC_AND_COPY_DEV(dual_state->variable_lower_bound,
                     original_state->variable_lower_bound,
                     num_var * sizeof(double));
  ALLOC_AND_COPY_DEV(dual_state->variable_upper_bound,
                     original_state->variable_upper_bound,
                     num_var * sizeof(double));

  SET_FINITE_TO_ZERO(dual_state->constraint_lower_bound, num_cons);
  SET_FINITE_TO_ZERO(dual_state->constraint_upper_bound, num_cons);
  SET_FINITE_TO_ZERO(dual_state->variable_lower_bound, num_var);
  SET_FINITE_TO_ZERO(dual_state->variable_upper_bound, num_var);

#define ALLOC_ZERO(dest, bytes)                                                \
  CUDA_CHECK(cudaMalloc(&dest, bytes));                                        \
  CUDA_CHECK(cudaMemset(dest, 0, bytes));

  ALLOC_ZERO(dual_state->constraint_lower_bound_finite_val,
             num_cons * sizeof(double));
  ALLOC_ZERO(dual_state->constraint_upper_bound_finite_val,
             num_cons * sizeof(double));
  ALLOC_ZERO(dual_state->variable_lower_bound_finite_val,
             num_var * sizeof(double));
  ALLOC_ZERO(dual_state->variable_upper_bound_finite_val,
             num_var * sizeof(double));

  // ALLOCATE AND COPY SOLUTION VECTORS
  ALLOC_AND_COPY_DEV(dual_state->initial_dual_solution,
                     original_state->initial_dual_solution,
                     num_cons * sizeof(double));
  ALLOC_AND_COPY_DEV(dual_state->current_dual_solution,
                     original_state->current_dual_solution,
                     num_cons * sizeof(double));
  ALLOC_AND_COPY_DEV(dual_state->fa_cp_dual_solution,
                     original_state->fa_cp_dual_solution,
                     num_cons * sizeof(double));
  ALLOC_AND_COPY_DEV(dual_state->reflected_dual_solution,
                     original_state->reflected_dual_solution,
                     num_cons * sizeof(double));
  ALLOC_AND_COPY_DEV(dual_state->dual_product, original_state->dual_product,
                     num_var * sizeof(double));
  ALLOC_AND_COPY_DEV(dual_state->dual_slack, original_state->dual_slack,
                     num_var * sizeof(double));

  // ALLOC ZERO FOR OTHERS
  ALLOC_ZERO(dual_state->initial_primal_solution, num_var * sizeof(double));
  ALLOC_ZERO(dual_state->current_primal_solution, num_var * sizeof(double));
  ALLOC_ZERO(dual_state->fa_cp_primal_solution, num_var * sizeof(double));
  ALLOC_ZERO(dual_state->reflected_primal_solution, num_var * sizeof(double));
  ALLOC_ZERO(dual_state->primal_product, num_cons * sizeof(double));
  ALLOC_ZERO(dual_state->al_dual_interaction, num_cons * sizeof(double));
  ALLOC_ZERO(dual_state->primal_slack, num_cons * sizeof(double));
  ALLOC_ZERO(dual_state->dual_residual, num_var * sizeof(double));
  ALLOC_ZERO(dual_state->primal_residual, num_cons * sizeof(double));
  ALLOC_ZERO(dual_state->delta_primal_solution, num_var * sizeof(double));
  ALLOC_ZERO(dual_state->delta_dual_solution, num_cons * sizeof(double));

  // RESET SCALAR
  dual_state->primal_weight_error_sum = 0.0;
  dual_state->primal_weight_last_error = 0.0;
  dual_state->best_primal_weight = 0.0;
  dual_state->fixed_point_error = INFINITY;
  dual_state->initial_fixed_point_error = INFINITY;
  dual_state->last_trial_fixed_point_error = INFINITY;
  dual_state->step_size = original_state->step_size;
  dual_state->primal_weight = original_state->primal_weight;
  dual_state->is_this_major_iteration = false;
  dual_state->total_count = 0;
  dual_state->inner_count = 0;
  dual_state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
  dual_state->cumulative_time_sec = 0.0;
  dual_state->best_primal_dual_residual_gap = INFINITY;

  // IGNORE PRIMAL RESIDUAL AND OBJECTIVE GAP
  dual_state->relative_primal_residual = 0.0;
  dual_state->absolute_primal_residual = 0.0;
  dual_state->relative_objective_gap = 0.0;
  dual_state->objective_gap = 0.0;
  return dual_state;
}

void dual_feas_polish_state_free(fa_cp_solver_state_t *state) {
#define SAFE_CUDA_FREE(p)                                                      \
  if ((p) != NULL) {                                                           \
    CUDA_CHECK(cudaFree(p));                                                   \
    (p) = NULL;                                                                \
  }

  if (!state)
    return;
  SAFE_CUDA_FREE(state->constraint_lower_bound);
  SAFE_CUDA_FREE(state->constraint_upper_bound);
  SAFE_CUDA_FREE(state->variable_lower_bound);
  SAFE_CUDA_FREE(state->variable_upper_bound);
  SAFE_CUDA_FREE(state->constraint_lower_bound_finite_val);
  SAFE_CUDA_FREE(state->constraint_upper_bound_finite_val);
  SAFE_CUDA_FREE(state->variable_lower_bound_finite_val);
  SAFE_CUDA_FREE(state->variable_upper_bound_finite_val);

  SAFE_CUDA_FREE(state->initial_primal_solution);
  SAFE_CUDA_FREE(state->current_primal_solution);
  SAFE_CUDA_FREE(state->fa_cp_primal_solution);
  SAFE_CUDA_FREE(state->reflected_primal_solution);

  SAFE_CUDA_FREE(state->dual_product);
  SAFE_CUDA_FREE(state->initial_dual_solution);
  SAFE_CUDA_FREE(state->current_dual_solution);
  SAFE_CUDA_FREE(state->fa_cp_dual_solution);
  SAFE_CUDA_FREE(state->reflected_dual_solution);
  SAFE_CUDA_FREE(state->primal_product);
  SAFE_CUDA_FREE(state->al_dual_interaction);

  SAFE_CUDA_FREE(state->primal_slack);
  SAFE_CUDA_FREE(state->dual_slack);
  SAFE_CUDA_FREE(state->primal_residual);
  SAFE_CUDA_FREE(state->dual_residual);
  SAFE_CUDA_FREE(state->delta_primal_solution);
  SAFE_CUDA_FREE(state->delta_dual_solution);
  free(state);
}

void perform_primal_restart(fa_cp_solver_state_t *state) {
  CUDA_CHECK(cudaMemcpy(
      state->initial_primal_solution, state->fa_cp_primal_solution,
      state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(
      state->current_primal_solution, state->fa_cp_primal_solution,
      state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
  state->inner_count = 0;
  state->last_trial_fixed_point_error = INFINITY;
}

void perform_dual_restart(fa_cp_solver_state_t *state) {
  CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, state->fa_cp_dual_solution,
                        state->num_constraints * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->fa_cp_dual_solution,
                        state->num_constraints * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  state->inner_count = 0;
  state->last_trial_fixed_point_error = INFINITY;
}

__global__ void
compute_delta_primal_solution_kernel(const double *initial_primal,
                                     const double *fa_cp_primal,
                                     double *delta_primal, int n_vars) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vars) {
    delta_primal[i] = fa_cp_primal[i] - initial_primal[i];
  }
}

__global__ void compute_delta_dual_solution_kernel(const double *initial_dual,
                                                   const double *fa_cp_dual,
                                                   double *delta_dual,
                                                   int n_cons) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_cons) {
    delta_dual[i] = fa_cp_dual[i] - initial_dual[i];
  }
}

void compute_primal_fixed_point_error(fa_cp_solver_state_t *state) {
  compute_delta_primal_solution_kernel<<<state->num_blocks_primal,
                                         THREADS_PER_BLOCK>>>(
      state->current_primal_solution, state->reflected_primal_solution,
      state->delta_primal_solution, state->num_variables);
  double primal_norm = 0.0;
  CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables,
                                 state->delta_primal_solution, 1,
                                 &primal_norm));
  state->fixed_point_error = primal_norm * primal_norm * state->primal_weight;
}

void compute_dual_fixed_point_error(fa_cp_solver_state_t *state) {
  compute_delta_dual_solution_kernel<<<state->num_blocks_dual,
                                       THREADS_PER_BLOCK>>>(
      state->current_dual_solution, state->reflected_dual_solution,
      state->delta_dual_solution, state->num_constraints);
  double dual_norm = 0.0;
  CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints,
                                 state->delta_dual_solution, 1, &dual_norm));
  state->fixed_point_error = dual_norm * dual_norm / state->primal_weight;
}
