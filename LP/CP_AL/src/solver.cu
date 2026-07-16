/*
Copyright 2025 Haihao Lu
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

#include "cp_al.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "presolve.h"
#include "solver.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__global__ void build_row_ind(const int *__restrict__ row_ptr, int num_rows, int *__restrict__ row_ind);
__global__ void build_transpose_map(const int *__restrict__ A_row_ind,
                                    const int *__restrict__ A_col_ind,
                                    const int *__restrict__ At_row_ptr,
                                    const int *__restrict__ At_col_ind,
                                    int nnz,
                                    int *__restrict__ A_to_At);
__global__ void fill_finite_bounds_kernel(const double *__restrict__ lower_bound,
                                          const double *__restrict__ upper_bound,
                                          double *__restrict__ lower_bound_finite_val,
                                          double *__restrict__ upper_bound_finite_val,
                                          int num_elements);
__global__ void rescale_solution_kernel(double *__restrict__ primal_solution,
                                        double *__restrict__ dual_solution,
                                        const double *__restrict__ variable_rescaling,
                                        const double *__restrict__ constraint_rescaling,
                                        const double objective_vector_rescaling,
                                        const double constraint_bound_rescaling,
                                        int n_vars,
                                        int n_cons);
__global__ void compute_delta_solution_kernel(const double *__restrict__ initial_primal,
                                              const double *__restrict__ cp_al_primal,
                                              double *__restrict__ delta_primal,
                                              const double *__restrict__ initial_dual,
                                              const double *__restrict__ cp_al_dual,
                                              double *__restrict__ delta_dual,
                                              int n_vars,
                                              int n_cons);
static cp_al_result_t *create_result_from_state(cp_al_solver_state_t *state, const lp_problem_t *original_problem);
static void perform_restart(cp_al_solver_state_t *state, const cp_al_parameters_t *params);
static void initialize_step_size_and_primal_weight(cp_al_solver_state_t *state, const cp_al_parameters_t *params);
static cp_al_solver_state_t *initialize_solver_state(const lp_problem_t *working_problem,
                                                    const cp_al_parameters_t *params);
static void compute_fixed_point_error(cp_al_solver_state_t *state);
static double compute_restart_delta_score(cp_al_solver_state_t *state,
                                          const double *candidate_primal,
                                          const double *reference_primal,
                                          const double *candidate_dual,
                                          const double *reference_dual);
static void choose_restart_candidate(cp_al_solver_state_t *state);
static void refresh_step_schedule(cp_al_solver_state_t *state);
static double get_env_double_or_default(const char *name, double default_value);
static int get_env_int_or_default(const char *name, int default_value);
static int env_var_is_set(const char *name);
static int parse_lambda_grid_env(const char *raw, double *values, int max_values);
static void set_lambda_grid(cp_al_solver_state_t *state, const double *values, int count);
static void initialize_adaptive_lambda(cp_al_solver_state_t *state);
static void update_adaptive_lambda_on_restart(cp_al_solver_state_t *state, bool verbose);
static double monotonic_seconds(void);
void cp_al_solver_state_free(cp_al_solver_state_t *state);
void rescale_info_free(rescale_info_t *info);

static void perform_primal_restart(cp_al_solver_state_t *state);
static void perform_dual_restart(cp_al_solver_state_t *state);
void primal_feasibility_polish(const cp_al_parameters_t *params,
                               cp_al_solver_state_t *state,
                               const cp_al_solver_state_t *ori_state);
void dual_feasibility_polish(const cp_al_parameters_t *params,
                             cp_al_solver_state_t *state,
                              const cp_al_solver_state_t *ori_state);

static double monotonic_seconds(void)
{
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return (double)timestamp.tv_sec + 1e-9 * (double)timestamp.tv_nsec;
}
void primal_feas_polish_state_free(cp_al_solver_state_t *state);
void dual_feas_polish_state_free(cp_al_solver_state_t *state);
void feasibility_polish(const cp_al_parameters_t *params, cp_al_solver_state_t *state);
static void compute_primal_fixed_point_error(cp_al_solver_state_t *state);
static void compute_dual_fixed_point_error(cp_al_solver_state_t *state);
static cp_al_solver_state_t *initialize_primal_feas_polish_state(const cp_al_solver_state_t *original_state);
static cp_al_solver_state_t *initialize_dual_feas_polish_state(const cp_al_solver_state_t *original_state);
__global__ void compute_next_primal_solution_kernel(double *__restrict__ current_primal,
                                                    double *__restrict__ reflected_primal,
                                                    const double *__restrict__ initial_primal,
                                                    const double *__restrict__ dual_product,
                                                    const double *__restrict__ objective,
                                                    const double *__restrict__ var_lb,
                                                    const double *__restrict__ var_ub,
                                                    int n,
                                                    const double *__restrict__ d_step_size,
                                                    const int *__restrict__ d_base_count,
                                                    int k_offset,
                                                    double reflection_coeff);
__global__ void compute_next_primal_solution_major_kernel(double *__restrict__ current_primal,
                                                          double *__restrict__ cp_al_primal,
                                                          double *__restrict__ halpern_primal,
                                                          double *__restrict__ reflected_primal,
                                                          const double *__restrict__ initial_primal,
                                                          const double *__restrict__ dual_product,
                                                          const double *__restrict__ objective,
                                                          const double *__restrict__ var_lb,
                                                          const double *__restrict__ var_ub,
                                                          int n,
                                                          const double *__restrict__ d_step_size,
                                                          double *__restrict__ dual_slack,
                                                          const int *__restrict__ d_base_count,
                                                          int k_offset,
                                                          double reflection_coeff);
__global__ void compute_next_dual_solution_kernel(double *__restrict__ current_dual,
                                                  const double *__restrict__ initial_dual,
                                                  const double *__restrict__ primal_product,
                                                  const double *__restrict__ const_lb,
                                                  const double *__restrict__ const_ub,
                                                  int n,
                                                  const double *__restrict__ d_step_size,
                                                  const double *__restrict__ d_al_penalty,
                                                  const int *__restrict__ d_base_count,
                                                  int k_offset,
                                                  double reflection_coeff);
__global__ void compute_next_dual_solution_major_kernel(double *__restrict__ current_dual,
                                                        double *__restrict__ cp_al_dual,
                                                        double *__restrict__ halpern_dual,
                                                        double *__restrict__ reflected_dual,
                                                        const double *__restrict__ initial_dual,
                                                        const double *__restrict__ primal_product,
                                                        const double *__restrict__ const_lb,
                                                        const double *__restrict__ const_ub,
                                                        int n,
                                                        const double *__restrict__ d_step_size,
                                                        const double *__restrict__ d_al_penalty,
                                                        const int *__restrict__ d_base_count,
                                                        int k_offset,
                                                        double reflection_coeff);
__global__ void compute_box_al_predictor_kernel(double *__restrict__ dual_predictor,
                                                const double *__restrict__ current_dual,
                                                const double *__restrict__ current_primal_product,
                                                const double *__restrict__ const_lb,
                                                const double *__restrict__ const_ub,
                                                const double *__restrict__ d_al_penalty,
                                                int n);
static void compute_next_primal_solution(cp_al_solver_state_t *state,
                                         const int k_offset,
                                         const double reflection_coefficient,
                                         bool is_major);
static void compute_next_dual_solution(cp_al_solver_state_t *state,
                                       const int k_offset,
                                       const double reflection_coefficient,
                                       bool is_major);
static void sync_step_sizes_to_gpu(cp_al_solver_state_t *state);
static void sync_inner_count_to_gpu(cp_al_solver_state_t *state);
static void check_params_validity(const cp_al_parameters_t *params);
static void initialize_sigma_parameters(cp_al_solver_state_t *state);
static int get_trace_log_frequency(void);
static bool should_emit_trace(long long iteration, int trace_log_frequency, int eval_frequency);
static void emit_trace_fp(cp_al_solver_state_t *state, long long iteration);
static void emit_checkpoint_fp(cp_al_solver_state_t *state, long long iteration);
static int get_progress_log_frequency(void);
static bool should_emit_progress(long long iteration, int progress_log_frequency);
static void emit_progress_metrics(cp_al_solver_state_t *state, long long iteration);

#define CP_AL_ALPHA_TRIAL_VECTOR_COUNT 5

typedef struct
{
    bool allocated;
    bool valid;
    cp_al_solver_state_t scalars;
    double *primal[CP_AL_ALPHA_TRIAL_VECTOR_COUNT];
    double *dual[CP_AL_ALPHA_TRIAL_VECTOR_COUNT];
} alpha_trial_snapshot_t;

typedef struct
{
    bool valid;
    int iterations;
    double seconds;
    double fp_rate;
    double primal_rate;
    double dual_rate;
    double gap_rate;
    double kkt_rate;
} alpha_zero_epoch_record_t;

typedef struct
{
    bool enabled;
    bool trace;
    bool force_reject;
    bool trial_active;
    bool epoch_start_valid;
    int trial_started;
    int trial_accepted;
    int trial_rejected;
    int positive_epochs;
    int positive_charged_iterations;
    int history_count;
    alpha_zero_epoch_record_t history[2];
    int epoch_start_total_count;
    double epoch_start_time_sec;
    double epoch_start_primal;
    double epoch_start_dual;
    double epoch_start_gap;
    int trial_start_total_count;
    int trial_start_charged_count;
    double trial_start_time_sec;
    double trial_start_fp;
    double trial_start_primal;
    double trial_start_dual;
    double trial_start_gap;
    int trial_max_iterations;
    double trial_max_seconds;
    alpha_trial_snapshot_t snapshot;
} alpha_trial_controller_t;

typedef enum
{
    ALPHA_DIRECT_PULSE_IDLE = 0,
    ALPHA_DIRECT_PULSE_POSITIVE = 1,
    ALPHA_DIRECT_PULSE_DONE = 2
} alpha_direct_pulse_phase_t;

typedef struct
{
    bool enabled;
    bool trace;
    bool force_pulse;
    alpha_direct_pulse_phase_t phase;
    int started;
    int completed;
    int aborted;
    int gate_evaluated;
    int gate_passed;
    int gate_rejected;
    int pulse_iterations;
    int gate_anchor_charged_count;
    int gate_previous_epoch_iterations;
    int gate_recent_epoch_iterations;
    double gate_delta_fp_rate;
    int anchor_trajectory_count;
    int anchor_charged_count;
    double positive_fp;
    double positive_primal;
    double positive_dual;
    double positive_gap;
    termination_reason_t positive_termination_reason;
} alpha_direct_pulse_controller_t;

static void alpha_trial_snapshot_allocate(alpha_trial_snapshot_t *snapshot,
                                          const cp_al_solver_state_t *state);
static void alpha_trial_snapshot_save(alpha_trial_snapshot_t *snapshot,
                                      cp_al_solver_state_t *state);
static void alpha_trial_snapshot_free(alpha_trial_snapshot_t *snapshot);
static double alpha_trial_snapshot_restore(alpha_trial_snapshot_t *snapshot,
                                           cp_al_solver_state_t *state);
static void alpha_trial_controller_initialize(alpha_trial_controller_t *controller);
static void alpha_trial_set_epoch_start(alpha_trial_controller_t *controller,
                                        const cp_al_solver_state_t *state);
static bool alpha_trial_record_zero_epoch(alpha_trial_controller_t *controller,
                                          const cp_al_solver_state_t *state);
static bool alpha_trial_is_eligible(const alpha_trial_controller_t *controller,
                                    const cp_al_solver_state_t *state,
                                    const cp_al_parameters_t *params,
                                    int charged_count);
static bool alpha_trial_start(alpha_trial_controller_t *controller,
                              cp_al_solver_state_t *state,
                              int charged_count);
static bool alpha_trial_safety_cap_reached(const alpha_trial_controller_t *controller,
                                           const cp_al_solver_state_t *state,
                                           int charged_count);
static bool alpha_trial_accepts(const alpha_trial_controller_t *controller,
                                const cp_al_solver_state_t *state,
                                int charged_count,
                                const char **decision_reason);
static void alpha_trial_log_final(const alpha_trial_controller_t *controller,
                                  const cp_al_solver_state_t *state,
                                  int charged_count);
static void alpha_direct_pulse_initialize(alpha_direct_pulse_controller_t *controller);
static bool alpha_direct_pulse_is_eligible(alpha_direct_pulse_controller_t *controller,
                                         const alpha_trial_controller_t *history,
                                         const cp_al_solver_state_t *state,
                                         const cp_al_parameters_t *params,
                                         int charged_count,
                                         int eval_frequency);
static void alpha_direct_pulse_start(alpha_direct_pulse_controller_t *controller,
                                     cp_al_solver_state_t *state,
                                     int charged_count,
                                     int eval_frequency);
static void alpha_direct_pulse_finish_positive(alpha_direct_pulse_controller_t *controller,
                                               cp_al_solver_state_t *state,
                                               int charged_count);
static void alpha_direct_pulse_log_final(const alpha_direct_pulse_controller_t *controller,
                                         const cp_al_solver_state_t *state,
                                         int charged_count);

static int get_trace_log_frequency(void)
{
    const char *raw = getenv("CP_AL_LOG_FREQ");
    if (!raw || !raw[0])
    {
        return 0;
    }
    const int value = atoi(raw);
    return value > 0 ? value : 0;
}

static bool should_emit_trace(long long iteration, int trace_log_frequency, int eval_frequency)
{
    if (trace_log_frequency <= 0 || iteration <= 0)
    {
        return false;
    }
    if (iteration % trace_log_frequency != 0)
    {
        return false;
    }
    if (eval_frequency > 0 && iteration % eval_frequency == 0)
    {
        return false;
    }
    return true;
}

static void emit_trace_fp(cp_al_solver_state_t *state, long long iteration)
{
    compute_fixed_point_error(state);
    if (isfinite(state->fixed_point_error) && state->fixed_point_error > 0.0)
    {
        printf("trace iter %lld fp %.12e\n", iteration, state->fixed_point_error);
    }
}

static void emit_checkpoint_fp(cp_al_solver_state_t *state, long long iteration)
{
    if (isfinite(state->fixed_point_error) && state->fixed_point_error > 0.0)
    {
        printf("checkpoint iter %lld fp %.12e\n", iteration, state->fixed_point_error);
    }
}

static int get_progress_log_frequency(void)
{
    const char *raw = getenv("CP_AL_PROGRESS_LOG_FREQ");
    if (!raw || !raw[0])
    {
        return 0;
    }
    const int value = atoi(raw);
    return value > 0 ? value : 0;
}

static double get_env_double_or_default(const char *name, double default_value)
{
    const char *raw = getenv(name);
    if (!raw || !raw[0])
    {
        return default_value;
    }
    char *end = NULL;
    const double value = strtod(raw, &end);
    if (end == raw || !isfinite(value))
    {
        return default_value;
    }
    return value;
}

static int get_env_int_or_default(const char *name, int default_value)
{
    const char *raw = getenv(name);
    if (!raw || !raw[0])
    {
        return default_value;
    }
    char *end = NULL;
    const long value = strtol(raw, &end, 10);
    if (end == raw)
    {
        return default_value;
    }
    return (int)value;
}

static int env_var_is_set(const char *name)
{
    const char *raw = getenv(name);
    return raw && raw[0] != '\0';
}

static double alpha_trial_metric(double value)
{
    return fmax(fabs(value), 1e-300);
}

static double alpha_trial_log_rate(double start, double end, int iterations)
{
    if (iterations <= 0 || !isfinite(start) || !isfinite(end))
    {
        return NAN;
    }
    const double rate = log(alpha_trial_metric(start) / alpha_trial_metric(end)) / (double)iterations;
    return isfinite(rate) ? rate : NAN;
}

static double alpha_trial_predict_ratio(double rate, int iterations)
{
    const double exponent = fmax(-700.0, fmin(700.0, -rate * (double)iterations));
    return exp(exponent);
}

static void alpha_trial_snapshot_allocate(alpha_trial_snapshot_t *snapshot,
                                          const cp_al_solver_state_t *state)
{
    if (snapshot->allocated)
    {
        return;
    }
    const size_t primal_bytes = (size_t)state->num_variables * sizeof(double);
    const size_t dual_bytes = (size_t)state->num_constraints * sizeof(double);
    for (int i = 0; i < CP_AL_ALPHA_TRIAL_VECTOR_COUNT; ++i)
    {
        CUDA_CHECK(cudaMalloc(&snapshot->primal[i], primal_bytes));
        CUDA_CHECK(cudaMalloc(&snapshot->dual[i], dual_bytes));
    }
    snapshot->allocated = true;
}

static void alpha_trial_snapshot_save(alpha_trial_snapshot_t *snapshot,
                                      cp_al_solver_state_t *state)
{
    alpha_trial_snapshot_allocate(snapshot, state);
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    snapshot->scalars = *state;
    double *primal[CP_AL_ALPHA_TRIAL_VECTOR_COUNT] = {
        state->initial_primal_solution,
        state->current_primal_solution,
        state->cp_al_primal_solution,
        state->halpern_primal_solution,
        state->reflected_primal_solution};
    double *dual[CP_AL_ALPHA_TRIAL_VECTOR_COUNT] = {
        state->initial_dual_solution,
        state->current_dual_solution,
        state->cp_al_dual_solution,
        state->halpern_dual_solution,
        state->reflected_dual_solution};
    const size_t primal_bytes = (size_t)state->num_variables * sizeof(double);
    const size_t dual_bytes = (size_t)state->num_constraints * sizeof(double);
    for (int i = 0; i < CP_AL_ALPHA_TRIAL_VECTOR_COUNT; ++i)
    {
        CUDA_CHECK(cudaMemcpy(snapshot->primal[i], primal[i], primal_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(snapshot->dual[i], dual[i], dual_bytes, cudaMemcpyDeviceToDevice));
    }
    snapshot->valid = true;
}

static void alpha_trial_snapshot_free(alpha_trial_snapshot_t *snapshot)
{
    if (!snapshot->allocated)
    {
        return;
    }
    for (int i = 0; i < CP_AL_ALPHA_TRIAL_VECTOR_COUNT; ++i)
    {
        if (snapshot->primal[i])
        {
            CUDA_CHECK(cudaFree(snapshot->primal[i]));
        }
        if (snapshot->dual[i])
        {
            CUDA_CHECK(cudaFree(snapshot->dual[i]));
        }
        snapshot->primal[i] = NULL;
        snapshot->dual[i] = NULL;
    }
    snapshot->allocated = false;
    snapshot->valid = false;
}

static double alpha_trial_snapshot_restore(alpha_trial_snapshot_t *snapshot,
                                           cp_al_solver_state_t *state)
{
    if (!snapshot->valid)
    {
        return INFINITY;
    }
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    const clock_t charged_start_time = state->start_time;
    const double charged_elapsed_time = state->cumulative_time_sec;
    *state = snapshot->scalars;
    state->start_time = charged_start_time;
    state->cumulative_time_sec = charged_elapsed_time;

    double *primal[CP_AL_ALPHA_TRIAL_VECTOR_COUNT] = {
        state->initial_primal_solution,
        state->current_primal_solution,
        state->cp_al_primal_solution,
        state->halpern_primal_solution,
        state->reflected_primal_solution};
    double *dual[CP_AL_ALPHA_TRIAL_VECTOR_COUNT] = {
        state->initial_dual_solution,
        state->current_dual_solution,
        state->cp_al_dual_solution,
        state->halpern_dual_solution,
        state->reflected_dual_solution};
    const size_t primal_bytes = (size_t)state->num_variables * sizeof(double);
    const size_t dual_bytes = (size_t)state->num_constraints * sizeof(double);
    for (int i = 0; i < CP_AL_ALPHA_TRIAL_VECTOR_COUNT; ++i)
    {
        CUDA_CHECK(cudaMemcpy(primal[i], snapshot->primal[i], primal_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(dual[i], snapshot->dual[i], dual_bytes, cudaMemcpyDeviceToDevice));
    }

    const double minus_one = -1.0;
    double max_l2_difference = 0.0;
    for (int i = 0; i < CP_AL_ALPHA_TRIAL_VECTOR_COUNT; ++i)
    {
        double difference = 0.0;
        CUBLAS_CHECK(cublasDcopy(state->blas_handle,
                                 state->num_variables,
                                 primal[i],
                                 1,
                                 state->delta_primal_solution,
                                 1));
        CUBLAS_CHECK(cublasDaxpy(state->blas_handle,
                                 state->num_variables,
                                 &minus_one,
                                 snapshot->primal[i],
                                 1,
                                 state->delta_primal_solution,
                                 1));
        CUBLAS_CHECK(cublasDnrm2(state->blas_handle,
                                 state->num_variables,
                                 state->delta_primal_solution,
                                 1,
                                 &difference));
        max_l2_difference = fmax(max_l2_difference, difference);

        CUBLAS_CHECK(cublasDcopy(state->blas_handle,
                                 state->num_constraints,
                                 dual[i],
                                 1,
                                 state->delta_dual_solution,
                                 1));
        CUBLAS_CHECK(cublasDaxpy(state->blas_handle,
                                 state->num_constraints,
                                 &minus_one,
                                 snapshot->dual[i],
                                 1,
                                 state->delta_dual_solution,
                                 1));
        CUBLAS_CHECK(cublasDnrm2(state->blas_handle,
                                 state->num_constraints,
                                 state->delta_dual_solution,
                                 1,
                                 &difference));
        max_l2_difference = fmax(max_l2_difference, difference);
    }
    sync_step_sizes_to_gpu(state);
    sync_inner_count_to_gpu(state);
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    return max_l2_difference;
}

static void alpha_trial_controller_initialize(alpha_trial_controller_t *controller)
{
    memset(controller, 0, sizeof(*controller));
    controller->enabled = get_env_int_or_default("CP_AL_ALPHA_ROLLBACK_TRIAL", 0) != 0;
    controller->trace = get_env_int_or_default("CP_AL_ALPHA_ROLLBACK_TRACE", 0) != 0;
    controller->force_reject = get_env_int_or_default("CP_AL_ALPHA_ROLLBACK_FORCE_REJECT", 0) != 0;
    if (controller->enabled)
    {
        fprintf(stderr,
                "[alpha-trial-config] enabled=1 alpha=2.000000000000e-03 history=2 "
                "fp_gain=0.10 kkt_gain=0.05 component_slack=0.05 force_reject=%d\n",
                controller->force_reject ? 1 : 0);
        fflush(stderr);
    }
}

static void alpha_trial_set_epoch_start(alpha_trial_controller_t *controller,
                                        const cp_al_solver_state_t *state)
{
    controller->epoch_start_valid = true;
    controller->epoch_start_total_count = state->total_count;
    controller->epoch_start_time_sec = state->cumulative_time_sec;
    controller->epoch_start_primal = alpha_trial_metric(state->relative_primal_residual);
    controller->epoch_start_dual = alpha_trial_metric(state->relative_dual_residual);
    controller->epoch_start_gap = alpha_trial_metric(state->relative_objective_gap);
}

static bool alpha_trial_record_zero_epoch(alpha_trial_controller_t *controller,
                                          const cp_al_solver_state_t *state)
{
    if (!controller->epoch_start_valid || controller->trial_active || state->al_lambda > 0.0)
    {
        return false;
    }
    alpha_zero_epoch_record_t record;
    memset(&record, 0, sizeof(record));
    record.iterations = state->total_count - controller->epoch_start_total_count;
    record.seconds = state->cumulative_time_sec - controller->epoch_start_time_sec;
    const double start_kkt = fmax(controller->epoch_start_primal,
                                  fmax(controller->epoch_start_dual, controller->epoch_start_gap));
    const double end_kkt = fmax(alpha_trial_metric(state->relative_primal_residual),
                                fmax(alpha_trial_metric(state->relative_dual_residual),
                                     alpha_trial_metric(state->relative_objective_gap)));
    record.fp_rate = alpha_trial_log_rate(state->initial_fixed_point_error,
                                          state->fixed_point_error,
                                          record.iterations);
    record.primal_rate = alpha_trial_log_rate(controller->epoch_start_primal,
                                              state->relative_primal_residual,
                                              record.iterations);
    record.dual_rate = alpha_trial_log_rate(controller->epoch_start_dual,
                                            state->relative_dual_residual,
                                            record.iterations);
    record.gap_rate = alpha_trial_log_rate(controller->epoch_start_gap,
                                           state->relative_objective_gap,
                                           record.iterations);
    record.kkt_rate = alpha_trial_log_rate(start_kkt, end_kkt, record.iterations);
    record.valid = record.iterations > 0 && record.seconds >= 0.0 &&
        isfinite(record.fp_rate) && isfinite(record.primal_rate) &&
        isfinite(record.dual_rate) && isfinite(record.gap_rate) &&
        isfinite(record.kkt_rate);
    if (!record.valid)
    {
        return false;
    }
    if (controller->history_count < 2)
    {
        controller->history[controller->history_count++] = record;
    }
    else
    {
        controller->history[0] = controller->history[1];
        controller->history[1] = record;
    }
    if (controller->trace)
    {
        fprintf(stderr,
                "[alpha-zero-epoch] iter=%d K=%d sec=%.12e rates(fp=%.12e p=%.12e d=%.12e g=%.12e max=%.12e) history=%d\n",
                state->total_count,
                record.iterations,
                record.seconds,
                record.fp_rate,
                record.primal_rate,
                record.dual_rate,
                record.gap_rate,
                record.kkt_rate,
                controller->history_count);
        fflush(stderr);
    }
    return true;
}

static bool alpha_trial_is_eligible(const alpha_trial_controller_t *controller,
                                    const cp_al_solver_state_t *state,
                                    const cp_al_parameters_t *params,
                                    int charged_count)
{
    if (!controller->enabled || controller->trial_started != 0 ||
        controller->history_count < 2 || charged_count < 10000 ||
        state->cumulative_time_sec < 2.0 ||
        state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
    {
        return false;
    }
    const int history_max_iterations =
        controller->history[0].iterations > controller->history[1].iterations
            ? controller->history[0].iterations
            : controller->history[1].iterations;
    const double history_max_seconds =
        fmax(2.0, fmax(controller->history[0].seconds, controller->history[1].seconds));
    const int remaining_iterations = params->termination_criteria.iteration_limit - charged_count;
    const double remaining_seconds =
        params->termination_criteria.time_sec_limit - state->cumulative_time_sec;
    return history_max_iterations > 0 && remaining_iterations >= 3 * history_max_iterations &&
        remaining_seconds >= 3.0 * history_max_seconds;
}

static bool alpha_trial_start(alpha_trial_controller_t *controller,
                              cp_al_solver_state_t *state,
                              int charged_count)
{
    alpha_trial_snapshot_save(&controller->snapshot, state);
    controller->trial_started = 1;
    controller->trial_active = true;
    controller->trial_start_total_count = state->total_count;
    controller->trial_start_charged_count = charged_count;
    controller->trial_start_time_sec = state->cumulative_time_sec;
    controller->trial_start_primal = alpha_trial_metric(state->relative_primal_residual);
    controller->trial_start_dual = alpha_trial_metric(state->relative_dual_residual);
    controller->trial_start_gap = alpha_trial_metric(state->relative_objective_gap);
    const int history_max_iterations =
        controller->history[0].iterations > controller->history[1].iterations
            ? controller->history[0].iterations
            : controller->history[1].iterations;
    const double history_max_seconds =
        fmax(2.0, fmax(controller->history[0].seconds, controller->history[1].seconds));
    controller->trial_max_iterations = 2 * history_max_iterations;
    controller->trial_max_seconds = 2.0 * history_max_seconds;

    state->al_sigma_mode = false;
    state->adaptive_lambda = false;
    state->al_lambda = 0.002;
    refresh_step_schedule(state);
    sync_step_sizes_to_gpu(state);
    compute_fixed_point_error(state);
    controller->trial_start_fp = alpha_trial_metric(state->fixed_point_error);
    if (!isfinite(controller->trial_start_fp) || controller->trial_start_fp <= 0.0)
    {
        controller->trial_active = false;
        controller->trial_rejected = 1;
        const double restore_difference = alpha_trial_snapshot_restore(&controller->snapshot, state);
        fprintf(stderr,
                "[alpha-trial-restore] reason=invalid-start charged_iter=%d trajectory_iter=%d max_vector_l2=%.12e\n",
                charged_count,
                state->total_count,
                restore_difference);
        fflush(stderr);
        alpha_trial_snapshot_free(&controller->snapshot);
        return false;
    }
    fprintf(stderr,
            "[alpha-trial-start] charged_iter=%d trajectory_iter=%d solve_sec=%.12e alpha=2.000000000000e-03 "
            "fp=%.12e primal=%.12e dual=%.12e gap=%.12e max_K=%d max_sec=%.12e\n",
            charged_count,
            state->total_count,
            state->cumulative_time_sec,
            controller->trial_start_fp,
            controller->trial_start_primal,
            controller->trial_start_dual,
            controller->trial_start_gap,
            controller->trial_max_iterations,
            controller->trial_max_seconds);
    fflush(stderr);
    return true;
}

static bool alpha_trial_safety_cap_reached(const alpha_trial_controller_t *controller,
                                           const cp_al_solver_state_t *state,
                                           int charged_count)
{
    if (!controller->trial_active)
    {
        return false;
    }
    const int charged_trial_iterations = charged_count - controller->trial_start_charged_count;
    const double trial_seconds = state->cumulative_time_sec - controller->trial_start_time_sec;
    return charged_trial_iterations >= controller->trial_max_iterations ||
        trial_seconds >= controller->trial_max_seconds;
}

static bool alpha_trial_accepts(const alpha_trial_controller_t *controller,
                                const cp_al_solver_state_t *state,
                                int charged_count,
                                const char **decision_reason)
{
    const int trial_iterations = state->total_count - controller->trial_start_total_count;
    if (trial_iterations <= 0)
    {
        *decision_reason = "empty-epoch";
        return false;
    }
    const double fp_rate = 0.5 * (controller->history[0].fp_rate + controller->history[1].fp_rate);
    const double primal_rate =
        0.5 * (controller->history[0].primal_rate + controller->history[1].primal_rate);
    const double dual_rate =
        0.5 * (controller->history[0].dual_rate + controller->history[1].dual_rate);
    const double gap_rate =
        0.5 * (controller->history[0].gap_rate + controller->history[1].gap_rate);
    const double kkt_rate =
        0.5 * (controller->history[0].kkt_rate + controller->history[1].kkt_rate);
    const double fp_prediction_ratio = alpha_trial_predict_ratio(fp_rate, trial_iterations);
    const double primal_prediction = controller->trial_start_primal *
        alpha_trial_predict_ratio(primal_rate, trial_iterations);
    const double dual_prediction = controller->trial_start_dual *
        alpha_trial_predict_ratio(dual_rate, trial_iterations);
    const double gap_prediction = controller->trial_start_gap *
        alpha_trial_predict_ratio(gap_rate, trial_iterations);
    const double start_kkt = fmax(controller->trial_start_primal,
                                  fmax(controller->trial_start_dual, controller->trial_start_gap));
    const double kkt_prediction = start_kkt * alpha_trial_predict_ratio(kkt_rate, trial_iterations);
    const double observed_fp_ratio = alpha_trial_metric(state->fixed_point_error) /
        controller->trial_start_fp;
    const double observed_primal = alpha_trial_metric(state->relative_primal_residual);
    const double observed_dual = alpha_trial_metric(state->relative_dual_residual);
    const double observed_gap = alpha_trial_metric(state->relative_objective_gap);
    const double observed_kkt = fmax(observed_primal, fmax(observed_dual, observed_gap));
    const bool finite = isfinite(fp_prediction_ratio) && isfinite(primal_prediction) &&
        isfinite(dual_prediction) && isfinite(gap_prediction) && isfinite(kkt_prediction) &&
        isfinite(observed_fp_ratio) && isfinite(observed_primal) &&
        isfinite(observed_dual) && isfinite(observed_gap) && isfinite(observed_kkt);
    bool accept = finite && observed_fp_ratio <= 0.90 * fp_prediction_ratio &&
        observed_kkt <= 0.95 * kkt_prediction &&
        observed_primal <= 1.05 * primal_prediction &&
        observed_dual <= 1.05 * dual_prediction &&
        observed_gap <= 1.05 * gap_prediction;
    if (controller->force_reject)
    {
        accept = false;
        *decision_reason = "forced-reject";
    }
    else if (!finite)
    {
        *decision_reason = "nonfinite";
    }
    else if (!accept)
    {
        *decision_reason = "acceptance-gate";
    }
    else
    {
        *decision_reason = "accepted";
    }
    fprintf(stderr,
            "[alpha-trial-decision] reason=%s accept=%d charged_iter=%d trajectory_iter=%d K=%d "
            "fp_ratio_obs=%.12e fp_ratio_pred=%.12e kkt_obs=%.12e kkt_pred=%.12e "
            "p_obs=%.12e p_pred=%.12e d_obs=%.12e d_pred=%.12e g_obs=%.12e g_pred=%.12e\n",
            *decision_reason,
            accept ? 1 : 0,
            charged_count,
            state->total_count,
            trial_iterations,
            observed_fp_ratio,
            fp_prediction_ratio,
            observed_kkt,
            kkt_prediction,
            observed_primal,
            primal_prediction,
            observed_dual,
            dual_prediction,
            observed_gap,
            gap_prediction);
    fflush(stderr);
    return accept;
}

static void alpha_trial_log_final(const alpha_trial_controller_t *controller,
                                  const cp_al_solver_state_t *state,
                                  int charged_count)
{
    if (!controller->enabled)
    {
        return;
    }
    fprintf(stderr,
            "[alpha-trial-final] charged_iter=%d trajectory_iter=%d started=%d accepted=%d rejected=%d "
            "positive_epochs=%d positive_charged_iters=%d final_alpha=%.12e final_penalty=%.12e\n",
            charged_count,
            state->total_count,
            controller->trial_started,
            controller->trial_accepted,
            controller->trial_rejected,
            controller->positive_epochs,
            controller->positive_charged_iterations,
            state->al_lambda,
            state->al_penalty);
    fflush(stderr);
}

static void alpha_direct_pulse_initialize(alpha_direct_pulse_controller_t *controller)
{
    memset(controller, 0, sizeof(*controller));
    controller->enabled = get_env_int_or_default("CP_AL_ALPHA_DIRECT_PULSE", 0) != 0;
    controller->trace = get_env_int_or_default("CP_AL_ALPHA_DIRECT_PULSE_TRACE", 0) != 0;
    controller->force_pulse =
        get_env_int_or_default("CP_AL_DETERMINISTIC_PULSE_FORCE", 0) != 0;
    controller->phase = ALPHA_DIRECT_PULSE_IDLE;
    controller->pulse_iterations = 200;
    if (controller->enabled)
    {
        fprintf(stderr,
                "[alpha-direct-pulse-config] enabled=1 alpha=2.000000000000e-03 "
                "pulse_iterations=200 history_epochs=2 deterministic_anchor_iter_ge=10000 "
                "wall_time_features=0 tail_only=1 force_pulse=%d\n",
                controller->force_pulse ? 1 : 0);
        fflush(stderr);
    }
}

static bool alpha_direct_pulse_is_eligible(alpha_direct_pulse_controller_t *controller,
                                         const alpha_trial_controller_t *history,
                                         const cp_al_solver_state_t *state,
                                         const cp_al_parameters_t *params,
                                         int charged_count,
                                         int eval_frequency)
{
    if (!controller->enabled || controller->started != 0 || controller->gate_evaluated != 0 ||
        controller->phase != ALPHA_DIRECT_PULSE_IDLE || history->history_count < 2 ||
        charged_count < 10000 ||
        state->termination_reason != TERMINATION_REASON_UNSPECIFIED || eval_frequency != 200)
    {
        return false;
    }
    const int remaining_iterations = params->termination_criteria.iteration_limit - charged_count;
    for (int i = 0; i < 2; ++i)
    {
        if (!history->history[i].valid || history->history[i].iterations <= 0 ||
            !isfinite(history->history[i].fp_rate))
        {
            return false;
        }
    }
    if (remaining_iterations < 400)
    {
        return false;
    }

    controller->gate_evaluated = 1;
    controller->gate_anchor_charged_count = charged_count;
    controller->gate_previous_epoch_iterations = history->history[0].iterations;
    controller->gate_recent_epoch_iterations = history->history[1].iterations;
    controller->gate_delta_fp_rate =
        history->history[1].fp_rate - history->history[0].fp_rate;
    const bool gate_passed = controller->force_pulse;
    controller->gate_passed = gate_passed ? 1 : 0;
    controller->gate_rejected = gate_passed ? 0 : 1;
    fprintf(stderr,
            "[alpha-deterministic-tail-anchor] charged_iter=%d trajectory_iter=%d "
            "previous_K=%d recent_K=%d previous_fp_rate=%.12e recent_fp_rate=%.12e "
            "delta_fp_rate=%.12e pass=%d force_pulse=%d\n",
            charged_count,
            state->total_count,
            history->history[0].iterations,
            history->history[1].iterations,
            history->history[0].fp_rate,
            history->history[1].fp_rate,
            controller->gate_delta_fp_rate,
            gate_passed ? 1 : 0,
            controller->force_pulse ? 1 : 0);
    fflush(stderr);
    return gate_passed;
}

static void alpha_direct_pulse_start(alpha_direct_pulse_controller_t *controller,
                                     cp_al_solver_state_t *state,
                                     int charged_count,
                                     int eval_frequency)
{
    controller->started = 1;
    controller->phase = ALPHA_DIRECT_PULSE_POSITIVE;
    controller->pulse_iterations = eval_frequency;
    controller->anchor_trajectory_count = state->total_count;
    controller->anchor_charged_count = charged_count;
    state->al_sigma_mode = false;
    state->adaptive_lambda = false;
    state->al_lambda = 0.002;
    refresh_step_schedule(state);
    sync_step_sizes_to_gpu(state);
    fprintf(stderr,
            "[alpha-direct-pulse-start] charged_iter=%d trajectory_iter=%d "
            "alpha=2.000000000000e-03 pulse_iterations=%d\n",
            charged_count,
            state->total_count,
            controller->pulse_iterations);
    fflush(stderr);
}

static void alpha_direct_pulse_finish_positive(alpha_direct_pulse_controller_t *controller,
                                               cp_al_solver_state_t *state,
                                               int charged_count)
{
    if (!controller->enabled || controller->phase != ALPHA_DIRECT_PULSE_POSITIVE)
    {
        return;
    }
    controller->positive_primal = alpha_trial_metric(state->relative_primal_residual);
    controller->positive_dual = alpha_trial_metric(state->relative_dual_residual);
    controller->positive_gap = alpha_trial_metric(state->relative_objective_gap);
    controller->positive_termination_reason = state->termination_reason;

    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    state->al_sigma_mode = false;
    state->adaptive_lambda = false;
    state->al_lambda = 0.0;
    refresh_step_schedule(state);
    sync_step_sizes_to_gpu(state);
    compute_fixed_point_error(state);
    controller->positive_fp = alpha_trial_metric(state->fixed_point_error);
    choose_restart_candidate(state);
    state->termination_reason = controller->positive_termination_reason;
    controller->completed = 1;
    controller->aborted =
        controller->positive_termination_reason != TERMINATION_REASON_UNSPECIFIED &&
        controller->positive_termination_reason != TERMINATION_REASON_OPTIMAL;
    controller->phase = ALPHA_DIRECT_PULSE_DONE;
    fprintf(stderr,
            "[alpha-direct-pulse-complete] charged_iter=%d trajectory_iter=%d "
            "positive_iters=%d fp0map=%.12e primal=%.12e dual=%.12e gap=%.12e "
            "term=%d\n",
            charged_count,
            state->total_count,
            controller->pulse_iterations,
            controller->positive_fp,
            controller->positive_primal,
            controller->positive_dual,
            controller->positive_gap,
            (int)controller->positive_termination_reason);
    fflush(stderr);
}

static void alpha_direct_pulse_log_final(const alpha_direct_pulse_controller_t *controller,
                                       const cp_al_solver_state_t *state,
                                       int charged_count)
{
    if (!controller->enabled)
    {
        return;
    }
    fprintf(stderr,
            "[alpha-direct-pulse-final] charged_iter=%d trajectory_iter=%d started=%d "
            "completed=%d aborted=%d gate_evaluated=%d gate_passed=%d gate_rejected=%d "
            "gate_anchor_charged_iter=%d gate_previous_K=%d gate_recent_K=%d "
            "gate_delta_fp_rate=%.12e positive_charged_iters=%d "
            "final_alpha=%.12e final_penalty=%.12e\n",
            charged_count,
            state->total_count,
            controller->started,
            controller->completed,
            controller->aborted,
            controller->gate_evaluated,
            controller->gate_passed,
            controller->gate_rejected,
            controller->gate_anchor_charged_count,
            controller->gate_previous_epoch_iterations,
            controller->gate_recent_epoch_iterations,
            controller->gate_delta_fp_rate,
            controller->completed ? controller->pulse_iterations : 0,
            state->al_lambda,
            state->al_penalty);
    fflush(stderr);
}

static int parse_lambda_grid_env(const char *raw, double *values, int max_values)
{
    if (!raw || !raw[0] || max_values <= 0)
    {
        return 0;
    }

    char buffer[512];
    strncpy(buffer, raw, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';

    int count = 0;
    char *token = strtok(buffer, ",;: \t");
    while (token && count < max_values)
    {
        char *end = NULL;
        const double value = strtod(token, &end);
        if (end != token && isfinite(value))
        {
            values[count++] = fmin(fmax(value, 0.0), 0.95);
        }
        token = strtok(NULL, ",;: \t");
    }
    return count;
}

static void set_lambda_grid(cp_al_solver_state_t *state, const double *values, int count)
{
    double grid[CP_AL_MAX_LAMBDA_TIERS];
    int n = 0;

    if (count <= 0)
    {
        grid[n++] = 0.0;
    }
    else
    {
        bool has_zero = false;
        for (int i = 0; i < count && n < CP_AL_MAX_LAMBDA_TIERS; ++i)
        {
            const double value = fmin(fmax(values[i], 0.0), 0.95);
            if (value <= 1e-15)
            {
                has_zero = true;
            }
            grid[n++] = value;
        }
        if (!has_zero && n < CP_AL_MAX_LAMBDA_TIERS)
        {
            grid[n++] = 0.0;
        }
    }

    for (int i = 1; i < n; ++i)
    {
        const double key = grid[i];
        int j = i - 1;
        while (j >= 0 && grid[j] > key)
        {
            grid[j + 1] = grid[j];
            --j;
        }
        grid[j + 1] = key;
    }

    int unique_count = 0;
    for (int i = 0; i < n; ++i)
    {
        if (unique_count == 0 || fabs(grid[i] - state->lambda_values[unique_count - 1]) > 1e-15)
        {
            state->lambda_values[unique_count++] = grid[i];
        }
    }
    if (unique_count == 0)
    {
        state->lambda_values[unique_count++] = 0.0;
    }
    state->lambda_num_tiers = unique_count;
    for (int i = unique_count; i < CP_AL_MAX_LAMBDA_TIERS; ++i)
    {
        state->lambda_values[i] = state->lambda_values[unique_count - 1];
    }

    state->lambda_base = state->lambda_values[0];
    state->lambda_low = state->lambda_values[unique_count > 1 ? 1 : 0];
    state->lambda_mid = state->lambda_values[unique_count > 2 ? unique_count / 2 : (unique_count > 1 ? 1 : 0)];
    state->lambda_high = state->lambda_values[unique_count - 1];
}

static bool should_emit_progress(long long iteration, int progress_log_frequency)
{
    if (progress_log_frequency <= 0 || iteration <= 0)
    {
        return false;
    }
    return iteration % progress_log_frequency == 0;
}

static void emit_progress_metrics(cp_al_solver_state_t *state, long long iteration)
{
    const double safe_rpr = fmax(state->relative_primal_residual, 1e-300);
    const double safe_rdr = fmax(state->relative_dual_residual, 1e-300);
    const double residual_balance = fabs(log10(safe_rdr / safe_rpr));
    const double fp = (isfinite(state->fixed_point_error) && state->fixed_point_error > 0.0) ? state->fixed_point_error : NAN;
    printf("progress iter %lld time %.6e rpr %.12e rdr %.12e rgap %.12e fp %.12e pw %.12e pw_best %.12e bal %.12e\n",
           iteration,
           state->cumulative_time_sec,
           state->relative_primal_residual,
           state->relative_dual_residual,
           state->relative_objective_gap,
           fp,
           state->primal_weight,
           state->best_primal_weight,
           residual_balance);
}

cp_al_result_t *optimize(const cp_al_parameters_t *params, lp_problem_t *original_problem)
{
    check_params_validity(params);
    print_initial_info(params, original_problem);

    cp_al_presolve_info_t *presolve_info = NULL;
    const lp_problem_t *working_problem = original_problem;

    if (params->presolve)
    {
        presolve_info = pslp_presolve(original_problem, params);
        if (presolve_info->problem_solved_during_presolve)
        {
            cp_al_result_t *result = create_result_from_presolve(presolve_info, original_problem);
            cp_al_presolve_info_free(presolve_info);
            cp_al_final_log(result, params);
            return result;
        }
        working_problem = presolve_info->reduced_problem;
    }

    cp_al_solver_state_t *state = initialize_solver_state(working_problem, params);
    display_iteration_stats(state, params->verbose);

    initialize_step_size_and_primal_weight(state, params);
    sync_step_sizes_to_gpu(state);

    alpha_trial_controller_t alpha_trial_controller;
    alpha_trial_controller_initialize(&alpha_trial_controller);
    alpha_direct_pulse_controller_t alpha_direct_pulse_controller;
    alpha_direct_pulse_initialize(&alpha_direct_pulse_controller);
    if (alpha_trial_controller.enabled && alpha_direct_pulse_controller.enabled)
    {
        fprintf(stderr,
                "[alpha-controller-error] rollback trial and direct pulse cannot be enabled together\n");
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
    const int alpha_trial_test_trajectory_limit =
        get_env_int_or_default("CP_AL_ALPHA_ROLLBACK_TEST_TRAJECTORY_LIMIT", 0);
    const int alpha_direct_pulse_test_trajectory_limit =
        get_env_int_or_default("CP_AL_ALPHA_DIRECT_PULSE_TEST_TRAJECTORY_LIMIT", 0);
    int charged_count = state->total_count;

    state->start_time = clock();
    bool do_restart = false;
    const int trace_log_frequency = params->verbose ? get_trace_log_frequency() : 0;
    const int progress_log_frequency = params->verbose ? get_progress_log_frequency() : 0;
    const bool use_dense_trace = trace_log_frequency > 0;

    cudaGraphExec_t graphExec = NULL;
    bool graph_created = false;
    bool graph_uses_al_predictor = false;
    int graph_eval_frequency = 0;
    const int base_eval_frequency = params->termination_evaluation_frequency;
    int eval_frequency = base_eval_frequency;
    int dynamic_eval_fast = get_env_int_or_default("CP_AL_DYNAMIC_EVAL_FAST", 50);
    dynamic_eval_fast = dynamic_eval_fast < 3 ? 3 : dynamic_eval_fast;
    dynamic_eval_fast = dynamic_eval_fast > eval_frequency ? eval_frequency : dynamic_eval_fast;
    const bool use_residual_dynamic_eval =
        get_env_int_or_default("CP_AL_DYNAMIC_EVAL", 0) != 0;
    const double dynamic_eval_after_sec =
        get_env_double_or_default("CP_AL_DYNAMIC_EVAL_AFTER_SEC", -1.0);
    const bool use_time_dynamic_eval = dynamic_eval_after_sec > 0.0;
    const int dynamic_eval_after_iters =
        get_env_int_or_default("CP_AL_DYNAMIC_EVAL_AFTER_ITERS", -1);
    const bool use_iteration_dynamic_eval = dynamic_eval_after_iters > 0;
    const bool use_dynamic_eval =
        (use_residual_dynamic_eval || use_time_dynamic_eval || use_iteration_dynamic_eval) &&
        dynamic_eval_fast < eval_frequency;
    const int dynamic_eval_min_iters =
        fmax(get_env_int_or_default("CP_AL_DYNAMIC_EVAL_MIN_ITERS", eval_frequency), eval_frequency);
    const double dynamic_eval_small_res =
        get_env_double_or_default("CP_AL_DYNAMIC_EVAL_SMALL_RES", 1e-6);
    const double dynamic_eval_large_res =
        get_env_double_or_default("CP_AL_DYNAMIC_EVAL_LARGE_RES", 1e-4);
    const double dynamic_eval_ratio =
        get_env_double_or_default("CP_AL_DYNAMIC_EVAL_RATIO", 1e4);
    const int dynamic_eval_bootstrap_iters =
        fmax(get_env_int_or_default("CP_AL_DYNAMIC_EVAL_BOOTSTRAP_ITERS", 0), 0);
    bool dynamic_eval_latched = false;
    const bool dynamic_eval_allowed = true;
    const bool use_checkpoint_diagnostic =
        get_env_int_or_default("CP_AL_CHECKPOINT_DIAGNOSTIC", 0) != 0;
    const bool use_online_diagnostic =
        get_env_int_or_default("CP_AL_ONLINE_DIAGNOSTIC", 0) != 0;
    int checkpoint_diagnostic_count = 0;
    int online_diagnostic_count = 0;
    double checkpoint_diagnostic_block_total = 0.0;
    double checkpoint_diagnostic_eval_total = 0.0;
    if (use_dynamic_eval && dynamic_eval_bootstrap_iters > 0)
    {
        eval_frequency = dynamic_eval_fast;
    }

    while (charged_count < params->termination_criteria.iteration_limit)
    {
        const int current_eval_frequency = eval_frequency;
        double checkpoint_diagnostic_block_start = 0.0;
        if (use_checkpoint_diagnostic)
        {
            CUDA_CHECK(cudaDeviceSynchronize());
            checkpoint_diagnostic_block_start = monotonic_seconds();
        }
        if (do_restart)
        {
            compute_fixed_point_error(state);
            state->initial_fixed_point_error = state->fixed_point_error;
            do_restart = false;
        }

        if (use_dense_trace)
        {
            sync_inner_count_to_gpu(state);
            for (int i = 1; i <= current_eval_frequency; i++)
            {
                const bool is_major = (i == current_eval_frequency);
                compute_next_primal_solution(state, i, params->reflection_coefficient, is_major);
                compute_next_dual_solution(state, i, params->reflection_coefficient, is_major);
                const long long iteration = state->total_count + i;
                if (should_emit_trace(iteration, trace_log_frequency, current_eval_frequency))
                {
                    emit_trace_fp(state, iteration);
                }
            }
        }
        else
        {
            sync_inner_count_to_gpu(state);
            compute_next_primal_solution(state, 1, params->reflection_coefficient, true);
            compute_next_dual_solution(state, 1, params->reflection_coefficient, true);

            const bool needs_al_predictor = state->al_penalty > 0.0;
            if (graph_created &&
                (graph_uses_al_predictor != needs_al_predictor || graph_eval_frequency != current_eval_frequency))
            {
                CUDA_CHECK(cudaGraphExecDestroy(graphExec));
                graphExec = NULL;
                graph_created = false;
            }
            if (!graph_created)
            {
                // Start CUDA graph capture
                cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal);

                for (int i = 2; i <= current_eval_frequency - 1; i++)
                {
                    compute_next_primal_solution(state, i, params->reflection_coefficient, false);
                    compute_next_dual_solution(state, i, params->reflection_coefficient, false);
                }

                compute_next_primal_solution(
                    state, current_eval_frequency, params->reflection_coefficient, true);
                compute_next_dual_solution(
                    state, current_eval_frequency, params->reflection_coefficient, true);
                // end CUDA graph capture

                cudaGraph_t graph;
                CUDA_CHECK(cudaStreamEndCapture(state->stream, &graph));
                CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
                CUDA_CHECK(cudaGraphDestroy(graph));
                graph_created = true;
                graph_uses_al_predictor = needs_al_predictor;
                graph_eval_frequency = current_eval_frequency;
            }
            CUDA_CHECK(cudaGraphLaunch(graphExec, state->stream));
        }
        double checkpoint_diagnostic_block_sec = 0.0;
        double checkpoint_diagnostic_eval_start = 0.0;
        if (use_checkpoint_diagnostic)
        {
            CUDA_CHECK(cudaDeviceSynchronize());
            checkpoint_diagnostic_block_sec = monotonic_seconds() - checkpoint_diagnostic_block_start;
            checkpoint_diagnostic_eval_start = monotonic_seconds();
        }
        compute_fixed_point_error(state);
        choose_restart_candidate(state);

        compute_residual(state, params->optimality_norm);
        state->inner_count += current_eval_frequency;
        state->total_count += current_eval_frequency;
        charged_count += current_eval_frequency;
        if (params->verbose)
        {
            emit_checkpoint_fp(state, state->total_count);
            if (should_emit_progress(state->total_count, progress_log_frequency))
            {
                emit_progress_metrics(state, state->total_count);
            }
        }

        // Logging
        if (state->total_count % get_print_frequency(state->total_count) == 0)
        {
            display_iteration_stats(state, params->verbose);
        }

        // Check Termination
        check_termination_criteria(state, &params->termination_criteria);
        if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED &&
            charged_count >= params->termination_criteria.iteration_limit)
        {
            state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        }
        if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED &&
            alpha_trial_controller.enabled && alpha_trial_test_trajectory_limit > 0 &&
            !alpha_trial_controller.trial_active && alpha_trial_controller.trial_rejected > 0 &&
            state->total_count >= alpha_trial_test_trajectory_limit)
        {
            state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        }
        if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED &&
            alpha_direct_pulse_controller.enabled && alpha_direct_pulse_test_trajectory_limit > 0 &&
            alpha_direct_pulse_controller.phase == ALPHA_DIRECT_PULSE_DONE &&
            state->total_count >= alpha_direct_pulse_test_trajectory_limit)
        {
            state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        }
        if (use_checkpoint_diagnostic)
        {
            CUDA_CHECK(cudaDeviceSynchronize());
            const double checkpoint_diagnostic_eval_sec =
                monotonic_seconds() - checkpoint_diagnostic_eval_start;
            checkpoint_diagnostic_count += 1;
            checkpoint_diagnostic_block_total += checkpoint_diagnostic_block_sec;
            checkpoint_diagnostic_eval_total += checkpoint_diagnostic_eval_sec;
            if (checkpoint_diagnostic_count <= 20 || checkpoint_diagnostic_count % 50 == 0 ||
                state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
            {
                const double total =
                    checkpoint_diagnostic_block_total + checkpoint_diagnostic_eval_total;
                const double share = total > 0.0 ? checkpoint_diagnostic_eval_total / total : 0.0;
                printf("[checkpoint-diagnostic] iter=%d eval_frequency=%d block_sec=%.12e "
                       "checkpoint_sec=%.12e cumulative_checkpoint_share=%.12e\n",
                       state->total_count,
                       current_eval_frequency,
                       checkpoint_diagnostic_block_sec,
                       checkpoint_diagnostic_eval_sec,
                       share);
            }
        }
        if (alpha_direct_pulse_controller.phase == ALPHA_DIRECT_PULSE_POSITIVE)
        {
            alpha_direct_pulse_finish_positive(
                &alpha_direct_pulse_controller, state, charged_count);
            if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED &&
                alpha_direct_pulse_test_trajectory_limit > 0 &&
                state->total_count >= alpha_direct_pulse_test_trajectory_limit)
            {
                state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
            }
        }
        if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
        {
            if (alpha_trial_controller.trial_active)
            {
                const int positive_iterations =
                    charged_count - alpha_trial_controller.trial_start_charged_count;
                alpha_trial_controller.positive_charged_iterations += positive_iterations;
                alpha_trial_controller.positive_epochs += positive_iterations > 0 ? 1 : 0;
                if (state->termination_reason == TERMINATION_REASON_OPTIMAL)
                {
                    alpha_trial_controller.trial_accepted += 1;
                    alpha_trial_controller.trial_active = false;
                    state->al_lambda = 0.0;
                    refresh_step_schedule(state);
                    sync_step_sizes_to_gpu(state);
                    fprintf(stderr,
                            "[alpha-trial-decision] reason=terminal-optimal accept=1 charged_iter=%d "
                            "trajectory_iter=%d K=%d\n",
                            charged_count,
                            state->total_count,
                            state->total_count - alpha_trial_controller.trial_start_total_count);
                    fflush(stderr);
                    alpha_trial_snapshot_free(&alpha_trial_controller.snapshot);
                }
                else
                {
                    const termination_reason_t charged_reason = state->termination_reason;
                    const double restore_difference =
                        alpha_trial_snapshot_restore(&alpha_trial_controller.snapshot, state);
                    state->termination_reason = charged_reason;
                    alpha_trial_controller.trial_rejected += 1;
                    alpha_trial_controller.trial_active = false;
                    fprintf(stderr,
                            "[alpha-trial-restore] reason=charged-termination charged_iter=%d "
                            "trajectory_iter=%d max_vector_l2=%.12e\n",
                            charged_count,
                            state->total_count,
                            restore_difference);
                    fflush(stderr);
                    alpha_trial_snapshot_free(&alpha_trial_controller.snapshot);
                }
            }
            break;
        }

        if (alpha_trial_safety_cap_reached(&alpha_trial_controller, state, charged_count))
        {
            const int positive_iterations =
                charged_count - alpha_trial_controller.trial_start_charged_count;
            alpha_trial_controller.positive_charged_iterations += positive_iterations;
            alpha_trial_controller.positive_epochs += positive_iterations > 0 ? 1 : 0;
            const double restore_difference =
                alpha_trial_snapshot_restore(&alpha_trial_controller.snapshot, state);
            alpha_trial_controller.trial_rejected += 1;
            alpha_trial_controller.trial_active = false;
            alpha_trial_set_epoch_start(&alpha_trial_controller, state);
            fprintf(stderr,
                    "[alpha-trial-restore] reason=safety-cap charged_iter=%d trajectory_iter=%d "
                    "positive_iters=%d max_vector_l2=%.12e\n",
                    charged_count,
                    state->total_count,
                    positive_iterations,
                    restore_difference);
            fflush(stderr);
            alpha_trial_snapshot_free(&alpha_trial_controller.snapshot);
            do_restart = true;
            continue;
        }

        const bool sigma_trigger_enabled =
            !state->al_sigma_mode && get_env_int_or_default("CP_AL_SIGMA_TRIGGER", 0) != 0;
        bool sigma_trigger_restart = false;
        if (sigma_trigger_enabled &&
            state->total_count >= get_env_int_or_default("CP_AL_SIGMA_TRIGGER_MIN_ITERS", 200))
        {
            const double rpr = fmax(state->relative_primal_residual, 1e-300);
            const double rdr = fmax(state->relative_dual_residual, 1e-300);
            const double trigger_primal =
                get_env_double_or_default("CP_AL_SIGMA_TRIGGER_PRIMAL", 1e-2);
            const double trigger_dual =
                get_env_double_or_default("CP_AL_SIGMA_TRIGGER_DUAL", 1e-6);
            const double trigger_ratio =
                get_env_double_or_default("CP_AL_SIGMA_TRIGGER_RATIO", 1e6);
            if (rpr >= trigger_primal && rdr <= trigger_dual && rpr / rdr >= trigger_ratio)
            {
                initialize_sigma_parameters(state);
                state->adaptive_lambda = false;
                sigma_trigger_restart = true;
                if (params->verbose)
                {
                    printf("sigma-trigger iter %d rpr %.12e rdr %.12e ratio %.12e sigma %.12e base %.12e\n",
                           state->total_count,
                           state->relative_primal_residual,
                           state->relative_dual_residual,
                           rpr / rdr,
                           state->al_sigma,
                           state->al_sigma_base);
                }
            }
        }

        if (use_dynamic_eval && dynamic_eval_allowed && !state->al_sigma_mode &&
            (state->total_count >= dynamic_eval_min_iters ||
             (dynamic_eval_bootstrap_iters > 0 && state->total_count >= dynamic_eval_bootstrap_iters)))
        {
            const double rpr = fmax(state->relative_primal_residual, 1e-300);
            const double rdr = fmax(state->relative_dual_residual, 1e-300);
            const double min_res = fmin(rpr, rdr);
            const double max_res = fmax(rpr, rdr);
            const double residual_ratio = max_res / min_res;
            const bool one_sided_residual =
                min_res <= dynamic_eval_small_res && max_res >= dynamic_eval_large_res;
            const bool ratio_lag =
                residual_ratio >= dynamic_eval_ratio && max_res >= dynamic_eval_large_res;
            const bool time_gate =
                use_time_dynamic_eval && !state->cr_early_stopped &&
                state->cumulative_time_sec >= dynamic_eval_after_sec;
            const bool iteration_gate =
                use_iteration_dynamic_eval && !state->cr_early_stopped &&
                state->total_count >= dynamic_eval_after_iters;
            if (time_gate || iteration_gate ||
                (use_residual_dynamic_eval && (one_sided_residual || ratio_lag)))
            {
                dynamic_eval_latched = true;
                if (eval_frequency != dynamic_eval_fast)
                {
                    eval_frequency = dynamic_eval_fast;
                    if (params->verbose || use_time_dynamic_eval || use_iteration_dynamic_eval)
                    {
                        printf("[dynamic-eval] iter=%d solve_sec=%.12e eval_frequency=%d->%d "
                               "relative_primal=%.12e relative_dual=%.12e residual_ratio=%.12e reason=%s\n",
                               state->total_count,
                               state->cumulative_time_sec,
                               current_eval_frequency,
                               eval_frequency,
                               state->relative_primal_residual,
                               state->relative_dual_residual,
                               residual_ratio,
                               time_gate ? "time" : iteration_gate ? "iteration" : "residual");
                    }
                }
            }
            else if (!dynamic_eval_latched && dynamic_eval_bootstrap_iters > 0 &&
                     eval_frequency == dynamic_eval_fast &&
                     state->total_count >= dynamic_eval_bootstrap_iters)
            {
                eval_frequency = base_eval_frequency;
                if (params->verbose)
                {
                    printf("dynamic-eval iter %d eval_freq %d->%d rpr %.12e rdr %.12e ratio %.12e\n",
                           state->total_count,
                           current_eval_frequency,
                           eval_frequency,
                           state->relative_primal_residual,
                           state->relative_dual_residual,
                           residual_ratio);
                }
            }
        }

        // Check Adaptive Restart
        do_restart =
            sigma_trigger_restart ||
            should_do_adaptive_restart(state, &params->restart_params, current_eval_frequency);
        if (!do_restart && state->adaptive_lambda && state->lambda_race && state->lambda_force_restart_iters > 0 &&
            state->total_count > 0 && state->total_count - state->lambda_last_restart_iter >= state->lambda_force_restart_iters)
        {
            do_restart = true;
            if (params->verbose)
            {
                printf("lambda-race artificial restart iter %d fp %.12e\n",
                       state->total_count,
                       state->fixed_point_error);
            }
        }
        if (use_online_diagnostic)
        {
            online_diagnostic_count += 1;
            if (online_diagnostic_count <= 20 || online_diagnostic_count % 50 == 0 || do_restart)
            {
                const double initial_fp = fmax(state->initial_fixed_point_error, 1e-300);
                const double fp_ratio = state->fixed_point_error / initial_fp;
                const double kkt_max = fmax(state->relative_primal_residual,
                                            fmax(state->relative_dual_residual,
                                                 state->relative_objective_gap));
                printf("[online-diagnostic] iter=%d epoch_iters=%d eval_frequency=%d "
                       "fp=%.12e initial_fp=%.12e fp_ratio=%.12e "
                       "relative_primal=%.12e relative_dual=%.12e relative_gap=%.12e "
                       "kkt_max=%.12e restart=%d step_size=%.12e primal_weight=%.12e\n",
                       state->total_count,
                       state->inner_count,
                       current_eval_frequency,
                       state->fixed_point_error,
                       state->initial_fixed_point_error,
                       fp_ratio,
                       state->relative_primal_residual,
                       state->relative_dual_residual,
                       state->relative_objective_gap,
                       kkt_max,
                       do_restart ? 1 : 0,
                       state->step_size,
                       state->primal_weight);
            }
        }
        if (do_restart)
        {
            if (params->verbose)
            {
                printf("restart iter %d fp %.12e\n", state->total_count, state->fixed_point_error);
            }

            if (alpha_trial_controller.trial_active)
            {
                const int positive_iterations =
                    charged_count - alpha_trial_controller.trial_start_charged_count;
                alpha_trial_controller.positive_charged_iterations += positive_iterations;
                alpha_trial_controller.positive_epochs += positive_iterations > 0 ? 1 : 0;
                const char *decision_reason = NULL;
                const bool accepted = alpha_trial_accepts(
                    &alpha_trial_controller, state, charged_count, &decision_reason);
                if (!accepted)
                {
                    const double restore_difference =
                        alpha_trial_snapshot_restore(&alpha_trial_controller.snapshot, state);
                    alpha_trial_controller.trial_rejected += 1;
                    alpha_trial_controller.trial_active = false;
                    alpha_trial_set_epoch_start(&alpha_trial_controller, state);
                    fprintf(stderr,
                            "[alpha-trial-restore] reason=%s charged_iter=%d trajectory_iter=%d "
                            "positive_iters=%d max_vector_l2=%.12e\n",
                            decision_reason,
                            charged_count,
                            state->total_count,
                            positive_iterations,
                            restore_difference);
                    fflush(stderr);
                    alpha_trial_snapshot_free(&alpha_trial_controller.snapshot);
                    do_restart = true;
                    continue;
                }
                alpha_trial_controller.trial_accepted += 1;
                alpha_trial_controller.trial_active = false;
                state->al_lambda = 0.0;
                refresh_step_schedule(state);
                alpha_trial_snapshot_free(&alpha_trial_controller.snapshot);
            }
            else
            {
                alpha_trial_record_zero_epoch(&alpha_trial_controller, state);
            }
            perform_restart(state, params);
            sync_step_sizes_to_gpu(state);
            alpha_trial_set_epoch_start(&alpha_trial_controller, state);
            if (alpha_trial_is_eligible(&alpha_trial_controller, state, params, charged_count))
            {
                alpha_trial_start(&alpha_trial_controller, state, charged_count);
            }
            if (alpha_direct_pulse_is_eligible(&alpha_direct_pulse_controller,
                                             &alpha_trial_controller,
                                             state,
                                             params,
                                             charged_count,
                                             current_eval_frequency))
            {
                alpha_direct_pulse_start(&alpha_direct_pulse_controller,
                                       state,
                                       charged_count,
                                       current_eval_frequency);
            }
        }
    }

    if (graphExec)
    {
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    }

    if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_residual(state, params->optimality_norm);
        display_iteration_stats(state, params->verbose);
    }

    alpha_trial_log_final(&alpha_trial_controller, state, charged_count);
    alpha_trial_snapshot_free(&alpha_trial_controller.snapshot);
    alpha_direct_pulse_log_final(&alpha_direct_pulse_controller, state, charged_count);

    if (params->feasibility_polishing && state->termination_reason != TERMINATION_REASON_DUAL_INFEASIBLE &&
        state->termination_reason != TERMINATION_REASON_PRIMAL_INFEASIBLE)
    {
        feasibility_polish(params, state);
    }

    const int trajectory_count = state->total_count;
    state->total_count = charged_count;
    cp_al_result_t *result = create_result_from_state(state, original_problem);
    state->total_count = trajectory_count;

    if (params->presolve && presolve_info)
    {
        pslp_postsolve(presolve_info, result, original_problem);
        cp_al_presolve_info_free(presolve_info);
    }

    cp_al_final_log(result, params);
    cp_al_solver_state_free(state);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

static void sync_step_sizes_to_gpu(cp_al_solver_state_t *state)
{
    double current_primal_step = state->step_size / state->primal_weight;
    double current_dual_step = state->step_size * state->primal_weight;

    CUDA_CHECK(cudaMemcpyAsync(
        state->d_primal_step_size, &current_primal_step, sizeof(double), cudaMemcpyHostToDevice, state->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        state->d_dual_step_size, &current_dual_step, sizeof(double), cudaMemcpyHostToDevice, state->stream));
    CUDA_CHECK(
        cudaMemcpyAsync(state->d_al_penalty, &state->al_penalty, sizeof(double), cudaMemcpyHostToDevice, state->stream));
}

static void sync_inner_count_to_gpu(cp_al_solver_state_t *state)
{
    CUDA_CHECK(
        cudaMemcpyAsync(state->d_inner_count, &state->inner_count, sizeof(int), cudaMemcpyHostToDevice, state->stream));
}

static void check_params_validity(const cp_al_parameters_t *params)
{
    if (params->termination_evaluation_frequency < 3)
    {
        fprintf(stderr,
                "Error: termination_evaluation_frequency must be >= 3 (got %d).\n",
                params->termination_evaluation_frequency);
        exit(EXIT_FAILURE);
    }
}

__global__ void compute_and_rescale_reduced_cost_kernel(double *__restrict__ reduced_cost,
                                                        const double *__restrict__ objective,
                                                        const double *__restrict__ dual_product,
                                                        const double *__restrict__ variable_rescaling,
                                                        const double objective_vector_rescaling,
                                                        const double constraint_bound_rescaling,
                                                        const double *__restrict__ variable_lower_bound,
                                                        const double *__restrict__ variable_upper_bound,
                                                        int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double rc = (objective[i] - dual_product[i]) * variable_rescaling[i] / objective_vector_rescaling;

        if (!isfinite(variable_lower_bound[i]))
        {
            rc = fmin(rc, 0.0);
        }
        if (!isfinite(variable_upper_bound[i]))
        {
            rc = fmax(rc, 0.0);
        }
        reduced_cost[i] = rc;
    }
}

static cp_al_solver_state_t *initialize_solver_state(const lp_problem_t *working_problem,
                                                    const cp_al_parameters_t *params)
{
    cp_al_solver_state_t *state = (cp_al_solver_state_t *)safe_calloc(1, sizeof(cp_al_solver_state_t));

    int n_vars = working_problem->num_variables;
    int n_cons = working_problem->num_constraints;
    int nnz = working_problem->constraint_matrix_num_nonzeros;
    size_t var_bytes = n_vars * sizeof(double);
    size_t con_bytes = n_cons * sizeof(double);

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->objective_constant = working_problem->objective_constant;

    state->constraint_matrix = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix_t = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));

    state->constraint_matrix->num_rows = n_cons;
    state->constraint_matrix->num_cols = n_vars;
    state->constraint_matrix->num_nonzeros = nnz;

    state->constraint_matrix_t->num_rows = n_vars;
    state->constraint_matrix_t->num_cols = n_cons;
    state->constraint_matrix_t->num_nonzeros = nnz;

    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;

    state->num_blocks_primal = (state->num_variables + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_dual = (state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_primal_dual =
        (state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_nnz = (state->constraint_matrix->num_nonzeros + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle));
    CUBLAS_CHECK(cublasCreate(&state->blas_handle));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));

#define ALLOC_AND_COPY(dest, src, bytes)                                                                               \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice));

    ALLOC_AND_COPY(
        state->constraint_matrix->row_ptr, working_problem->constraint_matrix_row_pointers, (n_cons + 1) * sizeof(int));
    ALLOC_AND_COPY(state->constraint_matrix->col_ind,
                   working_problem->constraint_matrix_col_indices,
                   working_problem->constraint_matrix_num_nonzeros * sizeof(int));
    ALLOC_AND_COPY(state->constraint_matrix->val,
                   working_problem->constraint_matrix_values,
                   working_problem->constraint_matrix_num_nonzeros * sizeof(double));

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix->row_ind, nnz * sizeof(int)));
    build_row_ind<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_matrix->row_ptr, n_cons, state->constraint_matrix->row_ind);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ptr, (n_vars + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->col_ind,
                          working_problem->constraint_matrix_num_nonzeros * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(&state->constraint_matrix_t->val, working_problem->constraint_matrix_num_nonzeros * sizeof(double)));

    size_t buffer_size = 0;
    void *buffer = nullptr;
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(state->sparse_handle,
                                                 state->constraint_matrix->num_rows,
                                                 state->constraint_matrix->num_cols,
                                                 state->constraint_matrix->num_nonzeros,
                                                 state->constraint_matrix->val,
                                                 state->constraint_matrix->row_ptr,
                                                 state->constraint_matrix->col_ind,
                                                 state->constraint_matrix_t->val,
                                                 state->constraint_matrix_t->row_ptr,
                                                 state->constraint_matrix_t->col_ind,
                                                 CUDA_R_64F,
                                                 CUSPARSE_ACTION_NUMERIC,
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                                 &buffer_size));
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

    CUSPARSE_CHECK(cusparseCsr2cscEx2(state->sparse_handle,
                                      state->constraint_matrix->num_rows,
                                      state->constraint_matrix->num_cols,
                                      state->constraint_matrix->num_nonzeros,
                                      state->constraint_matrix->val,
                                      state->constraint_matrix->row_ptr,
                                      state->constraint_matrix->col_ind,
                                      state->constraint_matrix_t->val,
                                      state->constraint_matrix_t->row_ptr,
                                      state->constraint_matrix_t->col_ind,
                                      CUDA_R_64F,
                                      CUSPARSE_ACTION_NUMERIC,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                      buffer));

    CUDA_CHECK(cudaFree(buffer));

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ind, nnz * sizeof(int)));
    build_row_ind<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->constraint_matrix_t->row_ptr, n_vars, state->constraint_matrix_t->row_ind);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix->transpose_map, nnz * sizeof(int)));
    state->constraint_matrix_t->transpose_map = NULL;
    build_transpose_map<<<state->num_blocks_nnz, THREADS_PER_BLOCK>>>(state->constraint_matrix->row_ind,
                                                                      state->constraint_matrix->col_ind,
                                                                      state->constraint_matrix_t->row_ptr,
                                                                      state->constraint_matrix_t->col_ind,
                                                                      nnz,
                                                                      state->constraint_matrix->transpose_map);
    CUDA_CHECK(cudaGetLastError());

    ALLOC_AND_COPY(state->variable_lower_bound, working_problem->variable_lower_bound, var_bytes);
    ALLOC_AND_COPY(state->variable_upper_bound, working_problem->variable_upper_bound, var_bytes);
    ALLOC_AND_COPY(state->objective_vector, working_problem->objective_vector, var_bytes);
    ALLOC_AND_COPY(state->constraint_lower_bound, working_problem->constraint_lower_bound, con_bytes);
    ALLOC_AND_COPY(state->constraint_upper_bound, working_problem->constraint_upper_bound, con_bytes);

#define ALLOC_ZERO(dest, bytes)                                                                                        \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

    ALLOC_ZERO(state->initial_primal_solution, var_bytes);
    ALLOC_ZERO(state->current_primal_solution, var_bytes);
    ALLOC_ZERO(state->cp_al_primal_solution, var_bytes);
    ALLOC_ZERO(state->halpern_primal_solution, var_bytes);
    ALLOC_ZERO(state->reflected_primal_solution, var_bytes);
    ALLOC_ZERO(state->dual_product, var_bytes);
    ALLOC_ZERO(state->dual_slack, var_bytes);
    ALLOC_ZERO(state->dual_residual, var_bytes);
    ALLOC_ZERO(state->delta_primal_solution, var_bytes);

    ALLOC_ZERO(state->initial_dual_solution, con_bytes);
    ALLOC_ZERO(state->current_dual_solution, con_bytes);
    ALLOC_ZERO(state->cp_al_dual_solution, con_bytes);
    ALLOC_ZERO(state->halpern_dual_solution, con_bytes);
    ALLOC_ZERO(state->reflected_dual_solution, con_bytes);
    ALLOC_ZERO(state->primal_product, con_bytes);
    ALLOC_ZERO(state->primal_slack, con_bytes);
    ALLOC_ZERO(state->primal_residual, con_bytes);
    ALLOC_ZERO(state->delta_dual_solution, con_bytes);

    if (working_problem->primal_start)
    {
        CUDA_CHECK(cudaMemcpy(
            state->initial_primal_solution, working_problem->primal_start, var_bytes, cudaMemcpyHostToDevice));
    }
    if (working_problem->dual_start)
    {
        CUDA_CHECK(
            cudaMemcpy(state->initial_dual_solution, working_problem->dual_start, con_bytes, cudaMemcpyHostToDevice));
    }

    rescale_info_t *rescale_info = rescale_problem(params, state);

    state->constraint_rescaling = rescale_info->con_rescale;
    state->variable_rescaling = rescale_info->var_rescale;
    state->constraint_bound_rescaling = rescale_info->con_bound_rescale;
    state->objective_vector_rescaling = rescale_info->obj_vec_rescale;
    state->rescaling_time_sec = rescale_info->rescaling_time_sec;

    rescale_info->con_rescale = NULL;
    rescale_info->var_rescale = NULL;
    rescale_info_free(rescale_info);

    CUDA_CHECK(cudaMemcpy(
        state->current_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(state->current_dual_solution, state->initial_dual_solution, con_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(state->cp_al_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->halpern_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(state->cp_al_dual_solution, state->initial_dual_solution, con_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(state->halpern_dual_solution, state->initial_dual_solution, con_bytes, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMalloc(&state->constraint_lower_bound_finite_val, con_bytes));
    CUDA_CHECK(cudaMalloc(&state->constraint_upper_bound_finite_val, con_bytes));
    CUDA_CHECK(cudaMalloc(&state->variable_lower_bound_finite_val, var_bytes));
    CUDA_CHECK(cudaMalloc(&state->variable_upper_bound_finite_val, var_bytes));

    fill_finite_bounds_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(state->constraint_lower_bound,
                                                                             state->constraint_upper_bound,
                                                                             state->constraint_lower_bound_finite_val,
                                                                             state->constraint_upper_bound_finite_val,
                                                                             n_cons);

    fill_finite_bounds_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(state->variable_lower_bound,
                                                                               state->variable_upper_bound,
                                                                               state->variable_lower_bound_finite_val,
                                                                               state->variable_upper_bound_finite_val,
                                                                               n_vars);

    CUDA_CHECK(cudaFree(state->constraint_matrix->row_ind));
    state->constraint_matrix->row_ind = NULL;
    CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ind));
    state->constraint_matrix_t->row_ind = NULL;
    CUDA_CHECK(cudaFree(state->constraint_matrix->transpose_map));
    state->constraint_matrix->transpose_map = NULL;

    double sum_of_squares = 0.0;
    double max_val = 0.0;
    double val = 0.0;
    for (int i = 0; i < n_vars; ++i)
    {
        if (params->optimality_norm == NORM_TYPE_L_INF)
        {
            val = fabs(working_problem->objective_vector[i]);
            if (val > max_val)
                max_val = val;
        }
        else
        {
            sum_of_squares += working_problem->objective_vector[i] * working_problem->objective_vector[i];
        }
    }

    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        state->objective_vector_norm = max_val;
    }
    else
    {
        state->objective_vector_norm = sqrt(sum_of_squares);
    }

    sum_of_squares = 0.0;
    max_val = 0.0;
    val = 0.0;
    for (int i = 0; i < n_cons; ++i)
    {
        double lower = working_problem->constraint_lower_bound[i];
        double upper = working_problem->constraint_upper_bound[i];

        if (params->optimality_norm == NORM_TYPE_L_INF)
        {
            if (isfinite(lower) && (lower != upper))
            {
                val = fabs(lower);
                if (val > max_val)
                    max_val = val;
            }
            if (isfinite(upper))
            {
                val = fabs(upper);
                if (val > max_val)
                    max_val = val;
            }
        }
        else
        {
            if (isfinite(lower) && (lower != upper))
            {
                sum_of_squares += lower * lower;
            }
            if (isfinite(upper))
            {
                sum_of_squares += upper * upper;
            }
        }
    }
    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        state->constraint_bound_norm = max_val;
    }
    else
    {
        state->constraint_bound_norm = sqrt(sum_of_squares);
    }

    state->best_primal_dual_residual_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size = 0.0;
    state->al_lambda = 0.0;
    state->al_penalty = 0.0;
    state->al_sigma_mode = false;
    state->al_sigma_adaptive = false;
    state->al_sigma = 0.0;
    state->al_sigma_base = 0.0;
    state->al_sigma_min = 0.0;
    state->al_sigma_max = 0.0;
    state->al_sigma_best = 0.0;
    state->al_sigma_error_sum = 0.0;
    state->al_sigma_last_error = 0.0;
    state->al_lambda_param = 0.0;
    state->lambda_est = 1.0;
    state->is_this_major_iteration = false;
    state->adaptive_lambda = false;
    state->adaptive_lambda_metrics_ready = false;
    state->lambda_num_tiers = 0;
    for (int i = 0; i < CP_AL_MAX_LAMBDA_TIERS; ++i)
    {
        state->lambda_values[i] = 0.0;
    }
    state->lambda_tier = 0;
    state->lambda_hold_remaining = 0;
    state->lambda_hold_epochs = 0;
    state->lambda_restart_count = 0;
    state->lambda_min_restarts = 0;
    state->lambda_min_iterations = 0;
    state->lambda_force_restart_iters = 0;
    state->lambda_last_restart_iter = 0;
    state->lambda_race = false;
    state->lambda_race_next_tier = 1;
    state->lambda_race_best_tier = 0;
    state->lambda_race_exploit_remaining = 0;
    state->lambda_race_exploit_epochs = 0;
    state->lambda_race_cooldown_remaining = 0;
    state->lambda_race_cooldown_epochs = 1000000000;
    state->lambda_race_best_score = -INFINITY;
    state->lambda_race_cycle_best_tier = 0;
    state->lambda_race_cycle_best_score = -INFINITY;
    state->lambda_race_bad_ratio = 50.0;
    state->lambda_base = 0.0;
    state->lambda_low = 0.10;
    state->lambda_mid = 0.10;
    state->lambda_high = 0.10;
    state->lambda_min_time_sec = 0.0;
    state->lambda_feas_plateau_tol = 0.98;
    state->lambda_gap_plateau_tol = 0.95;
    state->lambda_fp_plateau_tol = 0.98;
    state->lambda_feas_active_tol = 1e-6;
    state->lambda_gap_active_tol = 1e-6;
    state->lambda_fp_active_tol = 1e-8;
    state->lambda_gap_dominance = 100.0;
    state->lambda_epoch_feas_start = INFINITY;
    state->lambda_epoch_gap_start = INFINITY;
    state->lambda_epoch_fp_start = INFINITY;
    state->lambda_epoch_time_start = 0.0;
    state->restart_target = 0;
    state->restart_score_current = INFINITY;
    state->restart_score_average = INFINITY;
    state->restart_score_halpern = INFINITY;

    size_t primal_spmv_buffer_size;
    size_t dual_spmv_buffer_size;

    CUSPARSE_CHECK(cusparseCreateCsr(&state->matA,
                                     state->num_constraints,
                                     state->num_variables,
                                     state->constraint_matrix->num_nonzeros,
                                     state->constraint_matrix->row_ptr,
                                     state->constraint_matrix->col_ind,
                                     state->constraint_matrix->val,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));
    CUDA_CHECK(cudaGetLastError());

    CUSPARSE_CHECK(cusparseCreateCsr(&state->matAt,
                                     state->num_variables,
                                     state->num_constraints,
                                     state->constraint_matrix_t->num_nonzeros,
                                     state->constraint_matrix_t->row_ptr,
                                     state->constraint_matrix_t->col_ind,
                                     state->constraint_matrix_t->val,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));
    CUDA_CHECK(cudaGetLastError());

    CUSPARSE_CHECK(
        cusparseCreateDnVec(&state->vec_primal_sol, state->num_variables, state->cp_al_primal_solution, CUDA_R_64F));
    CUSPARSE_CHECK(
        cusparseCreateDnVec(&state->vec_dual_sol, state->num_constraints, state->cp_al_dual_solution, CUDA_R_64F));
    CUSPARSE_CHECK(
        cusparseCreateDnVec(&state->vec_primal_prod, state->num_constraints, state->primal_product, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_prod, state->num_variables, state->dual_product, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           state->matA,
                                           state->vec_primal_sol,
                                           &HOST_ZERO,
                                           state->vec_primal_prod,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           &primal_spmv_buffer_size));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           state->matAt,
                                           state->vec_dual_sol,
                                           &HOST_ZERO,
                                           state->vec_dual_prod,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           &dual_spmv_buffer_size));
    CUDA_CHECK(cudaMalloc(&state->primal_spmv_buffer, primal_spmv_buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(state->sparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           state->matA,
                                           state->vec_primal_sol,
                                           &HOST_ZERO,
                                           state->vec_primal_prod,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           state->primal_spmv_buffer));

    CUDA_CHECK(cudaMalloc(&state->dual_spmv_buffer, dual_spmv_buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(state->sparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           state->matAt,
                                           state->vec_dual_sol,
                                           &HOST_ZERO,
                                           state->vec_dual_prod,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           state->dual_spmv_buffer));

    CUDA_CHECK(cudaMalloc(&state->d_primal_step_size, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_dual_step_size, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_al_penalty, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_inner_count, sizeof(int)));

    CUDA_CHECK(cudaMemset(state->d_primal_step_size, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(state->d_dual_step_size, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(state->d_al_penalty, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(state->d_inner_count, 0, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&state->ones_primal_d, state->num_variables * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->ones_dual_d, state->num_constraints * sizeof(double)));

    double *ones_primal_h = (double *)safe_malloc(state->num_variables * sizeof(double));
    for (int i = 0; i < state->num_variables; ++i)
        ones_primal_h[i] = 1.0;
    CUDA_CHECK(
        cudaMemcpy(state->ones_primal_d, ones_primal_h, state->num_variables * sizeof(double), cudaMemcpyHostToDevice));
    free(ones_primal_h);

    double *ones_dual_h = (double *)safe_malloc(state->num_constraints * sizeof(double));
    for (int i = 0; i < state->num_constraints; ++i)
        ones_dual_h[i] = 1.0;
    CUDA_CHECK(
        cudaMemcpy(state->ones_dual_d, ones_dual_h, state->num_constraints * sizeof(double), cudaMemcpyHostToDevice));

    // --- CUDA Graph Initialization ---
    CUDA_CHECK(cudaStreamCreate(&state->stream));
    CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle, state->stream));
    CUBLAS_CHECK(cublasSetStream(state->blas_handle, state->stream));

    free(ones_dual_h);
    if (params->verbose)
    {
        printf("---------------------------------------------------------------------"
               "------------------\n");
        printf("%s | %s | %s | %s | %s \n",
               "   runtime    ",
               "    objective     ",
               "  absolute residuals   ",
               "  relative residuals   ",
               " fixed-point ");
        printf("%s %s | %s %s | %s %s %s | %s %s %s | %s \n",
               "  iter",
               "  time ",
               " pr obj ",
               "  du obj ",
               " pr res",
               " du res",
               "  gap  ",
               " pr res",
               " du res",
               "  gap  ",
               " fp err ");
        printf("---------------------------------------------------------------------"
               "------------------\n");
    }

    return state;
}

__global__ void build_row_ind(const int *__restrict__ row_ptr, int num_rows, int *__restrict__ row_ind)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows)
        return;

    int s = row_ptr[i];
    int e = row_ptr[i + 1];

    for (int k = s; k < e; ++k)
    {
        row_ind[k] = i;
    }
}

__global__ void build_transpose_map(const int *__restrict__ A_row_ind,
                                    const int *__restrict__ A_col_ind,
                                    const int *__restrict__ At_row_ptr,
                                    const int *__restrict__ At_col_ind,
                                    int nnz,
                                    int *__restrict__ A_to_At)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz)
        return;

    int i = A_row_ind[k];
    int j = A_col_ind[k];

    int start = At_row_ptr[j];
    int end = At_row_ptr[j + 1];

    int pos = -1;
    for (int idx = start; idx < end; ++idx)
    {
        if (At_col_ind[idx] == i)
        {
            pos = idx;
            break;
        }
    }

    if (pos < 0)
        return;

    A_to_At[k] = pos;
}

__global__ void fill_finite_bounds_kernel(const double *__restrict__ lb,
                                          const double *__restrict__ ub,
                                          double *__restrict__ lb_finite,
                                          double *__restrict__ ub_finite,
                                          int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements)
        return;

    double Li = lb[i];
    double Ui = ub[i];

    lb_finite[i] = isfinite(Li) ? Li : 0.0;
    ub_finite[i] = isfinite(Ui) ? Ui : 0.0;
}

__global__ void compute_next_primal_solution_kernel(double *__restrict__ current_primal,
                                                    double *__restrict__ reflected_primal,
                                                    const double *__restrict__ initial_primal,
                                                    const double *__restrict__ dual_product,
                                                    const double *__restrict__ objective,
                                                    const double *__restrict__ var_lb,
                                                    const double *__restrict__ var_ub,
                                                    int n,
                                                    const double *__restrict__ d_step_size,
                                                    const int *__restrict__ d_base_count,
                                                    int k_offset,
                                                    double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double step_size = *d_step_size;
    int current_k = *d_base_count + k_offset;
    double weight = (double)(current_k) / (double)(current_k + 1);
    if (i < n)
    {
        double temp = current_primal[i] - step_size * (objective[i] - dual_product[i]);
        double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
        reflected_primal[i] = 2.0 * temp_proj - current_primal[i];
        double reflected = reflection_coeff * reflected_primal[i] + (1.0 - reflection_coeff) * current_primal[i];
        current_primal[i] = weight * reflected + (1.0 - weight) * initial_primal[i];
    }
}

__global__ void compute_next_primal_solution_major_kernel(double *__restrict__ current_primal,
                                                          double *__restrict__ cp_al_primal,
                                                          double *__restrict__ halpern_primal,
                                                          double *__restrict__ reflected_primal,
                                                          const double *__restrict__ initial_primal,
                                                          const double *__restrict__ dual_product,
                                                          const double *__restrict__ objective,
                                                          const double *__restrict__ var_lb,
                                                          const double *__restrict__ var_ub,
                                                          int n,
                                                          const double *__restrict__ d_step_size,
                                                          double *__restrict__ dual_slack,
                                                          const int *__restrict__ d_base_count,
                                                          int k_offset,
                                                          double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double step_size = *d_step_size;
    int current_k = *d_base_count + k_offset;
    double weight = (double)(current_k) / (double)(current_k + 1);
    if (i < n)
    {
        double temp = current_primal[i] - step_size * (objective[i] - dual_product[i]);
        cp_al_primal[i] = fmax(var_lb[i], fmin(temp, var_ub[i]));
        dual_slack[i] = (cp_al_primal[i] - temp) / step_size;
        halpern_primal[i] = weight * cp_al_primal[i] + (1.0 - weight) * initial_primal[i];
        reflected_primal[i] = 2.0 * cp_al_primal[i] - current_primal[i];
        double reflected = reflection_coeff * reflected_primal[i] + (1.0 - reflection_coeff) * current_primal[i];
        current_primal[i] = weight * reflected + (1.0 - weight) * initial_primal[i];
    }
}

__global__ void compute_next_dual_solution_kernel(double *__restrict__ current_dual,
                                                  const double *__restrict__ initial_dual,
                                                  const double *__restrict__ primal_product,
                                                  const double *__restrict__ const_lb,
                                                  const double *__restrict__ const_ub,
                                                  int n,
                                                  const double *__restrict__ d_step_size,
                                                  const double *__restrict__ d_al_penalty,
                                                  const int *__restrict__ d_base_count,
                                                  int k_offset,
                                                  double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double step_size = *d_step_size;
    double al_penalty = *d_al_penalty;
    double prox_step = step_size + al_penalty;
    int current_k = *d_base_count + k_offset;
    double weight = (double)(current_k) / (double)(current_k + 1);
    if (i < n)
    {
        double z_bar = fmax(const_lb[i], fmin(primal_product[i] - current_dual[i] / prox_step, const_ub[i]));
        double y_next = current_dual[i] - step_size * (primal_product[i] - z_bar);
        double reflected = reflection_coeff * (2.0 * y_next - current_dual[i]) +
            (1.0 - reflection_coeff) * current_dual[i];
        current_dual[i] = weight * reflected + (1.0 - weight) * initial_dual[i];
    }
}

__global__ void compute_next_dual_solution_major_kernel(double *__restrict__ current_dual,
                                                        double *__restrict__ cp_al_dual,
                                                        double *__restrict__ halpern_dual,
                                                        double *__restrict__ reflected_dual,
                                                        const double *__restrict__ initial_dual,
                                                        const double *__restrict__ primal_product,
                                                        const double *__restrict__ const_lb,
                                                        const double *__restrict__ const_ub,
                                                        int n,
                                                        const double *__restrict__ d_step_size,
                                                        const double *__restrict__ d_al_penalty,
                                                        const int *__restrict__ d_base_count,
                                                        int k_offset,
                                                        double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double step_size = *d_step_size;
    double al_penalty = *d_al_penalty;
    double prox_step = step_size + al_penalty;
    int current_k = *d_base_count + k_offset;
    double weight = (double)(current_k) / (double)(current_k + 1);
    if (i < n)
    {
        double z_bar = fmax(const_lb[i], fmin(primal_product[i] - current_dual[i] / prox_step, const_ub[i]));
        cp_al_dual[i] = current_dual[i] - step_size * (primal_product[i] - z_bar);
        halpern_dual[i] = weight * cp_al_dual[i] + (1.0 - weight) * initial_dual[i];
        reflected_dual[i] = 2.0 * cp_al_dual[i] - current_dual[i];
        double reflected = reflection_coeff * reflected_dual[i] + (1.0 - reflection_coeff) * current_dual[i];
        current_dual[i] = weight * reflected + (1.0 - weight) * initial_dual[i];
    }
}

__global__ void compute_box_al_predictor_kernel(double *__restrict__ dual_predictor,
                                                const double *__restrict__ current_dual,
                                                const double *__restrict__ current_primal_product,
                                                const double *__restrict__ const_lb,
                                                const double *__restrict__ const_ub,
                                                const double *__restrict__ d_al_penalty,
                                                int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double al_penalty = *d_al_penalty;
    if (i < n)
    {
        if (al_penalty <= 0.0)
        {
            dual_predictor[i] = current_dual[i];
            return;
        }
        double z = fmax(const_lb[i], fmin(current_primal_product[i] - current_dual[i] / al_penalty, const_ub[i]));
        dual_predictor[i] = current_dual[i] - al_penalty * (current_primal_product[i] - z);
    }
}

__global__ void rescale_solution_kernel(double *__restrict__ primal_solution,
                                        double *__restrict__ dual_solution,
                                        const double *__restrict__ variable_rescaling,
                                        const double *__restrict__ constraint_rescaling,
                                        const double objective_vector_rescaling,
                                        const double constraint_bound_rescaling,
                                        int n_vars,
                                        int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        primal_solution[i] = primal_solution[i] / variable_rescaling[i] / constraint_bound_rescaling;
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        dual_solution[idx] = dual_solution[idx] / constraint_rescaling[idx] / objective_vector_rescaling;
    }
}

__global__ void compute_delta_solution_kernel(const double *__restrict__ initial_primal,
                                              const double *__restrict__ cp_al_primal,
                                              double *__restrict__ delta_primal,
                                              const double *__restrict__ initial_dual,
                                              const double *__restrict__ cp_al_dual,
                                              double *__restrict__ delta_dual,
                                              int n_vars,
                                              int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        delta_primal[i] = (cp_al_primal[i] - initial_primal[i]);
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        delta_dual[idx] = (cp_al_dual[idx] - initial_dual[idx]);
    }
}

static void compute_next_primal_solution(cp_al_solver_state_t *state,
                                         const int k_offset,
                                         const double reflection_coefficient,
                                         bool is_major)
{
    if (state->al_penalty > 0.0)
    {
        CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, state->current_primal_solution));
        CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

        CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &HOST_ONE,
                                    state->matA,
                                    state->vec_primal_sol,
                                    &HOST_ZERO,
                                    state->vec_primal_prod,
                                    CUDA_R_64F,
                                    CUSPARSE_SPMV_CSR_ALG2,
                                    state->primal_spmv_buffer));

        compute_box_al_predictor_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
            state->reflected_dual_solution,
            state->current_dual_solution,
            state->primal_product,
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            state->d_al_penalty,
            state->num_constraints);

        CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->reflected_dual_solution));
    }
    else
    {
        CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->current_dual_solution));
    }
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matAt,
                                state->vec_dual_sol,
                                &HOST_ZERO,
                                state->vec_dual_prod,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->dual_spmv_buffer));

    if (is_major)
    {
        compute_next_primal_solution_major_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
            state->current_primal_solution,
            state->cp_al_primal_solution,
            state->halpern_primal_solution,
            state->reflected_primal_solution,
            state->initial_primal_solution,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables,
            state->d_primal_step_size,
            state->dual_slack,
            state->d_inner_count,
            k_offset,
            reflection_coefficient);
    }
    else
    {
        compute_next_primal_solution_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
            state->current_primal_solution,
            state->reflected_primal_solution,
            state->initial_primal_solution,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables,
            state->d_primal_step_size,
            state->d_inner_count,
            k_offset,
            reflection_coefficient);
    }
}

static void compute_next_dual_solution(cp_al_solver_state_t *state,
                                       const int k_offset,
                                       const double reflection_coefficient,
                                       bool is_major)
{
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, state->reflected_primal_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matA,
                                state->vec_primal_sol,
                                &HOST_ZERO,
                                state->vec_primal_prod,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->primal_spmv_buffer));

    if (is_major)
    {
        compute_next_dual_solution_major_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
            state->current_dual_solution,
            state->cp_al_dual_solution,
            state->halpern_dual_solution,
            state->reflected_dual_solution,
            state->initial_dual_solution,
            state->primal_product,
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            state->num_constraints,
            state->d_dual_step_size,
            state->d_al_penalty,
            state->d_inner_count,
            k_offset,
            reflection_coefficient);
    }
    else
    {
        compute_next_dual_solution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
            state->current_dual_solution,
            state->initial_dual_solution,
            state->primal_product,
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            state->num_constraints,
            state->d_dual_step_size,
            state->d_al_penalty,
            state->d_inner_count,
            k_offset,
            reflection_coefficient);
    }
}

static double compute_restart_delta_score(cp_al_solver_state_t *state,
                                          const double *candidate_primal,
                                          const double *reference_primal,
                                          const double *candidate_dual,
                                          const double *reference_dual)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        reference_primal,
        candidate_primal,
        state->delta_primal_solution,
        reference_dual,
        candidate_dual,
        state->delta_dual_solution,
        state->num_variables,
        state->num_constraints);

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->delta_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matAt,
                                state->vec_dual_sol,
                                &HOST_ZERO,
                                state->vec_dual_prod,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->dual_spmv_buffer));

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, state->delta_primal_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matA,
                                state->vec_primal_sol,
                                &HOST_ZERO,
                                state->vec_primal_prod,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->primal_spmv_buffer));

    double primal_norm = 0.0;
    double dual_norm = 0.0;
    double ax_norm = 0.0;
    double cross_term = 0.0;
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_norm));
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_norm));
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->primal_product, 1, &ax_norm));

    double score = INFINITY;
    if (state->al_sigma_mode && state->al_sigma > 0.0 && state->al_lambda_param > 0.0)
    {
        const double val = state->al_sigma *
                (state->al_lambda_param * primal_norm * primal_norm - ax_norm * ax_norm) +
            dual_norm * dual_norm / fmax(state->al_sigma, 1e-16);
        score = sqrt(fmax(val, 0.0));
    }
    else
    {
        CUBLAS_CHECK(cublasDdot(state->blas_handle,
                                state->num_variables,
                                state->dual_product,
                                1,
                                state->delta_primal_solution,
                                1,
                                &cross_term));

        const double movement = primal_norm * primal_norm * state->primal_weight +
            dual_norm * dual_norm / state->primal_weight;
        const double interaction = 2 * state->step_size * cross_term;
        const double augmentation = state->step_size * state->al_penalty * ax_norm * ax_norm;
        score = sqrt(fmax(movement + interaction - augmentation, 0.0));
    }

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, state->cp_al_primal_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->cp_al_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));
    return score;
}

static void choose_restart_candidate(cp_al_solver_state_t *state)
{
    const int restart_select_mode = get_env_int_or_default("CP_AL_RESTART_SELECT", 0);
    state->restart_target = 0;
    state->restart_score_current = state->fixed_point_error;
    state->restart_score_average = INFINITY;
    state->restart_score_halpern = INFINITY;
    if (restart_select_mode == 0)
    {
        return;
    }
    const int restart_select_min_iters =
        get_env_int_or_default("CP_AL_RESTART_SELECT_MIN_ITERS", 0);
    const double restart_select_min_time =
        get_env_double_or_default("CP_AL_RESTART_SELECT_MIN_TIME", 0.0);
    if (state->total_count < restart_select_min_iters ||
        state->cumulative_time_sec < restart_select_min_time)
    {
        return;
    }
    state->restart_score_average = compute_restart_delta_score(state,
                                                               state->cp_al_primal_solution,
                                                               state->current_primal_solution,
                                                               state->cp_al_dual_solution,
                                                               state->current_dual_solution);
    const bool restart_select_threepoint =
        restart_select_mode == 3 || get_env_int_or_default("CP_AL_RESTART_SELECT_THREEPOINT", 0) != 0;
    if (restart_select_threepoint)
    {
        state->restart_score_halpern = compute_restart_delta_score(state,
                                                                   state->cp_al_primal_solution,
                                                                   state->halpern_primal_solution,
                                                                   state->cp_al_dual_solution,
                                                                   state->halpern_dual_solution);
    }
    const double restart_select_margin = get_env_double_or_default("CP_AL_RESTART_SELECT_MARGIN", 1.0);
    if (restart_select_mode == 2)
    {
        state->restart_target = 1;
        state->fixed_point_error = state->restart_score_average;
    }
    else if (isfinite(state->restart_score_current))
    {
        int best_target = 0;
        double best_score = state->restart_score_current;
        if (isfinite(state->restart_score_average) && state->restart_score_average < best_score)
        {
            best_target = 1;
            best_score = state->restart_score_average;
        }
        if (restart_select_threepoint && isfinite(state->restart_score_halpern) &&
            state->restart_score_halpern < best_score)
        {
            best_target = 2;
            best_score = state->restart_score_halpern;
        }
        if (best_target != 0 && best_score <= restart_select_margin * state->restart_score_current)
        {
            state->restart_target = best_target;
            state->fixed_point_error = best_score;
        }
        else
        {
            state->fixed_point_error = state->restart_score_current;
        }
    }
    else
    {
        state->fixed_point_error = state->restart_score_current;
    }
}

static void perform_restart(cp_al_solver_state_t *state, const cp_al_parameters_t *params)
{
    const int restart_select_mode = get_env_int_or_default("CP_AL_RESTART_SELECT", 0);
    const int restart_select_trace = get_env_int_or_default("CP_AL_RESTART_SELECT_TRACE", 0);
    const double restart_select_margin = get_env_double_or_default("CP_AL_RESTART_SELECT_MARGIN", 1.0);
    const int restart_target = restart_select_mode != 0 ? state->restart_target : 0;
    const bool restart_to_reflected_halpern = restart_target == 1;
    const bool restart_to_plain_halpern = restart_target == 2;
    const double *restart_primal_solution = restart_to_reflected_halpern ? state->current_primal_solution
        : restart_to_plain_halpern                         ? state->halpern_primal_solution
                                                            : state->cp_al_primal_solution;
    const double *restart_dual_solution = restart_to_reflected_halpern ? state->current_dual_solution
        : restart_to_plain_halpern                       ? state->halpern_dual_solution
                                                          : state->cp_al_dual_solution;
    const char *restart_target_name = restart_to_reflected_halpern ? "reflected_halpern"
        : restart_to_plain_halpern                              ? "halpern"
                                                                 : "current";

    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->initial_primal_solution,
        restart_primal_solution,
        state->delta_primal_solution,
        state->initial_dual_solution,
        restart_dual_solution,
        state->delta_dual_solution,
        state->num_variables,
        state->num_constraints);

    double primal_dist, dual_dist;
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_dist));
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_dist));

    double ratio_infeas = state->relative_dual_residual / fmax(state->relative_primal_residual, 1e-300);

    if (state->al_sigma_mode)
    {
        const double sigma_before = state->al_sigma;
        double sigma_error = NAN;
        double sigma_ratio = 1.0;
        bool sigma_updated = false;
        bool sigma_rollback = false;

        if (state->al_sigma_adaptive)
        {
            CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, state->delta_primal_solution));
            CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));
            CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &HOST_ONE,
                                        state->matA,
                                        state->vec_primal_sol,
                                        &HOST_ZERO,
                                        state->vec_primal_prod,
                                        CUDA_R_64F,
                                        CUSPARSE_SPMV_CSR_ALG2,
                                        state->primal_spmv_buffer));

            double ax_dist = 0.0;
            CUBLAS_CHECK(
                cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->primal_product, 1, &ax_dist));
            const double dx_metric_sq =
                state->al_lambda_param * primal_dist * primal_dist - ax_dist * ax_dist;
            if (dual_dist > 1e-16 && dual_dist < 1e12 && dx_metric_sq > 1e-24 && dx_metric_sq < 1e24 &&
                isfinite(state->al_sigma) && state->al_sigma > 1e-16 && ratio_infeas > 1e-8 &&
                ratio_infeas < 1e8)
            {
                const double sigma_balance_gain =
                    get_env_double_or_default("CP_AL_SIGMA_BALANCE_GAIN", 0.15);
                const double sigma_error_smooth =
                    get_env_double_or_default("CP_AL_SIGMA_I_SMOOTH", params->restart_params.i_smooth);
                const double sigma_pid_kp =
                    get_env_double_or_default("CP_AL_SIGMA_KP", params->restart_params.k_p);
                const double sigma_pid_ki =
                    get_env_double_or_default("CP_AL_SIGMA_KI", params->restart_params.k_i);
                const double sigma_pid_kd =
                    get_env_double_or_default("CP_AL_SIGMA_KD", params->restart_params.k_d);
                const double sigma_ratio_min =
                    get_env_double_or_default("CP_AL_SIGMA_UPDATE_MIN", 0.60);
                const double sigma_ratio_max =
                    get_env_double_or_default("CP_AL_SIGMA_UPDATE_MAX", 1.70);

                sigma_error = log(fmax(dual_dist, 1e-300)) -
                    0.5 * log(fmax(dx_metric_sq, 1e-300)) - log(fmax(state->al_sigma, 1e-300));
                double balance_error = 0.5 * log(fmax(ratio_infeas, 1e-300));
                balance_error = fmax(-2.0, fmin(balance_error, 2.0));
                sigma_error += sigma_balance_gain * balance_error;
                state->al_sigma_error_sum =
                    sigma_error_smooth * state->al_sigma_error_sum + sigma_error;
                const double delta_error = sigma_error - state->al_sigma_last_error;
                sigma_ratio = exp(sigma_pid_kp * sigma_error +
                                  sigma_pid_ki * state->al_sigma_error_sum +
                                  sigma_pid_kd * delta_error);
                sigma_ratio = fmin(fmax(sigma_ratio, sigma_ratio_min), sigma_ratio_max);
                state->al_sigma *= sigma_ratio;
                state->al_sigma_last_error = sigma_error;
                sigma_updated = true;
            }
            else
            {
                state->al_sigma = state->al_sigma_best;
                state->al_sigma_error_sum = 0.0;
                state->al_sigma_last_error = 0.0;
                sigma_rollback = true;
            }
        }

        if (!isfinite(state->al_sigma) || state->al_sigma <= 1e-16)
        {
            state->al_sigma = state->al_sigma_best;
            state->al_sigma_error_sum = 0.0;
            state->al_sigma_last_error = 0.0;
            sigma_rollback = true;
        }
        state->al_sigma = fmin(fmax(state->al_sigma, state->al_sigma_min), state->al_sigma_max);

        double primal_dual_residual_gap =
            fabs(log10(fmax(state->relative_dual_residual, 1e-300) /
                       fmax(state->relative_primal_residual, 1e-300)));
        if (isfinite(primal_dual_residual_gap) && primal_dual_residual_gap < state->best_primal_dual_residual_gap)
        {
            state->best_primal_dual_residual_gap = primal_dual_residual_gap;
            state->al_sigma_best = state->al_sigma;
        }
        state->primal_weight_error_sum = 0.0;
        state->primal_weight_last_error = 0.0;

        if (get_env_int_or_default("CP_AL_SIGMA_TRACE", 0) != 0 &&
            (sigma_updated || sigma_rollback || fabs(state->al_sigma - sigma_before) > 1e-15))
        {
            fprintf(stderr,
                    "[box-sigma] iter=%d sigma %.6e -> %.6e best=%.6e ratio=%.6e err=%.6e "
                    "gap=%.6e flags(update=%d rollback=%d)\n",
                    state->total_count,
                    sigma_before,
                    state->al_sigma,
                    state->al_sigma_best,
                    sigma_ratio,
                    sigma_error,
                    primal_dual_residual_gap,
                    sigma_updated ? 1 : 0,
                    sigma_rollback ? 1 : 0);
            fflush(stderr);
        }
    }
    else
    {
        if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 && dual_dist < 1e12 &&
            ratio_infeas > 1e-8 && ratio_infeas < 1e8)
        {
            double error = log(dual_dist) - log(primal_dist) - log(state->primal_weight);
            state->primal_weight_error_sum *= params->restart_params.i_smooth;
            state->primal_weight_error_sum += error;
            double delta_error = error - state->primal_weight_last_error;
            state->primal_weight *=
                exp(params->restart_params.k_p * error + params->restart_params.k_i * state->primal_weight_error_sum +
                    params->restart_params.k_d * delta_error);
            state->primal_weight_last_error = error;
        }
        else
        {
            state->primal_weight = state->best_primal_weight;
            state->primal_weight_error_sum = 0.0;
            state->primal_weight_last_error = 0.0;
        }

        double primal_dual_residual_gap =
            fabs(log10(fmax(state->relative_dual_residual, 1e-300) /
                       fmax(state->relative_primal_residual, 1e-300)));
        if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
        {
            state->best_primal_dual_residual_gap = primal_dual_residual_gap;
            state->best_primal_weight = state->primal_weight;
        }
        update_adaptive_lambda_on_restart(state, params->verbose);
    }
    refresh_step_schedule(state);

    if (restart_select_mode != 0 && (params->verbose || restart_select_trace != 0))
    {
        fprintf(stderr,
                "[restart-select] iter=%d target=%s score_current=%.12e score_reflected_halpern=%.12e "
                "score_halpern=%.12e candidate=%.12e margin=%.6e mode=%d\n",
                state->total_count,
                restart_target_name,
                state->restart_score_current,
                state->restart_score_average,
                state->restart_score_halpern,
                state->fixed_point_error,
                restart_select_margin,
                restart_select_mode);
        fflush(stderr);
    }

    if (restart_to_reflected_halpern)
    {
        CUDA_CHECK(cudaMemcpy(state->reflected_primal_solution,
                              state->cp_al_primal_solution,
                              state->num_variables * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->reflected_dual_solution,
                              state->cp_al_dual_solution,
                              state->num_constraints * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->initial_primal_solution,
                              state->current_primal_solution,
                              state->num_variables * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->cp_al_primal_solution,
                              state->current_primal_solution,
                              state->num_variables * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->initial_dual_solution,
                              state->current_dual_solution,
                              state->num_constraints * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->cp_al_dual_solution,
                              state->current_dual_solution,
                              state->num_constraints * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }
    else if (restart_to_plain_halpern)
    {
        CUDA_CHECK(cudaMemcpy(state->reflected_primal_solution,
                              state->cp_al_primal_solution,
                              state->num_variables * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->reflected_dual_solution,
                              state->cp_al_dual_solution,
                              state->num_constraints * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->initial_primal_solution,
                              state->halpern_primal_solution,
                              state->num_variables * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->current_primal_solution,
                              state->halpern_primal_solution,
                              state->num_variables * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->cp_al_primal_solution,
                              state->halpern_primal_solution,
                              state->num_variables * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->initial_dual_solution,
                              state->halpern_dual_solution,
                              state->num_constraints * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->current_dual_solution,
                              state->halpern_dual_solution,
                              state->num_constraints * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->cp_al_dual_solution,
                              state->halpern_dual_solution,
                              state->num_constraints * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(state->initial_primal_solution,
                              state->cp_al_primal_solution,
                              state->num_variables * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->current_primal_solution,
                              state->cp_al_primal_solution,
                              state->num_variables * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->initial_dual_solution,
                              state->cp_al_dual_solution,
                              state->num_constraints * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->current_dual_solution,
                              state->cp_al_dual_solution,
                              state->num_constraints * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }

    state->lambda_last_restart_iter = state->total_count;
    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

static void refresh_step_schedule(cp_al_solver_state_t *state)
{
    const double step_margin = get_env_double_or_default("CP_AL_STEP_MARGIN", 0.998 * 0.998);
    if (state->al_sigma_mode)
    {
        const double sigma = fmax(state->al_sigma, 0.0);
        if (sigma <= 0.0)
        {
            state->al_penalty = 0.0;
            state->al_lambda_param = 0.0;
            state->step_size = sqrt(fmax(step_margin / fmax(state->lambda_est, 1e-12), 1e-24));
            state->al_lambda = 0.0;
            return;
        }
        if (!isfinite(state->al_lambda_param) || state->al_lambda_param <= 0.0)
        {
            const double default_lambda_param =
                get_env_double_or_default("CP_AL_LAMBDA_FACTOR", 2.01) * fmax(state->lambda_est, 1e-12);
            state->al_lambda_param = get_env_double_or_default("CP_AL_LAMBDA_PARAM", default_lambda_param);
        }
        state->al_lambda_param = fmax(state->al_lambda_param, (2.0 + 1e-12) * fmax(state->lambda_est, 1e-12));
        const double tau = 1.0 / (sigma * state->al_lambda_param);
        const double rho = sigma;
        state->al_penalty = sigma;
        state->step_size = sqrt(fmax(tau * rho, 1e-24));
        state->primal_weight = sqrt(fmax(rho / tau, 1e-24));
        state->al_lambda = 0.0;
        return;
    }

    state->al_lambda = fmin(fmax(state->al_lambda, 0.0), 0.95);
    const double eta_sq =
        step_margin * (1.0 - state->al_lambda) / fmax(state->lambda_est, 1e-12);
    state->step_size = sqrt(fmax(eta_sq, 1e-24));
    const double dual_step = state->step_size * fmax(state->primal_weight, 1e-12);
    state->al_penalty = dual_step * state->al_lambda / fmax(1.0 - state->al_lambda, 1e-12);
}

static void initialize_sigma_parameters(cp_al_solver_state_t *state)
{
    const double default_lambda_param =
        get_env_double_or_default("CP_AL_LAMBDA_FACTOR", 2.01) * fmax(state->lambda_est, 1e-12);
    state->al_lambda_param = get_env_double_or_default("CP_AL_LAMBDA_PARAM", default_lambda_param);
    state->al_lambda_param = fmax(state->al_lambda_param, (2.0 + 1e-12) * fmax(state->lambda_est, 1e-12));
    state->al_sigma_base = 1.0 / sqrt(fmax(state->al_lambda_param, 1e-16));
    state->al_sigma_min =
        get_env_double_or_default("CP_AL_SIGMA_MIN_FACTOR", 1e-4) * state->al_sigma_base;
    state->al_sigma_max =
        get_env_double_or_default("CP_AL_SIGMA_MAX_FACTOR", 100.0) * state->al_sigma_base;
    const double sigma_init =
        get_env_double_or_default("CP_AL_SIGMA_INIT_FACTOR", 6.0) * state->al_sigma_base;
    state->al_sigma = get_env_double_or_default("CP_AL_SIGMA", sigma_init);
    state->al_sigma = fmin(fmax(state->al_sigma, state->al_sigma_min), state->al_sigma_max);
    state->al_sigma_best = state->al_sigma;
    state->al_sigma_error_sum = 0.0;
    state->al_sigma_last_error = 0.0;
    const int sigma_env_set = env_var_is_set("CP_AL_SIGMA");
    const int adaptive_default = sigma_env_set ? 0 : 1;
    state->al_sigma_adaptive =
        get_env_int_or_default("CP_AL_SIGMA_ADAPTIVE", adaptive_default) != 0;
    state->al_sigma_mode = true;
}

static void initialize_adaptive_lambda(cp_al_solver_state_t *state)
{
    const char *lambda_profile = getenv("CP_AL_LAMBDA_PROFILE");
    const char *lambda_grid_raw = getenv("CP_AL_LAMBDA_GRID");
    const bool use_grid4_race_profile =
        lambda_profile && (strcmp(lambda_profile, "grid4_race") == 0 || strcmp(lambda_profile, "race4") == 0);
    const bool use_mp_race_profile =
        lambda_profile && (strcmp(lambda_profile, "mp_race") == 0 || strcmp(lambda_profile, "safe_race") == 0);
    const bool use_mp_auto_profile =
        lambda_profile && (strcmp(lambda_profile, "mp_auto") == 0 || strcmp(lambda_profile, "safe_auto") == 0);
    const bool use_small_grid_profile =
        use_grid4_race_profile || use_mp_race_profile || use_mp_auto_profile;

    state->adaptive_lambda =
        use_small_grid_profile || get_env_int_or_default("CP_AL_ADAPTIVE_LAMBDA", 0) != 0;
    state->adaptive_lambda_metrics_ready = false;
    state->lambda_tier = 0;
    state->lambda_hold_remaining = 0;
    state->lambda_restart_count = 0;

    double lambda_grid[CP_AL_MAX_LAMBDA_TIERS];
    int lambda_grid_count = parse_lambda_grid_env(lambda_grid_raw, lambda_grid, CP_AL_MAX_LAMBDA_TIERS);
    if (lambda_grid_count == 0 && (use_mp_race_profile || use_mp_auto_profile))
    {
        const double default_grid[] = {0.0, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40};
        lambda_grid_count = (int)(sizeof(default_grid) / sizeof(default_grid[0]));
        for (int i = 0; i < lambda_grid_count; ++i)
        {
            lambda_grid[i] = default_grid[i];
        }
    }
    if (lambda_grid_count == 0)
    {
        lambda_grid[0] = get_env_double_or_default("CP_AL_LAMBDA_BASE", 0.0);
        lambda_grid[1] = get_env_double_or_default("CP_AL_LAMBDA_LOW", use_small_grid_profile ? 0.02 : 0.10);
        lambda_grid[2] = get_env_double_or_default("CP_AL_LAMBDA_MID", use_small_grid_profile ? 0.10 : 0.10);
        lambda_grid[3] = get_env_double_or_default("CP_AL_LAMBDA_HIGH", use_small_grid_profile ? 0.20 : 0.10);
        lambda_grid_count = 4;
    }
    set_lambda_grid(state, lambda_grid, lambda_grid_count);

    state->lambda_hold_epochs = get_env_int_or_default("CP_AL_LAMBDA_HOLD_EPOCHS", use_mp_auto_profile ? 1 : 2);
    state->lambda_min_restarts =
        get_env_int_or_default("CP_AL_LAMBDA_MIN_RESTARTS", use_mp_race_profile || use_mp_auto_profile ? 2 : (use_grid4_race_profile ? 1 : 0));
    state->lambda_min_iterations =
        get_env_int_or_default("CP_AL_LAMBDA_MIN_ITERS", use_mp_race_profile || use_mp_auto_profile ? 10000 : 0);
    state->lambda_force_restart_iters =
        get_env_int_or_default("CP_AL_LAMBDA_FORCE_RESTART_ITERS", use_mp_race_profile ? 20000 : 0);
    state->lambda_last_restart_iter = 0;
    state->lambda_race =
        use_grid4_race_profile || use_mp_race_profile ||
        get_env_int_or_default("CP_AL_LAMBDA_RACE", 0) != 0;
    state->lambda_race_next_tier = 1;
    state->lambda_race_best_tier = 0;
    state->lambda_race_exploit_remaining = 0;
    state->lambda_race_exploit_epochs =
        get_env_int_or_default("CP_AL_LAMBDA_RACE_EXPLOIT_EPOCHS", use_mp_race_profile ? 2 : (use_grid4_race_profile ? 9999 : 3));
    state->lambda_race_cooldown_remaining = 0;
    state->lambda_race_cooldown_epochs =
        get_env_int_or_default("CP_AL_LAMBDA_RACE_COOLDOWN_EPOCHS", use_mp_race_profile ? 3 : (use_grid4_race_profile ? 1000000000 : 0));
    state->lambda_race_best_score = -INFINITY;
    state->lambda_race_cycle_best_tier = 0;
    state->lambda_race_cycle_best_score = -INFINITY;
    state->lambda_race_bad_ratio =
        get_env_double_or_default("CP_AL_LAMBDA_RACE_BAD_RATIO", use_mp_race_profile ? 3.0 : (use_grid4_race_profile ? 10.0 : INFINITY));
    state->lambda_min_time_sec = get_env_double_or_default("CP_AL_LAMBDA_MIN_TIME", 0.0);
    state->lambda_feas_plateau_tol = get_env_double_or_default("CP_AL_LAMBDA_FEAS_PLATEAU", 0.98);
    state->lambda_gap_plateau_tol = get_env_double_or_default("CP_AL_LAMBDA_GAP_PLATEAU", 0.95);
    state->lambda_fp_plateau_tol = get_env_double_or_default("CP_AL_LAMBDA_FP_PLATEAU", 0.98);
    state->lambda_feas_active_tol = get_env_double_or_default("CP_AL_LAMBDA_FEAS_ACTIVE", 1e-6);
    state->lambda_gap_active_tol = get_env_double_or_default("CP_AL_LAMBDA_GAP_ACTIVE", 1e-6);
    state->lambda_fp_active_tol = get_env_double_or_default("CP_AL_LAMBDA_FP_ACTIVE", 1e-8);
    state->lambda_gap_dominance = get_env_double_or_default("CP_AL_LAMBDA_GAP_DOMINANCE", 100.0);

    state->lambda_hold_epochs = state->lambda_hold_epochs < 0 ? 0 : state->lambda_hold_epochs;
    state->lambda_min_restarts = state->lambda_min_restarts < 0 ? 0 : state->lambda_min_restarts;
    state->lambda_min_iterations = state->lambda_min_iterations < 0 ? 0 : state->lambda_min_iterations;
    state->lambda_force_restart_iters =
        state->lambda_force_restart_iters < 0 ? 0 : state->lambda_force_restart_iters;
    state->lambda_race_exploit_epochs = state->lambda_race_exploit_epochs < 0 ? 0 : state->lambda_race_exploit_epochs;
    state->lambda_race_cooldown_epochs =
        state->lambda_race_cooldown_epochs < 0 ? 0 : state->lambda_race_cooldown_epochs;
    state->lambda_race_bad_ratio = fmax(state->lambda_race_bad_ratio, 1.0);
    state->lambda_min_time_sec = fmax(state->lambda_min_time_sec, 0.0);

    if (state->adaptive_lambda)
    {
        state->al_lambda = state->lambda_base;
    }
    state->lambda_epoch_feas_start = INFINITY;
    state->lambda_epoch_gap_start = INFINITY;
    state->lambda_epoch_fp_start = INFINITY;
    state->lambda_epoch_time_start = state->cumulative_time_sec;
}

static double lambda_from_tier(const cp_al_solver_state_t *state, int tier)
{
    if (state->lambda_num_tiers > 0)
    {
        if (tier < 0)
        {
            tier = 0;
        }
        if (tier >= state->lambda_num_tiers)
        {
            tier = state->lambda_num_tiers - 1;
        }
        return state->lambda_values[tier];
    }
    return 0.0;
}

static void update_adaptive_lambda_on_restart(cp_al_solver_state_t *state, bool verbose)
{
    if (!state->adaptive_lambda)
    {
        return;
    }

    const double eps = 1e-300;
    const double feas_end = fmax(state->relative_primal_residual, state->relative_dual_residual);
    const double gap_end = fmax(fabs(state->relative_objective_gap), 0.0);
    const double fp_end =
        (isfinite(state->fixed_point_error) && state->fixed_point_error > 0.0) ? state->fixed_point_error : INFINITY;

    state->lambda_restart_count += 1;
    if (!state->adaptive_lambda_metrics_ready)
    {
        state->lambda_epoch_feas_start = feas_end;
        state->lambda_epoch_gap_start = gap_end;
        state->lambda_epoch_fp_start = fp_end;
        state->lambda_epoch_time_start = state->cumulative_time_sec;
        state->adaptive_lambda_metrics_ready = true;
        if (verbose)
        {
            printf("lambda-adapt init restart %d lambda %.3e feas %.3e gap %.3e fp %.3e\n",
                   state->lambda_restart_count,
                   state->al_lambda,
                   feas_end,
                   gap_end,
                   fp_end);
        }
        return;
    }

    const double feas_start = state->lambda_epoch_feas_start;
    const double gap_start = state->lambda_epoch_gap_start;
    const double fp_start = state->lambda_epoch_fp_start;
    const double feas_ratio = fmax(feas_end, eps) / fmax(feas_start, eps);
    const double gap_ratio = fmax(gap_end, eps) / fmax(gap_start, eps);
    const double fp_ratio = fmax(fp_end, eps) / fmax(fp_start, eps);
    const double epoch_time = fmax(state->cumulative_time_sec - state->lambda_epoch_time_start, 1e-9);
    const bool mature_enough = state->lambda_restart_count >= state->lambda_min_restarts &&
                               state->total_count >= state->lambda_min_iterations &&
                               state->cumulative_time_sec >= state->lambda_min_time_sec;

    if (state->lambda_race)
    {
        const int old_tier = state->lambda_tier;
        const double old_lambda = state->al_lambda;
        double score =
            0.45 * log(fmax(feas_start, eps) / fmax(feas_end, eps)) +
            0.35 * log(fmax(fp_start, eps) / fmax(fp_end, eps)) +
            0.20 * log(fmax(gap_start, eps) / fmax(gap_end, eps));
        score /= epoch_time;
        const bool bad_positive_epoch =
            old_tier > 0 &&
            (feas_ratio > state->lambda_race_bad_ratio ||
             gap_ratio > state->lambda_race_bad_ratio ||
             fp_ratio > state->lambda_race_bad_ratio);
        if (!bad_positive_epoch && isfinite(score) && score > state->lambda_race_best_score)
        {
            state->lambda_race_best_score = score;
            state->lambda_race_best_tier = old_tier;
        }
        if (!bad_positive_epoch && isfinite(score) && score > state->lambda_race_cycle_best_score)
        {
            state->lambda_race_cycle_best_score = score;
            state->lambda_race_cycle_best_tier = old_tier;
        }
        const double worst_ratio = fmax(feas_ratio, fmax(gap_ratio, fp_ratio));
        const bool moderate_instability = worst_ratio < 3.0;
        const bool feasibility_only_lag =
            score > 5.0 && state->lambda_restart_count <= 10 &&
            feas_ratio > 1.20 && feas_ratio < 2.00 &&
            gap_ratio < 1.20 && fp_ratio < 1.20;
        const bool feasibility_gain_tradeoff =
            score > 5.0 && feas_ratio < 0.50 &&
            fmax(gap_ratio, fp_ratio) > 3.00 &&
            worst_ratio < state->lambda_race_bad_ratio;
        const bool compact_plateau =
            score < -10.0 && moderate_instability && feas_ratio < 1.10;
        const bool race_plateau =
            mature_enough &&
            ((fmax(feas_start, feas_end) > state->lambda_feas_active_tol &&
              feas_ratio > state->lambda_feas_plateau_tol) ||
             (fmax(gap_start, gap_end) > state->lambda_gap_active_tol &&
              gap_ratio > state->lambda_gap_plateau_tol) ||
             (fmax(fp_start, fp_end) > state->lambda_fp_active_tol &&
              fp_ratio > state->lambda_fp_plateau_tol));
        const bool start_race =
            old_tier > 0 ||
            race_plateau ||
            feasibility_only_lag ||
            feasibility_gain_tradeoff ||
            compact_plateau;

        int requested_tier = 0;
        if (bad_positive_epoch)
        {
            requested_tier = 0;
            state->lambda_race_next_tier = 1;
            state->lambda_race_exploit_remaining = 0;
            state->lambda_race_cooldown_remaining = state->lambda_race_cooldown_epochs;
            state->lambda_race_cycle_best_tier = 0;
            state->lambda_race_cycle_best_score = -INFINITY;
        }
        else if (state->lambda_race_cooldown_remaining > 0)
        {
            requested_tier = 0;
            state->lambda_race_cooldown_remaining -= 1;
            state->lambda_race_next_tier = 1;
            state->lambda_race_exploit_remaining = 0;
            state->lambda_race_cycle_best_tier = 0;
            state->lambda_race_cycle_best_score = isfinite(score) ? score : -INFINITY;
        }
        else if (!start_race)
        {
            requested_tier = 0;
            state->lambda_race_next_tier = 1;
            state->lambda_race_exploit_remaining = 0;
            state->lambda_race_cycle_best_tier = 0;
            state->lambda_race_cycle_best_score = isfinite(score) ? score : -INFINITY;
        }
        else if (!mature_enough)
        {
            requested_tier = 0;
            state->lambda_race_cycle_best_tier = 0;
            state->lambda_race_cycle_best_score = isfinite(score) ? score : -INFINITY;
        }
        else if (state->lambda_race_next_tier < state->lambda_num_tiers)
        {
            requested_tier = state->lambda_race_next_tier;
            if (state->lambda_race_next_tier == 1)
            {
                state->lambda_race_cycle_best_tier = old_tier;
                state->lambda_race_cycle_best_score = isfinite(score) ? score : -INFINITY;
            }
            state->lambda_race_next_tier += 1;
            state->lambda_race_exploit_remaining = state->lambda_race_exploit_epochs;
        }
        else if (state->lambda_race_exploit_remaining > 0)
        {
            requested_tier = state->lambda_race_cycle_best_tier;
            state->lambda_race_exploit_remaining -= 1;
        }
        else
        {
            state->lambda_race_next_tier = 1;
            requested_tier = state->lambda_race_cycle_best_tier;
            state->lambda_race_cycle_best_tier = requested_tier;
            state->lambda_race_cycle_best_score = isfinite(score) ? score : -INFINITY;
        }

        state->lambda_tier = requested_tier;
        state->lambda_hold_remaining = 0;
        state->al_lambda = lambda_from_tier(state, state->lambda_tier);
        state->lambda_epoch_feas_start = feas_end;
        state->lambda_epoch_gap_start = gap_end;
        state->lambda_epoch_fp_start = fp_end;
        state->lambda_epoch_time_start = state->cumulative_time_sec;

        if (verbose || old_tier != state->lambda_tier || fabs(old_lambda - state->al_lambda) > 1e-15)
        {
            printf("lambda-race restart %d score %.3e best(tier=%d score=%.3e) cycle(tier=%d score=%.3e) "
                   "tier %d->%d lambda %.3e->%.3e next=%d exploit=%d cool=%d bad=%d start=%d mature=%d "
                   "ratios(feas %.3e gap %.3e fp %.3e)\n",
                   state->lambda_restart_count,
                   score,
                   state->lambda_race_best_tier,
                   state->lambda_race_best_score,
                   state->lambda_race_cycle_best_tier,
                   state->lambda_race_cycle_best_score,
                   old_tier,
                   state->lambda_tier,
                   old_lambda,
                   state->al_lambda,
                   state->lambda_race_next_tier,
                   state->lambda_race_exploit_remaining,
                   state->lambda_race_cooldown_remaining,
                   (int)bad_positive_epoch,
                   (int)start_race,
                   (int)mature_enough,
                   feas_ratio,
                   gap_ratio,
                   fp_ratio);
        }
        return;
    }

    const bool feas_plateau =
        fmax(feas_start, feas_end) > state->lambda_feas_active_tol && feas_ratio > state->lambda_feas_plateau_tol;
    const bool gap_plateau =
        fmax(gap_start, gap_end) > state->lambda_gap_active_tol && gap_ratio > state->lambda_gap_plateau_tol;
    const bool fp_plateau =
        fmax(fp_start, fp_end) > state->lambda_fp_active_tol && fp_ratio > state->lambda_fp_plateau_tol;
    const bool gap_dominates = gap_end > state->lambda_gap_dominance * fmax(feas_end, eps);
    const int low_tier = state->lambda_num_tiers > 1 ? 1 : 0;
    const int mid_tier = state->lambda_num_tiers > 2 ? state->lambda_num_tiers / 2 : low_tier;
    const int high_tier = state->lambda_num_tiers > 0 ? state->lambda_num_tiers - 1 : 0;
    int requested_tier = 0;
    if (!mature_enough)
    {
        requested_tier = 0;
    }
    else if (gap_plateau && gap_dominates)
    {
        requested_tier = high_tier;
    }
    else if (feas_plateau || (gap_plateau && fp_plateau))
    {
        requested_tier = mid_tier;
    }
    else if (gap_plateau || fp_plateau)
    {
        requested_tier = low_tier;
    }

    const int old_tier = state->lambda_tier;
    const double old_lambda = state->al_lambda;
    if (requested_tier > 0)
    {
        state->lambda_tier = requested_tier;
        state->lambda_hold_remaining = state->lambda_hold_epochs;
    }
    else if (state->lambda_hold_remaining > 0 && state->lambda_tier > 0)
    {
        state->lambda_hold_remaining -= 1;
    }
    else
    {
        state->lambda_tier = 0;
        state->lambda_hold_remaining = 0;
    }

    state->al_lambda = lambda_from_tier(state, state->lambda_tier);
    state->lambda_epoch_feas_start = feas_end;
    state->lambda_epoch_gap_start = gap_end;
    state->lambda_epoch_fp_start = fp_end;
    state->lambda_epoch_time_start = state->cumulative_time_sec;

    if (verbose || old_tier != state->lambda_tier || fabs(old_lambda - state->al_lambda) > 1e-15)
    {
        printf("lambda-adapt restart %d ratios(feas %.3e gap %.3e fp %.3e) "
               "plateau(feas %d gap %d fp %d dom %d mature %d) tier %d->%d hold %d lambda %.3e->%.3e\n",
               state->lambda_restart_count,
               feas_ratio,
               gap_ratio,
               fp_ratio,
               (int)feas_plateau,
               (int)gap_plateau,
               (int)fp_plateau,
               (int)gap_dominates,
               (int)mature_enough,
               old_tier,
               state->lambda_tier,
               state->lambda_hold_remaining,
               old_lambda,
               state->al_lambda);
    }
}

static void initialize_step_size_and_primal_weight(cp_al_solver_state_t *state, const cp_al_parameters_t *params)
{
    if (state->constraint_matrix->num_nonzeros == 0)
    {
        state->lambda_est = 1.0;
    }
    else
    {
        double max_sv = estimate_maximum_singular_value(state->sparse_handle,
                                                        state->blas_handle,
                                                        state->constraint_matrix,
                                                        state->constraint_matrix_t,
                                                        params->sv_max_iter,
                                                        params->sv_tol);
        state->lambda_est = max_sv * max_sv;
    }

    if (params->bound_objective_rescaling)
    {
        state->primal_weight = 1.0;
    }
    else
    {
        state->primal_weight = (state->objective_vector_norm + 1.0) / (state->constraint_bound_norm + 1.0);
    }
    const int sigma_env_set = env_var_is_set("CP_AL_SIGMA");
    state->al_sigma_mode = sigma_env_set || get_env_int_or_default("CP_AL_SIGMA_MODE", 0) != 0;
    if (state->al_sigma_mode)
    {
        initialize_sigma_parameters(state);
    }
    else
    {
        state->al_sigma = 0.0;
        state->al_sigma_base = 0.0;
        state->al_sigma_min = 0.0;
        state->al_sigma_max = 0.0;
        state->al_sigma_best = 0.0;
        state->al_sigma_error_sum = 0.0;
        state->al_sigma_last_error = 0.0;
    }
    state->al_lambda = get_env_double_or_default("CP_AL_LAMBDA", 0.05);
    initialize_adaptive_lambda(state);
    if (state->al_sigma_mode)
    {
        state->adaptive_lambda = false;
    }
    refresh_step_schedule(state);
    state->best_primal_weight = state->primal_weight;
    fprintf(stderr,
            "[box-cpal] pw_init=%.3e al_lambda=%.3e sigma_mode=%d sigma_adapt=%d al_sigma=%.3e "
            "sigma_base=%.3e lambda_param=%.3e eta=%.3e refl=%.3e al_penalty=%.3e lambdaA=%.3e adaptive=%d "
            "lambda_grid=",
            state->primal_weight,
            state->al_lambda,
            (int)state->al_sigma_mode,
            (int)state->al_sigma_adaptive,
            state->al_sigma,
            state->al_sigma_base,
            state->al_lambda_param,
            state->step_size,
            params->reflection_coefficient,
            state->al_penalty,
            state->lambda_est,
            (int)state->adaptive_lambda);
    for (int i = 0; i < state->lambda_num_tiers; ++i)
    {
        fprintf(stderr, "%s%.3e", i == 0 ? "" : "/", state->lambda_values[i]);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
}

static void compute_fixed_point_error(cp_al_solver_state_t *state)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->cp_al_primal_solution,
        state->reflected_primal_solution,
        state->delta_primal_solution,
        state->cp_al_dual_solution,
        state->reflected_dual_solution,
        state->delta_dual_solution,
        state->num_variables,
        state->num_constraints);

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->delta_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matAt,
                                state->vec_dual_sol,
                                &HOST_ZERO,
                                state->vec_dual_prod,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->dual_spmv_buffer));

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, state->delta_primal_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matA,
                                state->vec_primal_sol,
                                &HOST_ZERO,
                                state->vec_primal_prod,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->primal_spmv_buffer));

    double interaction, movement;

    double primal_norm = 0.0;
    double dual_norm = 0.0;
    double ax_norm = 0.0;
    double cross_term = 0.0;

    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_norm));
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_norm));
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->primal_product, 1, &ax_norm));

    if (state->al_sigma_mode && state->al_sigma > 0.0 && state->al_lambda_param > 0.0)
    {
        const double val = state->al_sigma *
                (state->al_lambda_param * primal_norm * primal_norm - ax_norm * ax_norm) +
            dual_norm * dual_norm / fmax(state->al_sigma, 1e-16);
        state->fixed_point_error = sqrt(fmax(val, 0.0));
        return;
    }

    movement = primal_norm * primal_norm * state->primal_weight + dual_norm * dual_norm / state->primal_weight;

    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            state->dual_product,
                            1,
                            state->delta_primal_solution,
                            1,
                            &cross_term));
    interaction = 2 * state->step_size * cross_term;

    const double augmentation = state->step_size * state->al_penalty * ax_norm * ax_norm;
    state->fixed_point_error = sqrt(fmax(movement + interaction - augmentation, 0.0));
}

void cp_al_solver_state_free(cp_al_solver_state_t *state)
{
    if (state == NULL)
    {
        return;
    }

    if (state->matA)
        CUSPARSE_CHECK(cusparseDestroySpMat(state->matA));
    if (state->matAt)
        CUSPARSE_CHECK(cusparseDestroySpMat(state->matAt));
    if (state->vec_primal_sol)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_primal_sol));
    if (state->vec_dual_sol)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_dual_sol));
    if (state->vec_primal_prod)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_primal_prod));
    if (state->vec_dual_prod)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_dual_prod));
    if (state->primal_spmv_buffer)
        CUDA_CHECK(cudaFree(state->primal_spmv_buffer));
    if (state->dual_spmv_buffer)
        CUDA_CHECK(cudaFree(state->dual_spmv_buffer));
    if (state->sparse_handle)
        CUSPARSE_CHECK(cusparseDestroy(state->sparse_handle));
    if (state->blas_handle)
        CUBLAS_CHECK(cublasDestroy(state->blas_handle));

    if (state->variable_lower_bound)
        CUDA_CHECK(cudaFree(state->variable_lower_bound));
    if (state->variable_upper_bound)
        CUDA_CHECK(cudaFree(state->variable_upper_bound));
    if (state->objective_vector)
        CUDA_CHECK(cudaFree(state->objective_vector));
    if (state->constraint_matrix->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix->row_ptr));
    if (state->constraint_matrix->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix->col_ind));
    if (state->constraint_matrix->row_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix->row_ind));
    if (state->constraint_matrix->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix->val));
    if (state->constraint_matrix->transpose_map)
        CUDA_CHECK(cudaFree(state->constraint_matrix->transpose_map));
    if (state->constraint_matrix_t->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ptr));
    if (state->constraint_matrix_t->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->col_ind));
    if (state->constraint_matrix_t->row_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ind));
    if (state->constraint_matrix_t->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->val));
    if (state->constraint_matrix_t->transpose_map)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->transpose_map));
    if (state->constraint_lower_bound)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound));
    if (state->constraint_upper_bound)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound));
    if (state->constraint_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound_finite_val));
    if (state->constraint_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound_finite_val));
    if (state->variable_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_lower_bound_finite_val));
    if (state->variable_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_upper_bound_finite_val));
    if (state->initial_primal_solution)
        CUDA_CHECK(cudaFree(state->initial_primal_solution));
    if (state->current_primal_solution)
        CUDA_CHECK(cudaFree(state->current_primal_solution));
    if (state->cp_al_primal_solution)
        CUDA_CHECK(cudaFree(state->cp_al_primal_solution));
    if (state->halpern_primal_solution)
        CUDA_CHECK(cudaFree(state->halpern_primal_solution));
    if (state->reflected_primal_solution)
        CUDA_CHECK(cudaFree(state->reflected_primal_solution));
    if (state->dual_product)
        CUDA_CHECK(cudaFree(state->dual_product));
    if (state->initial_dual_solution)
        CUDA_CHECK(cudaFree(state->initial_dual_solution));
    if (state->current_dual_solution)
        CUDA_CHECK(cudaFree(state->current_dual_solution));
    if (state->cp_al_dual_solution)
        CUDA_CHECK(cudaFree(state->cp_al_dual_solution));
    if (state->halpern_dual_solution)
        CUDA_CHECK(cudaFree(state->halpern_dual_solution));
    if (state->reflected_dual_solution)
        CUDA_CHECK(cudaFree(state->reflected_dual_solution));
    if (state->primal_product)
        CUDA_CHECK(cudaFree(state->primal_product));
    if (state->constraint_rescaling)
        CUDA_CHECK(cudaFree(state->constraint_rescaling));
    if (state->variable_rescaling)
        CUDA_CHECK(cudaFree(state->variable_rescaling));
    if (state->primal_slack)
        CUDA_CHECK(cudaFree(state->primal_slack));
    if (state->dual_slack)
        CUDA_CHECK(cudaFree(state->dual_slack));
    if (state->primal_residual)
        CUDA_CHECK(cudaFree(state->primal_residual));
    if (state->dual_residual)
        CUDA_CHECK(cudaFree(state->dual_residual));
    if (state->delta_primal_solution)
        CUDA_CHECK(cudaFree(state->delta_primal_solution));
    if (state->delta_dual_solution)
        CUDA_CHECK(cudaFree(state->delta_dual_solution));
    if (state->ones_primal_d)
        CUDA_CHECK(cudaFree(state->ones_primal_d));
    if (state->ones_dual_d)
        CUDA_CHECK(cudaFree(state->ones_dual_d));
    if (state->d_primal_step_size)
        CUDA_CHECK(cudaFree(state->d_primal_step_size));
    if (state->d_dual_step_size)
        CUDA_CHECK(cudaFree(state->d_dual_step_size));
    if (state->d_al_penalty)
        CUDA_CHECK(cudaFree(state->d_al_penalty));
    if (state->d_inner_count)
        CUDA_CHECK(cudaFree(state->d_inner_count));

    if (state->constraint_matrix)
        free(state->constraint_matrix);
    if (state->constraint_matrix_t)
        free(state->constraint_matrix_t);

    free(state);
}

void rescale_info_free(rescale_info_t *info)
{
    if (info == NULL)
    {
        return;
    }

    CUDA_CHECK(cudaFree(info->con_rescale));
    CUDA_CHECK(cudaFree(info->var_rescale));

    free(info);
}

static cp_al_result_t *create_result_from_state(cp_al_solver_state_t *state, const lp_problem_t *original_problem)
{
    cp_al_result_t *results = (cp_al_result_t *)safe_calloc(1, sizeof(cp_al_result_t));

    // Compute reduced cost
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->cp_al_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matAt,
                                state->vec_dual_sol,
                                &HOST_ZERO,
                                state->vec_dual_prod,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->dual_spmv_buffer));

    compute_and_rescale_reduced_cost_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->dual_slack,
        state->objective_vector,
        state->dual_product,
        state->variable_rescaling,
        state->objective_vector_rescaling,
        state->constraint_bound_rescaling,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->num_variables);

    rescale_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->cp_al_primal_solution,
        state->cp_al_dual_solution,
        state->variable_rescaling,
        state->constraint_rescaling,
        state->objective_vector_rescaling,
        state->constraint_bound_rescaling,
        state->num_variables,
        state->num_constraints);

    results->primal_solution = (double *)safe_malloc(state->num_variables * sizeof(double));
    results->dual_solution = (double *)safe_malloc(state->num_constraints * sizeof(double));
    results->reduced_cost = (double *)safe_malloc(state->num_variables * sizeof(double));

    CUDA_CHECK(cudaMemcpy(results->primal_solution,
                          state->cp_al_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->dual_solution,
                          state->cp_al_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        results->reduced_cost, state->dual_slack, state->num_variables * sizeof(double), cudaMemcpyDeviceToHost));

    results->num_variables = original_problem->num_variables;
    results->num_constraints = original_problem->num_constraints;
    results->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;
    results->total_count = state->total_count;
    results->rescaling_time_sec = state->rescaling_time_sec;
    results->cumulative_time_sec = state->cumulative_time_sec;
    results->relative_primal_residual = state->relative_primal_residual;
    results->relative_dual_residual = state->relative_dual_residual;
    results->primal_objective_value = state->primal_objective_value;
    results->dual_objective_value = state->dual_objective_value;
    results->objective_gap = state->objective_gap;
    results->relative_objective_gap = state->relative_objective_gap;
    results->max_primal_ray_infeasibility = state->max_primal_ray_infeasibility;
    results->max_dual_ray_infeasibility = state->max_dual_ray_infeasibility;
    results->primal_ray_linear_objective = state->primal_ray_linear_objective;
    results->dual_ray_objective = state->dual_ray_objective;
    results->termination_reason = state->termination_reason;
    results->feasibility_polishing_time = state->feasibility_polishing_time;
    results->feasibility_iteration = state->feasibility_iteration;
    // if (presolve_stats != NULL) {
    //     results->presolve_stats = *presolve_stats;
    // } else {
    //     memset(&(results->presolve_stats), 0, sizeof(PresolveStats));
    // }

    return results;
}

// Feasibility Polishing
void feasibility_polish(const cp_al_parameters_t *params, cp_al_solver_state_t *state)
{
    clock_t feasibility_polishing_start_time = clock();
    if (state->relative_primal_residual < params->termination_criteria.eps_feas_polish_relative &&
        state->relative_dual_residual < params->termination_criteria.eps_feas_polish_relative)
    {
        printf("Skipping feasibility polishing as the solution is already sufficiently feasible.\n");
        return;
    }
    double original_primal_weight = 0.0;
    if (params->bound_objective_rescaling)
    {
        original_primal_weight = 1.0;
    }
    else
    {
        original_primal_weight = (state->objective_vector_norm + 1.0) / (state->constraint_bound_norm + 1.0);
    }

    // PRIMAL FEASIBILITY POLISHING
    cp_al_solver_state_t *primal_state = initialize_primal_feas_polish_state(state);
    primal_state->primal_weight = original_primal_weight;
    primal_state->best_primal_weight = original_primal_weight;
    primal_feasibility_polish(params, primal_state, state);

    if (primal_state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS)
    {
        CUDA_CHECK(cudaMemcpy(state->cp_al_primal_solution,
                              primal_state->cp_al_primal_solution,
                              state->num_variables * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        state->absolute_primal_residual = primal_state->absolute_primal_residual;
        state->relative_primal_residual = primal_state->relative_primal_residual;
        state->primal_objective_value = primal_state->primal_objective_value;
    }
    state->feasibility_iteration += primal_state->total_count - 1;

    // DUAL FEASIBILITY POLISHING
    cp_al_solver_state_t *dual_state = initialize_dual_feas_polish_state(state);
    dual_state->primal_weight = original_primal_weight;
    dual_state->best_primal_weight = original_primal_weight;
    dual_feasibility_polish(params, dual_state, state);

    if (dual_state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS)
    {
        CUDA_CHECK(cudaMemcpy(state->cp_al_dual_solution,
                              dual_state->cp_al_dual_solution,
                              state->num_constraints * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        state->absolute_dual_residual = dual_state->absolute_dual_residual;
        state->relative_dual_residual = dual_state->relative_dual_residual;
        state->dual_objective_value = dual_state->dual_objective_value;
    }
    state->feasibility_iteration += dual_state->total_count - 1;

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);
    state->relative_objective_gap =
        state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));

    // FINAL LOGGING
    cp_al_feas_polish_final_log(primal_state, dual_state, params->verbose);
    primal_feas_polish_state_free(primal_state);
    dual_feas_polish_state_free(dual_state);

    state->feasibility_polishing_time = (double)(clock() - feasibility_polishing_start_time) / CLOCKS_PER_SEC;
    return;
}

void primal_feasibility_polish(const cp_al_parameters_t *params,
                               cp_al_solver_state_t *state,
                               const cp_al_solver_state_t *ori_state)
{
    print_initial_feas_polish_info(true, params);
    bool do_restart = false;
    cudaGraphExec_t graphExec = NULL;
    bool graph_created = false;

    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        sync_inner_count_to_gpu(state);
        compute_next_primal_solution(state, 1, params->reflection_coefficient, true);
        compute_next_dual_solution(state, 1, params->reflection_coefficient, true);

        if (do_restart)
        {
            compute_primal_fixed_point_error(state);
            state->initial_fixed_point_error = state->fixed_point_error;
            do_restart = false;
        }

        if (!graph_created)
        {
            // Start CUDA graph capture
            cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal);

            for (int i = 2; i <= params->termination_evaluation_frequency - 1; i++)
            {
                compute_next_primal_solution(state, i, params->reflection_coefficient, false);
                compute_next_dual_solution(state, i, params->reflection_coefficient, false);
            }

            compute_next_primal_solution(
                state, params->termination_evaluation_frequency, params->reflection_coefficient, true);
            compute_next_dual_solution(
                state, params->termination_evaluation_frequency, params->reflection_coefficient, true);
            // end CUDA graph capture

            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamEndCapture(state->stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            graph_created = true;
        }
        CUDA_CHECK(cudaGraphLaunch(graphExec, state->stream));

        compute_primal_fixed_point_error(state);
        compute_primal_feas_polish_residual(state, ori_state, params->optimality_norm);
        state->inner_count += params->termination_evaluation_frequency;
        state->total_count += params->termination_evaluation_frequency;

        check_feas_polishing_termination_criteria(state, ori_state, &params->termination_criteria, true);
        if (state->total_count % get_print_frequency(state->total_count) == 0)
        {
            display_feas_polish_iteration_stats(state, params->verbose, true);
        }

        // Check Adaptive Restart
        do_restart =
            should_do_adaptive_restart(state, &params->restart_params, params->termination_evaluation_frequency);
        if (do_restart)
        {
            perform_primal_restart(state);
            // sync_step_sizes_to_gpu(state);
        }
    }

    if (graphExec)
    {
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    }
    return;
}

void dual_feasibility_polish(const cp_al_parameters_t *params,
                             cp_al_solver_state_t *state,
                             const cp_al_solver_state_t *ori_state)
{
    print_initial_feas_polish_info(false, params);
    bool do_restart = false;
    cudaGraphExec_t graphExec = NULL;
    bool graph_created = false;

    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        sync_inner_count_to_gpu(state);
        compute_next_primal_solution(state, 1, params->reflection_coefficient, true);
        compute_next_dual_solution(state, 1, params->reflection_coefficient, true);

        if (do_restart)
        {
            compute_dual_fixed_point_error(state);
            state->initial_fixed_point_error = state->fixed_point_error;
            do_restart = false;
        }

        if (!graph_created)
        {
            // Start CUDA graph capture
            cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal);

            for (int i = 2; i <= params->termination_evaluation_frequency - 1; i++)
            {
                compute_next_primal_solution(state, i, params->reflection_coefficient, false);
                compute_next_dual_solution(state, i, params->reflection_coefficient, false);
            }

            compute_next_primal_solution(
                state, params->termination_evaluation_frequency, params->reflection_coefficient, true);
            compute_next_dual_solution(
                state, params->termination_evaluation_frequency, params->reflection_coefficient, true);
            // end CUDA graph capture

            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamEndCapture(state->stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            graph_created = true;
        }
        CUDA_CHECK(cudaGraphLaunch(graphExec, state->stream));

        compute_dual_fixed_point_error(state);
        compute_dual_feas_polish_residual(state, ori_state, params->optimality_norm);
        state->inner_count += params->termination_evaluation_frequency;
        state->total_count += params->termination_evaluation_frequency;

        check_feas_polishing_termination_criteria(state, ori_state, &params->termination_criteria, false);
        if (state->total_count % get_print_frequency(state->total_count) == 0)
        {
            display_feas_polish_iteration_stats(state, params->verbose, false);
        }

        // Check Adaptive Restart
        do_restart =
            should_do_adaptive_restart(state, &params->restart_params, params->termination_evaluation_frequency);
        if (do_restart)
        {
            perform_dual_restart(state);
            // sync_step_sizes_to_gpu(state);
        }
    }

    if (graphExec)
    {
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    }
    return;
}

static cp_al_solver_state_t *initialize_primal_feas_polish_state(const cp_al_solver_state_t *original_state)
{
    cp_al_solver_state_t *primal_state = (cp_al_solver_state_t *)safe_malloc(sizeof(cp_al_solver_state_t));
    *primal_state = *original_state;
    int num_var = original_state->num_variables;
    int num_cons = original_state->num_constraints;

#define ALLOC_ZERO(dest, bytes)                                                                                        \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

    // RESET PROBLEM TO FEASIBILITY PROBLEM
    ALLOC_ZERO(primal_state->objective_vector, num_var * sizeof(double));
    primal_state->objective_constant = 0.0;

#define ALLOC_AND_COPY_DEV(dest, src, bytes)                                                                           \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice));

    // ALLOCATE AND COPY SOLUTION VECTORS
    ALLOC_AND_COPY_DEV(
        primal_state->initial_primal_solution, original_state->initial_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(
        primal_state->current_primal_solution, original_state->current_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(
        primal_state->cp_al_primal_solution, original_state->cp_al_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(
        primal_state->halpern_primal_solution, original_state->halpern_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(
        primal_state->reflected_primal_solution, original_state->reflected_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(primal_state->primal_product, original_state->primal_product, num_cons * sizeof(double));

    // ALLOC ZERO FOR OTHERS
    ALLOC_ZERO(primal_state->initial_dual_solution, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->current_dual_solution, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->cp_al_dual_solution, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->halpern_dual_solution, num_cons * sizeof(double));
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
    primal_state->start_time = clock();
    primal_state->cumulative_time_sec = 0.0;
    primal_state->best_primal_dual_residual_gap = INFINITY;

    // IGNORE DUAL RESIDUAL AND OBJECTIVE GAP
    primal_state->relative_dual_residual = 0.0;
    primal_state->absolute_dual_residual = 0.0;
    primal_state->relative_objective_gap = 0.0;
    primal_state->objective_gap = 0.0;

    return primal_state;
}

void primal_feas_polish_state_free(cp_al_solver_state_t *state)
{
#define SAFE_CUDA_FREE(p)                                                                                              \
    if ((p) != NULL)                                                                                                   \
    {                                                                                                                  \
        CUDA_CHECK(cudaFree(p));                                                                                       \
        (p) = NULL;                                                                                                    \
    }

    if (!state)
        return;
    SAFE_CUDA_FREE(state->objective_vector);
    SAFE_CUDA_FREE(state->initial_primal_solution);
    SAFE_CUDA_FREE(state->current_primal_solution);
    SAFE_CUDA_FREE(state->cp_al_primal_solution);
    SAFE_CUDA_FREE(state->halpern_primal_solution);
    SAFE_CUDA_FREE(state->reflected_primal_solution);
    SAFE_CUDA_FREE(state->dual_product);
    SAFE_CUDA_FREE(state->initial_dual_solution);
    SAFE_CUDA_FREE(state->current_dual_solution);
    SAFE_CUDA_FREE(state->cp_al_dual_solution);
    SAFE_CUDA_FREE(state->halpern_dual_solution);
    SAFE_CUDA_FREE(state->reflected_dual_solution);
    SAFE_CUDA_FREE(state->primal_product);
    SAFE_CUDA_FREE(state->primal_slack);
    SAFE_CUDA_FREE(state->dual_slack);
    SAFE_CUDA_FREE(state->primal_residual);
    SAFE_CUDA_FREE(state->dual_residual);
    SAFE_CUDA_FREE(state->delta_primal_solution);
    SAFE_CUDA_FREE(state->delta_dual_solution);
    free(state);
}

__global__ void zero_finite_value_vectors_kernel(double *__restrict__ vec, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        if (isfinite(vec[idx]))
            vec[idx] = 0.0;
    }
}

static cp_al_solver_state_t *initialize_dual_feas_polish_state(const cp_al_solver_state_t *original_state)
{
    cp_al_solver_state_t *dual_state = (cp_al_solver_state_t *)safe_malloc(sizeof(cp_al_solver_state_t));
    *dual_state = *original_state;
    int num_var = original_state->num_variables;
    int num_cons = original_state->num_constraints;

#define ALLOC_AND_COPY_DEV(dest, src, bytes)                                                                           \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice));

// RESET PROBLEM TO DUAL FEASIBILITY PROBLEM
#define SET_FINITE_TO_ZERO(vec, n)                                                                                     \
    {                                                                                                                  \
        int threads = 256;                                                                                             \
        int blocks = (n + threads - 1) / threads;                                                                      \
        zero_finite_value_vectors_kernel<<<blocks, threads>>>(vec, n);                                                 \
        CUDA_CHECK(cudaDeviceSynchronize());                                                                           \
    }

    ALLOC_AND_COPY_DEV(
        dual_state->constraint_lower_bound, original_state->constraint_lower_bound, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(
        dual_state->constraint_upper_bound, original_state->constraint_upper_bound, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(
        dual_state->variable_lower_bound, original_state->variable_lower_bound, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(
        dual_state->variable_upper_bound, original_state->variable_upper_bound, num_var * sizeof(double));

    SET_FINITE_TO_ZERO(dual_state->constraint_lower_bound, num_cons);
    SET_FINITE_TO_ZERO(dual_state->constraint_upper_bound, num_cons);
    SET_FINITE_TO_ZERO(dual_state->variable_lower_bound, num_var);
    SET_FINITE_TO_ZERO(dual_state->variable_upper_bound, num_var);

#define ALLOC_ZERO(dest, bytes)                                                                                        \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

    ALLOC_ZERO(dual_state->constraint_lower_bound_finite_val, num_cons * sizeof(double));
    ALLOC_ZERO(dual_state->constraint_upper_bound_finite_val, num_cons * sizeof(double));
    ALLOC_ZERO(dual_state->variable_lower_bound_finite_val, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->variable_upper_bound_finite_val, num_var * sizeof(double));

    // ALLOCATE AND COPY SOLUTION VECTORS
    ALLOC_AND_COPY_DEV(
        dual_state->initial_dual_solution, original_state->initial_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(
        dual_state->current_dual_solution, original_state->current_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->cp_al_dual_solution, original_state->cp_al_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(
        dual_state->halpern_dual_solution, original_state->halpern_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(
        dual_state->reflected_dual_solution, original_state->reflected_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->dual_product, original_state->dual_product, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->dual_slack, original_state->dual_slack, num_var * sizeof(double));

    // ALLOC ZERO FOR OTHERS
    ALLOC_ZERO(dual_state->initial_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->current_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->cp_al_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->halpern_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->reflected_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->primal_product, num_cons * sizeof(double));
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
    dual_state->start_time = clock();
    dual_state->cumulative_time_sec = 0.0;
    dual_state->best_primal_dual_residual_gap = INFINITY;

    // IGNORE PRIMAL RESIDUAL AND OBJECTIVE GAP
    dual_state->relative_primal_residual = 0.0;
    dual_state->absolute_primal_residual = 0.0;
    dual_state->relative_objective_gap = 0.0;
    dual_state->objective_gap = 0.0;
    return dual_state;
}

void dual_feas_polish_state_free(cp_al_solver_state_t *state)
{
#define SAFE_CUDA_FREE(p)                                                                                              \
    if ((p) != NULL)                                                                                                   \
    {                                                                                                                  \
        CUDA_CHECK(cudaFree(p));                                                                                       \
        (p) = NULL;                                                                                                    \
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
    SAFE_CUDA_FREE(state->cp_al_primal_solution);
    SAFE_CUDA_FREE(state->halpern_primal_solution);
    SAFE_CUDA_FREE(state->reflected_primal_solution);

    SAFE_CUDA_FREE(state->dual_product);
    SAFE_CUDA_FREE(state->initial_dual_solution);
    SAFE_CUDA_FREE(state->current_dual_solution);
    SAFE_CUDA_FREE(state->cp_al_dual_solution);
    SAFE_CUDA_FREE(state->halpern_dual_solution);
    SAFE_CUDA_FREE(state->reflected_dual_solution);
    SAFE_CUDA_FREE(state->primal_product);

    SAFE_CUDA_FREE(state->primal_slack);
    SAFE_CUDA_FREE(state->dual_slack);
    SAFE_CUDA_FREE(state->primal_residual);
    SAFE_CUDA_FREE(state->dual_residual);
    SAFE_CUDA_FREE(state->delta_primal_solution);
    SAFE_CUDA_FREE(state->delta_dual_solution);
    free(state);
}

static void perform_primal_restart(cp_al_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution,
                          state->cp_al_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution,
                          state->cp_al_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

static void perform_dual_restart(cp_al_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution,
                          state->cp_al_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution,
                          state->cp_al_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

__global__ void compute_delta_primal_solution_kernel(const double *__restrict__ initial_primal,
                                                     const double *__restrict__ cp_al_primal,
                                                     double *__restrict__ delta_primal,
                                                     int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        delta_primal[i] = cp_al_primal[i] - initial_primal[i];
    }
}

__global__ void compute_delta_dual_solution_kernel(const double *__restrict__ initial_dual,
                                                   const double *__restrict__ cp_al_dual,
                                                   double *__restrict__ delta_dual,
                                                   int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_cons)
    {
        delta_dual[i] = cp_al_dual[i] - initial_dual[i];
    }
}

static void compute_primal_fixed_point_error(cp_al_solver_state_t *state)
{
    compute_delta_primal_solution_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->cp_al_primal_solution,
        state->reflected_primal_solution,
        state->delta_primal_solution,
        state->num_variables);
    double primal_norm = 0.0;
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_norm));
    state->fixed_point_error = primal_norm * primal_norm * state->primal_weight;
}

static void compute_dual_fixed_point_error(cp_al_solver_state_t *state)
{
    compute_delta_dual_solution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->cp_al_dual_solution, state->reflected_dual_solution, state->delta_dual_solution, state->num_constraints);
    double dual_norm = 0.0;
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_norm));
    state->fixed_point_error = dual_norm * dual_norm / state->primal_weight;
}
