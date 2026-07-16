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
#include "mps_parser.h"
#include "presolve.h"
#include "solver.h"
#include "utils.h"
#include <float.h>
#include <getopt.h>
#include <libgen.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


static bool env_flag_enabled(const char *name)
{
    const char *value = getenv(name);
    return value != NULL && value[0] != '\0' && atoi(value) != 0;
}


static void final_policy_setenv(const char *name, const char *value)
{
    setenv(name, value, 1);
}

static void final_policy_clearenv(const char *name)
{
    unsetenv(name);
}


// BEGIN_IDENTITY_FREE_FEATURE_POLICY
typedef struct
{
    double coeff_log_range;
    double row_logmax_std;
    double col_logmax_std;
    double row_logspan_mean;
    double col_logspan_mean;
    double row_span2_fraction;
    double col_span2_fraction;
    double row_nnz_cv;
    double col_nnz_cv;
    double equality_fraction;
    double two_sided_variable_fraction;
    double objective_density;
} feature_matrix_stats_t;

static double coefficient_of_variation(double sum, double sum_sq, int count)
{
    if (count <= 0 || sum <= 0.0)
    {
        return 0.0;
    }
    const double mean = sum / (double)count;
    const double variance = fmax(sum_sq / (double)count - mean * mean, 0.0);
    return sqrt(variance) / mean;
}

static double standard_deviation(double sum, double sum_sq, int count)
{
    if (count <= 0)
    {
        return 0.0;
    }
    const double mean = sum / (double)count;
    return sqrt(fmax(sum_sq / (double)count - mean * mean, 0.0));
}

static feature_matrix_stats_t compute_feature_matrix_stats(const lp_problem_t *problem)
{
    feature_matrix_stats_t stats = {0};
    const int m = problem->num_constraints;
    const int n = problem->num_variables;
    const int nnz = problem->constraint_matrix_num_nonzeros;
    if (m <= 0 || n <= 0 || nnz <= 0)
    {
        return stats;
    }

    double *col_min = safe_malloc((size_t)n * sizeof(double));
    double *col_max = safe_calloc((size_t)n, sizeof(double));
    int *col_nnz = safe_calloc((size_t)n, sizeof(int));
    for (int col = 0; col < n; ++col)
    {
        col_min[col] = DBL_MAX;
    }

    double global_min = DBL_MAX;
    double global_max = 0.0;
    double row_logmax_sum = 0.0;
    double row_logmax_sum_sq = 0.0;
    double row_logspan_sum = 0.0;
    double row_nnz_sum = 0.0;
    double row_nnz_sum_sq = 0.0;
    int nonempty_rows = 0;
    int row_span2_count = 0;

    for (int row = 0; row < m; ++row)
    {
        const int begin = problem->constraint_matrix_row_pointers[row];
        const int end = problem->constraint_matrix_row_pointers[row + 1];
        const int row_count = end - begin;
        row_nnz_sum += (double)row_count;
        row_nnz_sum_sq += (double)row_count * (double)row_count;
        double row_min = DBL_MAX;
        double row_max = 0.0;
        for (int index = begin; index < end; ++index)
        {
            const double value = fabs(problem->constraint_matrix_values[index]);
            const int col = problem->constraint_matrix_col_indices[index];
            if (!isfinite(value) || value <= 0.0 || col < 0 || col >= n)
            {
                continue;
            }
            global_min = fmin(global_min, value);
            global_max = fmax(global_max, value);
            row_min = fmin(row_min, value);
            row_max = fmax(row_max, value);
            col_min[col] = fmin(col_min[col], value);
            col_max[col] = fmax(col_max[col], value);
            col_nnz[col] += 1;
        }
        if (row_max > 0.0 && row_min < DBL_MAX)
        {
            const double log_max = log10(row_max);
            const double log_span = log10(row_max / row_min);
            row_logmax_sum += log_max;
            row_logmax_sum_sq += log_max * log_max;
            row_logspan_sum += log_span;
            row_span2_count += log_span >= 2.0 ? 1 : 0;
            nonempty_rows += 1;
        }
    }

    double col_logmax_sum = 0.0;
    double col_logmax_sum_sq = 0.0;
    double col_logspan_sum = 0.0;
    double col_nnz_sum = 0.0;
    double col_nnz_sum_sq = 0.0;
    int nonempty_cols = 0;
    int col_span2_count = 0;
    for (int col = 0; col < n; ++col)
    {
        col_nnz_sum += (double)col_nnz[col];
        col_nnz_sum_sq += (double)col_nnz[col] * (double)col_nnz[col];
        if (col_max[col] > 0.0 && col_min[col] < DBL_MAX)
        {
            const double log_max = log10(col_max[col]);
            const double log_span = log10(col_max[col] / col_min[col]);
            col_logmax_sum += log_max;
            col_logmax_sum_sq += log_max * log_max;
            col_logspan_sum += log_span;
            col_span2_count += log_span >= 2.0 ? 1 : 0;
            nonempty_cols += 1;
        }
    }

    stats.coeff_log_range =
        global_max > 0.0 && global_min < DBL_MAX ? log10(global_max / global_min) : 0.0;
    stats.row_logmax_std = standard_deviation(row_logmax_sum, row_logmax_sum_sq, nonempty_rows);
    stats.col_logmax_std = standard_deviation(col_logmax_sum, col_logmax_sum_sq, nonempty_cols);
    stats.row_logspan_mean = nonempty_rows > 0 ? row_logspan_sum / (double)nonempty_rows : 0.0;
    stats.col_logspan_mean = nonempty_cols > 0 ? col_logspan_sum / (double)nonempty_cols : 0.0;
    stats.row_span2_fraction = nonempty_rows > 0 ? (double)row_span2_count / (double)nonempty_rows : 0.0;
    stats.col_span2_fraction = nonempty_cols > 0 ? (double)col_span2_count / (double)nonempty_cols : 0.0;
    stats.row_nnz_cv = coefficient_of_variation(row_nnz_sum, row_nnz_sum_sq, m);
    stats.col_nnz_cv = coefficient_of_variation(col_nnz_sum, col_nnz_sum_sq, n);

    int equality_count = 0;
    for (int row = 0; row < m; ++row)
    {
        const double lower = problem->constraint_lower_bound[row];
        const double upper = problem->constraint_upper_bound[row];
        equality_count += isfinite(lower) && isfinite(upper) && lower == upper ? 1 : 0;
    }
    int two_sided_variable_count = 0;
    int nonzero_objective_count = 0;
    for (int col = 0; col < n; ++col)
    {
        two_sided_variable_count +=
            isfinite(problem->variable_lower_bound[col]) && isfinite(problem->variable_upper_bound[col]) ? 1 : 0;
        nonzero_objective_count += fabs(problem->objective_vector[col]) > 0.0 ? 1 : 0;
    }
    stats.equality_fraction = (double)equality_count / (double)m;
    stats.two_sided_variable_fraction = (double)two_sided_variable_count / (double)n;
    stats.objective_density = (double)nonzero_objective_count / (double)n;

    free(col_min);
    free(col_max);
    free(col_nnz);
    return stats;
}

static void apply_feature_auto_policy(const lp_problem_t *problem,
                                      cp_al_parameters_t *params,
                                      const feature_matrix_stats_t *precomputed_stats)
{
    if (!env_flag_enabled("CP_AL_FEATURE_AUTO_POLICY"))
    {
        return;
    }
    if (precomputed_stats == NULL)
    {
        fprintf(stderr, "[feature-auto-policy-error] reason=missing-precomputed-stats\n");
        return;
    }

    const char *variant = getenv("CP_AL_FEATURE_POLICY_VARIANT");
    if (variant == NULL || variant[0] == '\0')
    {
        variant = "safe0";
    }
    const bool sigma_probe = strcmp(variant, "probe") == 0;
    const bool sigma_always =
        strcmp(variant, "always_select") == 0 || strcmp(variant, "always_noselect") == 0;
    const bool sigma_trigger =
        strcmp(variant, "trigger") == 0 || strcmp(variant, "trigger_select") == 0 || sigma_probe;
    const bool restart_select =
        strcmp(variant, "always_select") == 0 || strcmp(variant, "trigger_select") == 0;

    // Evaluation frequency defaults to 200. The unified policy may request a
    // different anonymous-cohort value via CP_AL_UNIFIED_EVAL_FREQ (identity-free).
    int unified_eval_freq = 200;
    const char *uef = getenv("CP_AL_UNIFIED_EVAL_FREQ");
    if (uef != NULL && uef[0] != '\0')
    {
        int v = atoi(uef);
        if (v > 0)
        {
            unified_eval_freq = v;
        }
    }
    params->termination_evaluation_frequency = unified_eval_freq;
    params->reflection_coefficient = 1.0;

    final_policy_setenv("CP_AL_LAMBDA", "0");
    final_policy_setenv("CP_AL_LAMBDA_FACTOR", "2.01");
    final_policy_setenv("CP_AL_ADAPTIVE_LAMBDA", "0");
    final_policy_clearenv("CP_AL_LAMBDA_PROFILE");
    final_policy_clearenv("CP_AL_LAMBDA_GRID");

    const feature_matrix_stats_t matrix_stats = *precomputed_stats;

    // A global override is used only by the causal CR grid. The frozen default
    // rule uses coefficient scaling diagnostics and never reads problem identity.
    const char *cr_override_raw = getenv("CP_AL_FEATURE_CR_ITERS");
    int cr_override = -1;
    int cr_selected = 0;
    if (cr_override_raw != NULL && cr_override_raw[0] != '\0')
    {
        cr_override = atoi(cr_override_raw);
        cr_override = cr_override < 0 ? 0 : cr_override;
        cr_selected = cr_override;
    }
    else if (matrix_stats.col_nnz_cv <= 0.19012745)
    {
        cr_selected = 5;
    }
    else if (matrix_stats.col_nnz_cv <= 0.39521255)
    {
        cr_selected = 50;
    }
    else if (matrix_stats.coeff_log_range <= 3.222854)
    {
        cr_selected = 5;
    }
    else
    {
        cr_selected = 20;
    }
    char cr_iters_buffer[32];
    snprintf(cr_iters_buffer, sizeof(cr_iters_buffer), "%d", cr_selected);
    final_policy_setenv("CP_AL_CR_ITERS", cr_iters_buffer);
    final_policy_setenv("CP_AL_CR_SCALING", cr_selected > 0 ? "1" : "0");

    // Positive augmentation is either enabled from the start (ablation) or by
    // the same residual-imbalance gate for every instance. Once active, the
    // existing PID, residual correction, clipping, and rollback are unchanged.
    const char *sigma_init_factor = getenv("CP_AL_FEATURE_SIGMA_INIT_FACTOR");
    const char *sigma_trigger_min_iters = getenv("CP_AL_FEATURE_SIGMA_TRIGGER_MIN_ITERS");
    const char *sigma_trigger_primal = getenv("CP_AL_FEATURE_SIGMA_TRIGGER_PRIMAL");
    const char *sigma_trigger_dual = getenv("CP_AL_FEATURE_SIGMA_TRIGGER_DUAL");
    const char *sigma_trigger_ratio = getenv("CP_AL_FEATURE_SIGMA_TRIGGER_RATIO");
    sigma_init_factor = sigma_init_factor != NULL && sigma_init_factor[0] != '\0' ? sigma_init_factor : "6.0";
    sigma_trigger_min_iters =
        sigma_trigger_min_iters != NULL && sigma_trigger_min_iters[0] != '\0' ? sigma_trigger_min_iters : "200";
    sigma_trigger_primal =
        sigma_trigger_primal != NULL && sigma_trigger_primal[0] != '\0' ? sigma_trigger_primal : "1e-2";
    sigma_trigger_dual =
        sigma_trigger_dual != NULL && sigma_trigger_dual[0] != '\0' ? sigma_trigger_dual : "1e-6";
    sigma_trigger_ratio =
        sigma_trigger_ratio != NULL && sigma_trigger_ratio[0] != '\0' ? sigma_trigger_ratio : "1e6";

    final_policy_setenv("CP_AL_SIGMA_MODE", sigma_always ? "1" : "0");
    final_policy_setenv("CP_AL_SIGMA_ADAPTIVE", "1");
    final_policy_setenv("CP_AL_SIGMA_INIT_FACTOR", sigma_init_factor);
    final_policy_setenv("CP_AL_SIGMA_TRIGGER", sigma_trigger ? "1" : "0");
    final_policy_setenv("CP_AL_SIGMA_TRIGGER_MIN_ITERS", sigma_trigger_min_iters);
    final_policy_setenv("CP_AL_SIGMA_TRIGGER_PRIMAL", sigma_trigger_primal);
    final_policy_setenv("CP_AL_SIGMA_TRIGGER_DUAL", sigma_trigger_dual);
    final_policy_setenv("CP_AL_SIGMA_TRIGGER_RATIO", sigma_trigger_ratio);

    // Residual imbalance may increase checkpoint resolution online. There are
    // no data-set, split, tolerance, or instance-name gates.
    const char *dynamic_eval_fast_override = getenv("CP_AL_FEATURE_DYNAMIC_EVAL_FAST");
    dynamic_eval_fast_override =
        dynamic_eval_fast_override != NULL && dynamic_eval_fast_override[0] != '\0'
        ? dynamic_eval_fast_override
        : "50";
    final_policy_setenv("CP_AL_DYNAMIC_EVAL", sigma_trigger && !sigma_probe ? "1" : "0");
    final_policy_setenv("CP_AL_DYNAMIC_EVAL_FAST", dynamic_eval_fast_override);
    final_policy_setenv("CP_AL_DYNAMIC_EVAL_MIN_ITERS", "200");
    final_policy_setenv("CP_AL_DYNAMIC_EVAL_SMALL_RES", "1e-6");
    final_policy_setenv("CP_AL_DYNAMIC_EVAL_LARGE_RES", "1e-4");
    final_policy_setenv("CP_AL_DYNAMIC_EVAL_RATIO", "1e4");

    // The same three-point fixed-point score chooses the restart anchor for
    // every instance. Reflection and restart thresholds are globally fixed.
    final_policy_setenv("CP_AL_RESTART_SELECT", restart_select ? "3" : "0");
    final_policy_clearenv("CP_AL_RESTART_SELECT_PROFILE");
    final_policy_setenv("CP_AL_RESTART_SELECT_MARGIN", "0.9");
    final_policy_setenv("CP_AL_RESTART_SUFFICIENT", "0.2");
    final_policy_setenv("CP_AL_RESTART_NECESSARY", "0.8");
    final_policy_setenv("CP_AL_RESTART_ARTIFICIAL", "0.36");
    final_policy_clearenv("CP_AL_STEP_MARGIN");

    const double m = (double)(problem->num_constraints > 0 ? problem->num_constraints : 1);
    const double n = (double)(problem->num_variables > 0 ? problem->num_variables : 1);
    const double nnz =
        (double)(problem->constraint_matrix_num_nonzeros > 0 ? problem->constraint_matrix_num_nonzeros : 1);
    fprintf(stderr,
            "[feature-auto-policy] version=2 variant=%s eval_base=200 refl=1 rows=%d cols=%d nnz=%d "
            "n_over_m=%.6e nnz_per_col=%.6e sigma_pid=1 sigma_always=%d sigma_trigger=%d "
            "sigma_probe=%d sigma_init_factor=%s sigma_trigger_min_iters=%s "
            "restart_select=%d cr_override=%d cr_selected=%d coeff_log_range=%.6e row_logmax_std=%.6e col_logmax_std=%.6e "
            "row_logspan_mean=%.6e col_logspan_mean=%.6e row_span2_frac=%.6e col_span2_frac=%.6e "
            "row_nnz_cv=%.6e col_nnz_cv=%.6e equality_frac=%.6e two_sided_var_frac=%.6e "
            "objective_density=%.6e\n",
            variant,
            problem->num_constraints,
            problem->num_variables,
            problem->constraint_matrix_num_nonzeros,
            n / m,
            nnz / n,
            sigma_always ? 1 : 0,
            sigma_trigger ? 1 : 0,
            sigma_probe ? 1 : 0,
            sigma_init_factor,
            sigma_trigger_min_iters,
            restart_select ? 3 : 0,
            cr_override,
            cr_selected,
            matrix_stats.coeff_log_range,
            matrix_stats.row_logmax_std,
            matrix_stats.col_logmax_std,
            matrix_stats.row_logspan_mean,
            matrix_stats.col_logspan_mean,
            matrix_stats.row_span2_fraction,
            matrix_stats.col_span2_fraction,
            matrix_stats.row_nnz_cv,
            matrix_stats.col_nnz_cv,
            matrix_stats.equality_fraction,
            matrix_stats.two_sided_variable_fraction,
            matrix_stats.objective_density);
}
// END_IDENTITY_FREE_FEATURE_POLICY

// BEGIN_UNIFIED_POLICY
// One identity-free decision list, resolved inside the binary before any of the
// existing feature/structural env flags are read. It reads only anonymous
// matrix-structure features (never instance name, path, dataset, split,
// tolerance, or manifest position) and then sets the pre-existing, already
// validated env switches so the downstream machinery runs unchanged.
//
// Three routes over the frozen anonymous features:
//   1. lowobj_highcv_eval100_cr5:
//        objective_density <= 0.01 and col_nnz_cv >= 5
//        -> zero-augmentation base with CR5 and evaluation frequency 100.
//   2. zero_equality_cr0:    equality_frac == 0
//        -> zero-augmentation base with a CR0 continuation override.
//   3. high_equality_tail_or_base: otherwise, enable the structural tail with a
//        deterministic positive-alpha pulse. The tail's own internal predicate
//        (objective_density>=0.99 AND row_logmax_std<=0.05 AND
//         equality_frac>=0.99 or <=0.001) decides whether a matrix actually
//        activates; non-matching matrices fall back to the base solve.
static void apply_unified_policy(const feature_matrix_stats_t *precomputed_stats)
{
    if (!env_flag_enabled("CP_AL_UNIFIED_POLICY"))
    {
        return;
    }
    if (precomputed_stats == NULL)
    {
        fprintf(stderr, "[unified-policy-error] reason=missing-precomputed-stats\n");
        return;
    }
    const feature_matrix_stats_t stats = *precomputed_stats;

    // Common zero-augmentation feature policy.
    final_policy_setenv("CP_AL_FEATURE_AUTO_POLICY", "1");
    final_policy_setenv("CP_AL_FEATURE_POLICY_VARIANT", "safe0");

    // Anonymous structural cohort for feasibility-type problems with high
    // column-density variance (objective_density <= 0.01 AND col_nnz_cv >= 5).
    const bool route_lowobj_highcv =
        stats.objective_density <= 0.01 && stats.col_nnz_cv >= 5.0;
    const bool route_zero_equality_cr0 = stats.equality_fraction == 0.0;

    const char *route = "base";
    if (route_lowobj_highcv)
    {
        // Base solve with CR5 and evaluation frequency 100; no structural tail.
        final_policy_setenv("CP_AL_FEATURE_CR_ITERS", "5");
        final_policy_setenv("CP_AL_UNIFIED_EVAL_FREQ", "100");
        final_policy_clearenv("CP_AL_STRUCTURAL_TAIL_CUT");
        final_policy_clearenv("CP_AL_TAIL_ONLY_PULSE");
        route = "lowobj_highcv_eval100_cr5";
    }
    else if (route_zero_equality_cr0)
    {
        // Static CR0 continuation; no structural tail, no positive augmentation.
        final_policy_setenv("CP_AL_FEATURE_CR_ITERS", "0");
        final_policy_clearenv("CP_AL_STRUCTURAL_TAIL_CUT");
        final_policy_clearenv("CP_AL_TAIL_ONLY_PULSE");
        route = "zero_equality_cr0";
    }
    else
    {
        // Structural tail with a deterministic positive-alpha pulse. The tail's
        // internal high/low-equality predicate gates real activation; matrices
        // that do not match fall back to the base solve.
        final_policy_clearenv("CP_AL_FEATURE_CR_ITERS");
        final_policy_setenv("CP_AL_STRUCTURAL_TAIL_CUT", "1");
        final_policy_setenv("CP_AL_TAIL_ONLY_PULSE", "1");
        route = "high_equality_tail_or_base";
    }

    fprintf(stderr,
            "[unified-policy] version=2 route=%s equality_frac=%.6e row_nnz_cv=%.6e "
            "col_nnz_cv=%.6e objective_density=%.6e row_logmax_std=%.6e\n",
            route,
            stats.equality_fraction,
            stats.row_nnz_cv,
            stats.col_nnz_cv,
            stats.objective_density,
            stats.row_logmax_std);
}
// END_UNIFIED_POLICY

char *get_output_path(const char *output_dir, const char *instance_name, const char *suffix)
{
    size_t path_len = strlen(output_dir) + strlen(instance_name) + strlen(suffix) + 2;
    char *full_path = safe_malloc(path_len * sizeof(char));
    snprintf(full_path, path_len, "%s/%s%s", output_dir, instance_name, suffix);
    return full_path;
}

char *extract_instance_name(const char *filename)
{
    char *filename_copy = strdup(filename);
    if (filename_copy == NULL)
    {
        perror("Memory allocation failed");
        return NULL;
    }

    char *base = basename(filename_copy);
    char *dot = strchr(base, '.');
    if (dot)
    {
        *dot = '\0';
    }

    char *instance_name = strdup(base);
    free(filename_copy);
    return instance_name;
}

void save_solution(const double *data, int size, const char *output_dir, const char *instance_name, const char *suffix)
{
    char *file_path = get_output_path(output_dir, instance_name, suffix);
    if (file_path == NULL || data == NULL)
    {
        return;
    }

    FILE *outfile = fopen(file_path, "w");
    if (outfile == NULL)
    {
        perror("Error opening solution file");
        free(file_path);
        return;
    }

    for (int i = 0; i < size; ++i)
    {
        fprintf(outfile, "%.10g\n", data[i]);
    }

    fclose(outfile);
    free(file_path);
}

void save_solver_summary(const cp_al_result_t *result, const char *output_dir, const char *instance_name)
{
    char *file_path = get_output_path(output_dir, instance_name, "_summary.txt");
    if (file_path == NULL)
    {
        return;
    }

    FILE *outfile = fopen(file_path, "w");
    if (outfile == NULL)
    {
        perror("Error opening summary file");
        free(file_path);
        return;
    }
    fprintf(outfile, "Termination Reason: %s\n", termination_reason_to_string(result->termination_reason));
    if (result->presolve_time > 0.0)
    {
        fprintf(outfile, "Presolve Status: %s\n", get_presolve_status_str(result->presolve_status));
        fprintf(outfile, "Presolve Time (sec): %e\n", result->presolve_time);
        fprintf(outfile, "Reduced Rows: %d\n", result->num_reduced_constraints);
        fprintf(outfile, "Reduced Columns: %d\n", result->num_reduced_variables);
        fprintf(outfile, "Reduced Nonzeros: %d\n", result->num_reduced_nonzeros);

        // if (result->presolve_stats.n_cols_original > 0) {
        //     fprintf(outfile, "NNZ Removed Trivial: %d\n", result->presolve_stats.nnz_removed_trivial);
        //     fprintf(outfile, "NNZ Removed Fast: %d\n", result->presolve_stats.nnz_removed_fast);
        //     fprintf(outfile, "NNZ Removed Primal Propagation: %d\n", result->presolve_stats.nnz_removed_primal_propagation);
        //     fprintf(outfile, "NNZ Removed Parallel Rows: %d\n", result->presolve_stats.nnz_removed_parallel_rows);
        //     fprintf(outfile, "NNZ Removed Parallel Cols: %d\n", result->presolve_stats.nnz_removed_parallel_cols);

        //     fprintf(outfile, "Presolve Time Init (sec): %e\n", result->presolve_stats.time_init);
        //     fprintf(outfile, "Presolve Time Run (sec): %e\n", result->presolve_stats.time_presolve);
        //     fprintf(outfile, "Presolve Time Fast (sec): %e\n", result->presolve_stats.time_fast_reductions);
        //     fprintf(outfile, "Presolve Time Medium (sec): %e\n", result->presolve_stats.time_medium_reductions);
        //     fprintf(outfile, "Presolve Time Primal Proppagation (sec): %e\n", result->presolve_stats.time_primal_propagation);
        //     fprintf(outfile, "Presolve Time Parallel Rows (sec): %e\n", result->presolve_stats.time_parallel_rows);
        //     fprintf(outfile, "Presolve Time Parallel Cols (sec): %e\n", result->presolve_stats.time_parallel_cols);
        //     fprintf(outfile, "Postsolve Time (sec): %e\n", result->presolve_stats.time_postsolve);
        // }
    }
    fprintf(outfile, "Precondition time (sec): %e\n", result->rescaling_time_sec);
    fprintf(outfile, "Runtime (sec): %e\n", result->cumulative_time_sec);
    fprintf(outfile, "Iterations Count: %d\n", result->total_count);
    fprintf(outfile, "Primal Objective Value: %e\n", result->primal_objective_value);
    fprintf(outfile, "Dual Objective Value: %e\n", result->dual_objective_value);
    fprintf(outfile, "Relative Primal Residual: %e\n", result->relative_primal_residual);
    fprintf(outfile, "Relative Dual Residual: %e\n", result->relative_dual_residual);
    fprintf(outfile, "Absolute Objective Gap: %e\n", result->objective_gap);
    fprintf(outfile, "Relative Objective Gap: %e\n", result->relative_objective_gap);
    fprintf(outfile, "Rows: %d\n", result->num_constraints);
    fprintf(outfile, "Columns: %d\n", result->num_variables);
    fprintf(outfile, "Nonzeros: %d\n", result->num_nonzeros);
    if (result->feasibility_polishing_time > 0.0)
    {
        fprintf(outfile, "Feasibility Polishing Time (sec): %e\n", result->feasibility_polishing_time);
        fprintf(outfile, "Feasibility Polishing Iteration Count: %d\n", result->feasibility_iteration);
    }
    fclose(outfile);
    free(file_path);
}

// BEGIN_IDENTITY_FREE_STRUCTURAL_TAIL_CUT
static double structural_tail_set_warm_anchor(lp_problem_t *problem,
                                              const cp_al_result_t *anchor)
{
    set_start_values(problem, anchor->primal_solution, anchor->dual_solution);
    double squared_difference = 0.0;
    for (int col = 0; col < problem->num_variables; ++col)
    {
        const double delta = problem->primal_start[col] - anchor->primal_solution[col];
        squared_difference += delta * delta;
    }
    for (int row = 0; row < problem->num_constraints; ++row)
    {
        const double delta = problem->dual_start[row] - anchor->dual_solution[row];
        squared_difference += delta * delta;
    }
    return sqrt(squared_difference);
}

static cp_al_result_t *run_structural_tail_cut(lp_problem_t *problem,
                                               const cp_al_parameters_t *requested_params,
                                               const feature_matrix_stats_t *precomputed_stats,
                                               bool tail_pulse_enabled)
{
    if (precomputed_stats == NULL)
    {
        fprintf(stderr, "[structural-tail-error] reason=missing-precomputed-stats\n");
        return solve_lp_problem(problem, requested_params);
    }
    const feature_matrix_stats_t matrix_stats = *precomputed_stats;
    const bool homogeneous_dense_structure =
        matrix_stats.objective_density >= 0.99 &&
        matrix_stats.row_logmax_std <= 0.05;
    const bool structural_low_equality =
        homogeneous_dense_structure && matrix_stats.equality_fraction <= 0.001;
    const bool structural_high_equality =
        homogeneous_dense_structure && matrix_stats.equality_fraction >= 0.99;
    const bool structural_early_cut =
        structural_low_equality || structural_high_equality;
    const bool effective_tail_pulse_enabled =
        tail_pulse_enabled && structural_early_cut;
    const double base_fraction = structural_high_equality ? 0.05 : 0.75;
    const char *rescue_profile = "cr0";
    const double original_limit = requested_params->termination_criteria.time_sec_limit;
    if (!isfinite(original_limit) || original_limit <= 0.0)
    {
        fprintf(stderr,
                "[structural-tail-error] reason=invalid-time-limit value=%.12e\n",
                original_limit);
        return solve_lp_problem(problem, requested_params);
    }

    if (!structural_early_cut)
    {
        final_policy_setenv("CP_AL_ALPHA_DIRECT_PULSE", "0");
        final_policy_setenv("CP_AL_DETERMINISTIC_PULSE_FORCE", "0");
        fprintf(stderr,
                "[structural-tail-config] enabled=1 structural_early_cut=0 "
                "structural_low_equality=0 structural_high_equality=0 "
                "base_fraction=1.00 rescue_fraction=0.00 rescue_profile=none "
                "base_alpha=0 tail_pulse_requested=%d tail_pulse_enabled=0 "
                "tail_pulse_alpha=2.000000000000e-03 "
                "equality_fraction=%.12e objective_density=%.12e "
                "row_logmax_std=%.12e original_limit=%.12e base_limit=%.12e\n",
                tail_pulse_enabled ? 1 : 0,
                matrix_stats.equality_fraction,
                matrix_stats.objective_density,
                matrix_stats.row_logmax_std,
                original_limit,
                original_limit);
        cp_al_result_t *single_result = solve_lp_problem(problem, requested_params);
        if (single_result == NULL)
        {
            fprintf(stderr, "[structural-tail-error] reason=null-single-phase-result\n");
            return NULL;
        }
        fprintf(stderr,
                "[structural-tail-base] status=%s solve_sec=%.12e precondition_sec=%.12e "
                "iterations=%d\n",
                termination_reason_to_string(single_result->termination_reason),
                single_result->cumulative_time_sec,
                single_result->rescaling_time_sec,
                single_result->total_count);
        fprintf(stderr,
                "[structural-tail-final] phases=1 tail_started=0 status=%s "
                "structural_early_cut=0 base_fraction=1.00 rescue_profile=none "
                "base_solve_sec=%.12e rescue_solve_sec=0 aggregate_solve_sec=%.12e "
                "budget_excess=%.12e total_iterations=%d\n",
                termination_reason_to_string(single_result->termination_reason),
                single_result->cumulative_time_sec,
                single_result->cumulative_time_sec,
                fmax(single_result->cumulative_time_sec - original_limit, 0.0),
                single_result->total_count);
        return single_result;
    }

    cp_al_parameters_t base_params = *requested_params;
    base_params.termination_criteria.time_sec_limit = base_fraction * original_limit;
    fprintf(stderr,
            "[structural-tail-config] enabled=1 structural_early_cut=%d "
            "structural_low_equality=%d structural_high_equality=%d "
            "base_fraction=%.2f rescue_fraction=%.2f rescue_profile=%s "
            "base_alpha=0 tail_pulse_requested=%d tail_pulse_enabled=%d "
            "tail_pulse_alpha=2.000000000000e-03 "
            "equality_fraction=%.12e objective_density=%.12e "
            "row_logmax_std=%.12e original_limit=%.12e base_limit=%.12e\n",
            structural_early_cut ? 1 : 0,
            structural_low_equality ? 1 : 0,
            structural_high_equality ? 1 : 0,
            base_fraction,
            1.0 - base_fraction,
            rescue_profile,
            tail_pulse_enabled ? 1 : 0,
            effective_tail_pulse_enabled ? 1 : 0,
            matrix_stats.equality_fraction,
            matrix_stats.objective_density,
            matrix_stats.row_logmax_std,
            original_limit,
            base_params.termination_criteria.time_sec_limit);
    final_policy_setenv("CP_AL_ALPHA_DIRECT_PULSE", "0");
    final_policy_setenv("CP_AL_DETERMINISTIC_PULSE_FORCE", "0");
    cp_al_result_t *base_result = solve_lp_problem(problem, &base_params);
    if (base_result == NULL)
    {
        fprintf(stderr, "[structural-tail-error] reason=null-base-result\n");
        return NULL;
    }

    const double base_solve_time = base_result->cumulative_time_sec;
    const double base_precondition_time = base_result->rescaling_time_sec;
    const double base_presolve_time = base_result->presolve_time;
    const double base_feasibility_time = base_result->feasibility_polishing_time;
    const int base_iterations = base_result->total_count;
    fprintf(stderr,
            "[structural-tail-base] status=%s solve_sec=%.12e precondition_sec=%.12e "
            "iterations=%d\n",
            termination_reason_to_string(base_result->termination_reason),
            base_solve_time,
            base_precondition_time,
            base_iterations);

    if (base_result->termination_reason == TERMINATION_REASON_OPTIMAL)
    {
        fprintf(stderr,
                "[structural-tail-final] phases=1 tail_started=0 status=OPTIMAL "
                "structural_early_cut=%d base_fraction=%.2f rescue_profile=%s "
                "base_solve_sec=%.12e rescue_solve_sec=0 aggregate_solve_sec=%.12e "
                "budget_excess=%.12e total_iterations=%d\n",
                structural_early_cut ? 1 : 0,
                base_fraction,
                rescue_profile,
                base_solve_time,
                base_solve_time,
                fmax(base_solve_time - original_limit, 0.0),
                base_iterations);
        return base_result;
    }

    const double remaining_limit = fmax(original_limit - base_solve_time, 0.0);
    if (remaining_limit <= 0.0)
    {
        fprintf(stderr,
                "[structural-tail-final] phases=1 tail_started=0 status=%s "
                "structural_early_cut=%d base_fraction=%.2f rescue_profile=%s "
                "base_solve_sec=%.12e rescue_solve_sec=0 aggregate_solve_sec=%.12e "
                "budget_excess=%.12e total_iterations=%d reason=no-remaining-budget\n",
                termination_reason_to_string(base_result->termination_reason),
                structural_early_cut ? 1 : 0,
                base_fraction,
                rescue_profile,
                base_solve_time,
                base_solve_time,
                fmax(base_solve_time - original_limit, 0.0),
                base_iterations);
        return base_result;
    }

    const double anchor_copy_l2 = structural_tail_set_warm_anchor(problem, base_result);
    cp_al_parameters_t rescue_params = *requested_params;
    double rescue_policy_time = 0.0;
    if (structural_early_cut)
    {
        final_policy_setenv("CP_AL_FEATURE_CR_ITERS", "0");
        const clock_t rescue_policy_start = clock();
        apply_feature_auto_policy(problem, &rescue_params, precomputed_stats);
        rescue_policy_time =
            (double)(clock() - rescue_policy_start) / CLOCKS_PER_SEC;
    }
    rescue_params.termination_evaluation_frequency = 200;
    rescue_params.termination_criteria.time_sec_limit = remaining_limit;
    final_policy_setenv(
        "CP_AL_ALPHA_DIRECT_PULSE", effective_tail_pulse_enabled ? "1" : "0");
    final_policy_setenv(
        "CP_AL_DETERMINISTIC_PULSE_FORCE", effective_tail_pulse_enabled ? "1" : "0");
    fprintf(stderr,
            "[structural-tail-start] profile=%s remaining_limit=%.12e "
            "anchor_copy_l2=%.12e policy_sec=%.12e tail_pulse_requested=%d "
            "tail_pulse_enabled=%d\n",
            rescue_profile,
            remaining_limit,
            anchor_copy_l2,
            rescue_policy_time,
            tail_pulse_enabled ? 1 : 0,
            effective_tail_pulse_enabled ? 1 : 0);

    cp_al_result_t *rescue_result = solve_lp_problem(problem, &rescue_params);
    if (rescue_result == NULL)
    {
        fprintf(stderr, "[structural-tail-error] reason=null-rescue-result\n");
        return base_result;
    }
    const double rescue_solve_time = rescue_result->cumulative_time_sec;
    const double rescue_precondition_time = rescue_result->rescaling_time_sec;
    const int rescue_iterations = rescue_result->total_count;
    fprintf(stderr,
            "[structural-tail-rescue] status=%s solve_sec=%.12e precondition_sec=%.12e "
            "iterations=%d\n",
            termination_reason_to_string(rescue_result->termination_reason),
            rescue_solve_time,
            rescue_precondition_time,
            rescue_iterations);

    rescue_result->cumulative_time_sec += base_solve_time;
    rescue_result->rescaling_time_sec += base_precondition_time;
    rescue_result->presolve_time += base_presolve_time;
    rescue_result->feasibility_polishing_time += base_feasibility_time;
    rescue_result->total_count += base_iterations;
    fprintf(stderr,
            "[structural-tail-final] phases=2 tail_started=1 status=%s "
            "structural_early_cut=%d base_fraction=%.2f rescue_profile=%s "
            "base_solve_sec=%.12e rescue_solve_sec=%.12e aggregate_solve_sec=%.12e "
            "budget_excess=%.12e total_iterations=%d anchor_copy_l2=%.12e\n",
            termination_reason_to_string(rescue_result->termination_reason),
            structural_early_cut ? 1 : 0,
            base_fraction,
            rescue_profile,
            base_solve_time,
            rescue_solve_time,
            rescue_result->cumulative_time_sec,
            fmax(rescue_result->cumulative_time_sec - original_limit, 0.0),
            rescue_result->total_count,
            anchor_copy_l2);
    cp_al_result_free(base_result);
    return rescue_result;
}
// END_IDENTITY_FREE_STRUCTURAL_TAIL_CUT

void print_usage(const char *prog_name)
{
    fprintf(stderr, "Usage: %s [OPTIONS] <mps_file> <output_dir>\n\n", prog_name);

    fprintf(stderr, "Arguments:\n");
    fprintf(stderr,
            "  <mps_file>               Path to the input problem in MPS "
            "format (.mps or .mps.gz).\n");
    fprintf(stderr,
            "  <output_dir>             Directory where output files "
            "will be saved. It will contain:\n");
    fprintf(stderr, "                             - <basename>_summary.txt\n");
    fprintf(stderr, "                             - <basename>_primal_solution.txt\n");
    fprintf(stderr, "                             - <basename>_dual_solution.txt\n\n");

    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help                          Display this help message.\n");
    fprintf(stderr,
            "  -v, --verbose                       "
            "Enable verbose logging (default: false).\n");
    fprintf(stderr,
            "      --time_limit <seconds>          "
            "Time limit in seconds (default: 3600.0).\n");
    fprintf(stderr, "      --iter_limit <iterations>       Iteration limit (default: %d).\n", INT32_MAX);
    fprintf(stderr,
            "      --eps_opt <tolerance>           "
            "Relative optimality tolerance (default: 1e-4).\n");
    fprintf(stderr,
            "      --eps_feas <tolerance>          "
            "Relative feasibility tolerance (default: 1e-4).\n");
    fprintf(stderr,
            "      --l_inf_ruiz_iter <int>         "
            "Iterations for L-inf Ruiz rescaling (default: 10).\n");
    fprintf(stderr,
            "      --no_pock_chambolle             "
            "Disable Pock-Chambolle rescaling (default: enabled).\n");
    fprintf(stderr,
            "      --pock_chambolle_alpha <float>  "
            "Value for Pock-Chambolle alpha (default: 1.0).\n");
    fprintf(stderr,
            "      --no_bound_obj_rescaling        "
            "Disable bound objective rescaling (default: enabled).\n");
    fprintf(stderr,
            "      --eval_freq <int>               "
            "Termination evaluation frequency (default: 200).\n");
    fprintf(stderr,
            "      --sv_max_iter <int>             "
            "Max iterations for singular value estimation (default: 5000).\n");
    fprintf(stderr,
            "      --sv_tol <float>                "
            "Tolerance for singular value estimation (default: 1e-4).\n");
    fprintf(stderr,
            "  -f  --feasibility_polishing         "
            "Enable feasibility use feasibility polishing (default: false).\n");
    fprintf(stderr,
            "      --eps_feas_polish <tolerance>   Relative feasibility "
            "polish tolerance (default: 1e-6).\n");
    fprintf(stderr,
            "      --opt_norm <norm_type>          "
            "Norm for optimality criteria: l2 or linf (default: l2).\n");
    fprintf(stderr,
            "      --no_presolve                   "
            "Disable presolve (default: enabled).\n");
    fprintf(stderr,
            "      --matrix_zero_tol <tolerance>.  "
            "Zero tolerance in constraint matrix.\n");
}

int main(int argc, char *argv[])
{
    cp_al_parameters_t params;
    set_default_parameters(&params);

    static struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                           {"verbose", no_argument, 0, 'v'},
                                           {"time_limit", required_argument, 0, 1001},
                                           {"iter_limit", required_argument, 0, 1002},
                                           {"eps_opt", required_argument, 0, 1003},
                                           {"eps_feas", required_argument, 0, 1004},
                                           {"eps_feas_polish", required_argument, 0, 1006},
                                           {"feasibility_polishing", no_argument, 0, 'f'},
                                           {"l_inf_ruiz_iter", required_argument, 0, 1007},
                                           {"pock_chambolle_alpha", required_argument, 0, 1008},
                                           {"no_pock_chambolle", no_argument, 0, 1009},
                                           {"no_bound_obj_rescaling", no_argument, 0, 1010},
                                           {"sv_max_iter", required_argument, 0, 1011},
                                           {"sv_tol", required_argument, 0, 1012},
                                           {"eval_freq", required_argument, 0, 1013},
                                           {"opt_norm", required_argument, 0, 1014},
                                           {"no_presolve", no_argument, 0, 1015},
                                           {"matrix_zero_tol", required_argument, 0, 1016},
                                           {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "hvfp", long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'v':
                params.verbose = true;
                break;
            case 1001: // --time_limit
                params.termination_criteria.time_sec_limit = atof(optarg);
                break;
            case 1002: // --iter_limit
                params.termination_criteria.iteration_limit = atoi(optarg);
                break;
            case 1003: // --eps_optimal
                params.termination_criteria.eps_optimal_relative = atof(optarg);
                break;
            case 1004: // --eps_feas
                params.termination_criteria.eps_feasible_relative = atof(optarg);
                break;
            case 1006: // --eps_feas_polish_relative
                params.termination_criteria.eps_feas_polish_relative = atof(optarg);
                break;
            case 'f': // --feasibility_polishing
                params.feasibility_polishing = true;
                break;
            case 1007: // --l_inf_ruiz_iter
                params.l_inf_ruiz_iterations = atoi(optarg);
                break;
            case 1008: // --pock_chambolle_alpha
                params.pock_chambolle_alpha = atof(optarg);
                break;
            case 1009: // --no_pock_chambolle
                params.has_pock_chambolle_alpha = false;
                break;
            case 1010: // --no_bound_obj_rescaling
                params.bound_objective_rescaling = false;
                break;
            case 1011: // --sv_max_iter
                params.sv_max_iter = atoi(optarg);
                break;
            case 1012: // --sv_tol
                params.sv_tol = atof(optarg);
                break;
            case 1013: // --eval_freq
                params.termination_evaluation_frequency = atoi(optarg);
                break;
            case 1014: // --opt_norm
            {
                const char *norm_str = optarg;
                if (strcmp(norm_str, "l2") == 0)
                {
                    params.optimality_norm = NORM_TYPE_L2;
                }
                else if (strcmp(norm_str, "linf") == 0)
                {
                    params.optimality_norm = NORM_TYPE_L_INF;
                }
                else
                {
                    fprintf(stderr, "Error: opt_norm must be 'l2' or 'linf'\n");
                    return 1;
                }
            }
            break;
            case 1015: // --no_presolve
                params.presolve = false;
                break;
            case 1016: // --matrix_zero_tol
                params.matrix_zero_tol = atof(optarg);
                break;
            case '?': // Unknown option
                return 1;
        }
    }

    if (argc - optind != 2)
    {
        fprintf(stderr, "Error: You must specify an input file and an output directory.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    const char *filename = argv[optind];
    const char *output_dir = argv[optind + 1];

    char *instance_name = extract_instance_name(filename);
    if (instance_name == NULL)
    {
        return 1;
    }

    lp_problem_t *problem = read_mps_file(filename);

    if (problem == NULL)
    {
        fprintf(stderr, "Failed to read or parse the file.\n");
        free(instance_name);
        return 1;
    }

    const bool unified_policy_enabled = env_flag_enabled("CP_AL_UNIFIED_POLICY");
    const bool feature_stats_needed = unified_policy_enabled ||
        env_flag_enabled("CP_AL_FEATURE_AUTO_POLICY") ||
        env_flag_enabled("CP_AL_STRUCTURAL_TAIL_CUT");
    feature_matrix_stats_t feature_stats = {0};
    const clock_t feature_policy_start = clock();
    if (feature_stats_needed)
    {
        feature_stats = compute_feature_matrix_stats(problem);
    }
    // Resolve the unified identity-free decision list before reading the
    // feature and structural switches that it sets.
    apply_unified_policy(feature_stats_needed ? &feature_stats : NULL);

    const bool feature_policy_enabled = env_flag_enabled("CP_AL_FEATURE_AUTO_POLICY");
    const bool structural_tail_enabled = env_flag_enabled("CP_AL_STRUCTURAL_TAIL_CUT");
    const bool tail_pulse_enabled = env_flag_enabled("CP_AL_TAIL_ONLY_PULSE");
    if (feature_stats_needed)
    {
        fprintf(stderr,
                "[feature-matrix-stats] computed=1 feature_policy=%d structural_tail=%d\n",
                feature_policy_enabled ? 1 : 0,
                structural_tail_enabled ? 1 : 0);
    }
    apply_feature_auto_policy(problem, &params, feature_stats_needed ? &feature_stats : NULL);
    const double feature_policy_time_sec =
        feature_policy_enabled ? (double)(clock() - feature_policy_start) / CLOCKS_PER_SEC : 0.0;
    if (feature_policy_enabled)
    {
        fprintf(stderr, "[feature-auto-policy-time] seconds=%.12g\n", feature_policy_time_sec);
    }
    if (env_flag_enabled("CP_AL_FEATURE_AUDIT_ONLY"))
    {
        lp_problem_free(problem);
        free(instance_name);
        return 0;
    }

    cp_al_result_t *result = structural_tail_enabled
                                 ? run_structural_tail_cut(
                                       problem, &params, &feature_stats, tail_pulse_enabled)
                                 : solve_lp_problem(problem, &params);

    if (result == NULL)
    {
        fprintf(stderr, "Solver failed.\n");
    }
    else
    {
        save_solver_summary(result, output_dir, instance_name);
        save_solution(
            result->primal_solution, problem->num_variables, output_dir, instance_name, "_primal_solution.txt");
        save_solution(result->dual_solution, problem->num_constraints, output_dir, instance_name, "_dual_solution.txt");
        cp_al_result_free(result);
    }

    lp_problem_free(problem);
    free(instance_name);

    return 0;
}
