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

#include "mps_parser.h"
#include "fa_cp.h"
#include "solver.h"
#include "utils.h"
#include <getopt.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_output_path(const char *output_dir, const char *instance_name,
                      const char *suffix) {
  size_t path_len =
      strlen(output_dir) + strlen(instance_name) + strlen(suffix) + 2;
  char *full_path = safe_malloc(path_len * sizeof(char));
  snprintf(full_path, path_len, "%s/%s%s", output_dir, instance_name, suffix);
  return full_path;
}

bool parse_sigma_update_rule(const char *value, sigma_update_rule_t *rule) {
  if (strcmp(value, "fixed") == 0 || strcmp(value, "none") == 0) {
    *rule = FA_CP_SIGMA_UPDATE_FIXED;
    return true;
  }
  if (strcmp(value, "residual") == 0) {
    *rule = FA_CP_SIGMA_UPDATE_RESIDUAL_BALANCE;
    return true;
  }
  if (strcmp(value, "kkt") == 0) {
    *rule = FA_CP_SIGMA_UPDATE_KKT_BALANCE;
    return true;
  }
  if (strcmp(value, "displacement") == 0) {
    *rule = FA_CP_SIGMA_UPDATE_DISPLACEMENT;
    return true;
  }
  if (strcmp(value, "progress") == 0) {
    *rule = FA_CP_SIGMA_UPDATE_PROGRESS;
    return true;
  }
  if (strcmp(value, "greedy") == 0) {
    *rule = FA_CP_SIGMA_UPDATE_GREEDY_SCORE;
    return true;
  }
  if (strcmp(value, "guarded") == 0 || strcmp(value, "zero-aware") == 0 ||
      strcmp(value, "zero_aware") == 0) {
    *rule = FA_CP_SIGMA_UPDATE_GUARDED;
    return true;
  }
  if (strcmp(value, "balanced_guarded") == 0 ||
      strcmp(value, "balanced-guarded") == 0 ||
      strcmp(value, "guarded_balanced") == 0 ||
      strcmp(value, "scaled_guarded") == 0) {
    *rule = FA_CP_SIGMA_UPDATE_BALANCED_GUARDED;
    return true;
  }
  return false;
}

char *extract_instance_name(const char *filename) {
  char *filename_copy = strdup(filename);
  if (filename_copy == NULL) {
    perror("Memory allocation failed");
    return NULL;
  }

  char *base = basename(filename_copy);
  char *dot = strchr(base, '.');
  if (dot) {
    *dot = '\0';
  }

  char *instance_name = strdup(base);
  free(filename_copy);
  return instance_name;
}

void save_solution(const double *data, int size, const char *output_dir,
                   const char *instance_name, const char *suffix) {
  char *file_path = get_output_path(output_dir, instance_name, suffix);
  if (file_path == NULL || data == NULL) {
    return;
  }

  FILE *outfile = fopen(file_path, "w");
  if (outfile == NULL) {
    perror("Error opening solution file");
    free(file_path);
    return;
  }

  for (int i = 0; i < size; ++i) {
    fprintf(outfile, "%.10g\n", data[i]);
  }

  fclose(outfile);
  free(file_path);
}

void save_solver_summary(const fa_cp_result_t *result, const char *output_dir,
                         const char *instance_name) {
  char *file_path = get_output_path(output_dir, instance_name, "_summary.txt");
  if (file_path == NULL) {
    return;
  }

  FILE *outfile = fopen(file_path, "w");
  if (outfile == NULL) {
    perror("Error opening summary file");
    free(file_path);
    return;
  }
  fprintf(outfile, "Termination Reason: %s\n",
          termination_reason_to_string(result->termination_reason));
  fprintf(outfile, "Runtime (sec): %e\n", result->cumulative_time_sec);
  fprintf(outfile, "Iterations Count: %d\n", result->total_count);
  if (result->total_inner_count > 0) {
    fprintf(outfile, "Inner Iterations Count: %d\n", result->total_inner_count);
  }
  fprintf(outfile, "Primal Objective Value: %e\n",
          result->primal_objective_value);
  fprintf(outfile, "Dual Objective Value: %e\n", result->dual_objective_value);
  fprintf(outfile, "Relative Primal Residual: %e\n",
          result->relative_primal_residual);
  fprintf(outfile, "Relative Dual Residual: %e\n",
          result->relative_dual_residual);
  fprintf(outfile, "Absolute Objective Gap: %e\n", result->objective_gap);
  fprintf(outfile, "Relative Objective Gap: %e\n",
          result->relative_objective_gap);
  fprintf(outfile, "Final AL Sigma: %e\n", result->final_al_sigma);
  fprintf(outfile, "Sigma Update Rule: %s\n",
          sigma_update_rule_to_string(result->sigma_update_rule));
  fprintf(outfile, "Sigma Update Count: %d\n", result->sigma_update_count);
  fprintf(outfile, "Rows: %d\n", result->num_constraints);
  fprintf(outfile, "Columns: %d\n", result->num_variables);
  fprintf(outfile, "Nonzeros: %d\n", result->num_nonzeros);

  if (result->feasibility_polishing_time > 0.0) {
    fprintf(outfile, "Feasibility Polishing Time (sec): %e\n",
            result->feasibility_polishing_time);
    fprintf(outfile, "Feasibility Polishing Iteration Count: %d\n",
            result->feasibility_iteration);
  }
  fclose(outfile);
  free(file_path);
}

void print_usage(const char *prog_name) {
  fprintf(stderr, "Usage: %s [OPTIONS] <mps_file> <output_dir>\n\n", prog_name);

  fprintf(stderr, "Arguments:\n");
  fprintf(stderr, "  <mps_file>               Path to the input problem in MPS "
                  "format (.mps .QPS or .mps.gz).\n");
  fprintf(stderr, "  <output_dir>             Directory where output files "
                  "will be saved. It will contain:\n");
  fprintf(stderr, "                             - <basename>_summary.txt\n");
  fprintf(stderr,
          "                             - <basename>_primal_solution.txt\n");
  fprintf(stderr,
          "                             - <basename>_dual_solution.txt\n\n");

  fprintf(stderr, "Options:\n");
  fprintf(stderr,
          "  -h, --help                          Display this help message.\n");
  fprintf(stderr, "  -v, --verbose <int>                 "
                  "Verbosity level: 0=silent, 1=summary, 2=detailed (default: 1).\n");
  fprintf(stderr, "      --time_limit <seconds>          "
                  "Time limit in seconds (default: 3600.0).\n");
  fprintf(
      stderr,
      "      --iter_limit <iterations>       Iteration limit (default: %d).\n",
      INT32_MAX);
  fprintf(stderr, "      --eps_opt <tolerance>           "
                  "Relative optimality tolerance (default: 1e-4).\n");
  fprintf(stderr, "      --eps_feas <tolerance>          "
                  "Relative feasibility tolerance (default: 1e-4).\n");
  fprintf(stderr, "      --eps_infeas_detect <tolerance> "
                  "Infeasibility detection tolerance (default: 1e-10).\n");
  fprintf(stderr, "      --l_inf_ruiz_iter <int>         "
                  "Iterations for L-inf Ruiz rescaling (default: 10).\n");
  fprintf(stderr, "      --no_pock_chambolle             "
                  "Disable Pock-Chambolle rescaling (default: enabled).\n");
  fprintf(stderr, "      --pock_chambolle_alpha <float>  "
                  "Value for Pock-Chambolle alpha (default: 1.0).\n");
  fprintf(stderr, "      --no_bound_obj_rescaling        "
                  "Disable bound objective rescaling (default: enabled).\n");
  fprintf(stderr, "      --no_al_qp                      "
                  "Disable the AL primal-dual update and use the "
                  "non-augmented baseline branch.\n");
  fprintf(stderr, "      --al_sigma <float>              "
                  "AL penalty parameter used in the full AL primal subproblem"
                  " (default: 0).\n");
  fprintf(stderr, "      --sigma_update <rule>           "
                  "Restart-only sigma rule: fixed, residual, kkt, "
                  "displacement, progress, greedy, guarded, or "
                  "balanced_guarded (default: guarded).\n");
  fprintf(stderr, "      --sigma_update_gain <float>     "
                  "Log-domain damping gain for sigma updates (default: 0.5).\n");
  fprintf(stderr, "      --sigma_clip_low <float>        "
                  "Lower per-restart multiplicative clip (default: 0.6).\n");
  fprintf(stderr, "      --sigma_clip_high <float>       "
                  "Upper per-restart multiplicative clip (default: 1.7).\n");
  fprintf(stderr, "      --sigma_min <float>             "
                  "Minimum positive sigma for adaptive rules (default: 1e-8).\n");
  fprintf(stderr, "      --sigma_max <float>             "
                  "Maximum sigma for adaptive rules (default: 10).\n");
  fprintf(stderr, "      --sigma_rollback_tol <float>    "
                  "Rollback to the best sigma if restart score worsens by this "
                  "factor (default: 1.25; nonpositive disables rollback).\n");
  fprintf(stderr, "      --sigma_zero_threshold <float>  "
                  "Set guarded sigma to exactly zero below this value (default: 1e-8).\n");
  fprintf(stderr, "      --sigma_activation_threshold <float> "
                  "Primal bottleneck ratio needed to activate guarded sigma (default: 2).\n");
  fprintf(stderr, "      --sigma_normal_motion_threshold <float> "
                  "Minimum ||A dx||/||dx|| for guarded activation (default: 1e-10).\n");
  fprintf(stderr, "      --sigma_activation_value <float> "
                  "Positive sigma used when guarded mode activates from zero; "
                  "for balanced_guarded this is theta in sigma = theta * dual_step "
                  "(default: 0.1).\n");
  fprintf(stderr, "      --eval_freq <int>               "
                  "Termination evaluation frequency (default: 200).\n");
  fprintf(stderr, "      --trajectory_csv <path>         "
                  "Write iteration trajectory CSV for plotting.\n");
  fprintf(stderr,
          "      --sv_max_iter <int>             "
          "Max iterations for singular value estimation (default: 5000).\n");
  fprintf(stderr, "      --sv_tol <float>                "
                  "Tolerance for singular value estimation (default: 1e-4).\n");
  // fprintf(stderr, "  -f  --feasibility_polishing         "
  //                 "Enable feasibility use feasibility polishing (default:
  //                 false).\n");
  // fprintf(stderr, "      --eps_feas_polish <tolerance>   Relative feasibility
  // "
  //                 "polish tolerance (default: 1e-6).\n");
  fprintf(stderr,
          "      --opt_norm <norm_type>          "
          "Norm for optimality criteria: l2 or linf (default: linf).\n");
}

int run_fa_cp(int argc, char *argv[]) {
  fa_cp_parameters_t params;
  set_default_parameters(&params);

  static struct option long_options[] = {
      {"help", no_argument, 0, 'h'},
      {"verbose", required_argument, 0, 'v'},
      {"time_limit", required_argument, 0, 1001},
      {"iter_limit", required_argument, 0, 1002},
      {"eps_opt", required_argument, 0, 1003},
      {"eps_feas", required_argument, 0, 1004},
      {"eps_infeas_detect", required_argument, 0, 1005},
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
      {"no_al_qp", no_argument, 0, 1015},
      {"al_sigma", required_argument, 0, 1016},
      {"trajectory_csv", required_argument, 0, 1017},
      {"sigma_update", required_argument, 0, 1018},
      {"sigma_update_gain", required_argument, 0, 1019},
      {"sigma_clip_low", required_argument, 0, 1020},
      {"sigma_clip_high", required_argument, 0, 1021},
      {"sigma_min", required_argument, 0, 1022},
      {"sigma_max", required_argument, 0, 1023},
      {"sigma_rollback_tol", required_argument, 0, 1024},
      {"sigma_zero_threshold", required_argument, 0, 1025},
      {"sigma_activation_threshold", required_argument, 0, 1026},
      {"sigma_normal_motion_threshold", required_argument, 0, 1027},
      {"sigma_activation_value", required_argument, 0, 1028},
      {0, 0, 0, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "hv:fp", long_options, NULL)) != -1) {
    switch (opt) {
    case 'h':
      print_usage(argv[0]);
      return 0;
    case 'v':
      params.verbose = atoi(optarg);
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
    case 1005: // --eps_infeas_detect
      params.termination_criteria.eps_infeasible = atof(optarg);
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
      if (strcmp(norm_str, "l2") == 0) {
        params.optimality_norm = NORM_TYPE_L2;
      } else if (strcmp(norm_str, "linf") == 0) {
        params.optimality_norm = NORM_TYPE_L_INF;
      } else {
        fprintf(stderr, "Error: opt_norm must be 'l2' or 'linf'\n");
        return 1;
      }
    } break;
    case 1015: // --no_al_qp
      params.use_al_qp = false;
      break;
    case 1016: // --al_sigma
      params.al_sigma = atof(optarg);
      break;
    case 1017: // --trajectory_csv
      params.trajectory_csv = optarg;
      break;
    case 1018: // --sigma_update
      if (!parse_sigma_update_rule(optarg, &params.sigma_update_rule)) {
        fprintf(stderr,
                "Error: sigma_update must be fixed, residual, kkt, "
                "displacement, progress, greedy, guarded, or balanced_guarded\n");
        return 1;
      }
      break;
    case 1019: // --sigma_update_gain
      params.sigma_update_gain = atof(optarg);
      break;
    case 1020: // --sigma_clip_low
      params.sigma_clip_low = atof(optarg);
      break;
    case 1021: // --sigma_clip_high
      params.sigma_clip_high = atof(optarg);
      break;
    case 1022: // --sigma_min
      params.sigma_min = atof(optarg);
      break;
    case 1023: // --sigma_max
      params.sigma_max = atof(optarg);
      break;
    case 1024: // --sigma_rollback_tol
      params.sigma_rollback_tol = atof(optarg);
      break;
    case 1025: // --sigma_zero_threshold
      params.sigma_zero_threshold = atof(optarg);
      break;
    case 1026: // --sigma_activation_threshold
      params.sigma_activation_threshold = atof(optarg);
      break;
    case 1027: // --sigma_normal_motion_threshold
      params.sigma_normal_motion_threshold = atof(optarg);
      break;
    case 1028: // --sigma_activation_value
      params.sigma_activation_value = atof(optarg);
      break;
    case '?': // Unknown option
      return 1;
    }
  }

  if (argc - optind != 2) {
    fprintf(
        stderr,
        "Error: You must specify an input file and an output directory.\n\n");
    print_usage(argv[0]);
    return 1;
  }

  const char *filename = argv[optind];
  const char *output_dir = argv[optind + 1];

  char *instance_name = extract_instance_name(filename);
  if (instance_name == NULL) {
    return 1;
  }

  qp_problem_t *problem = read_mps_file(filename);

  if (problem == NULL) {
    fprintf(stderr, "Failed to read or parse the file.\n");
    free(instance_name);
    return 1;
  }
  fa_cp_result_t *result = optimize(&params, problem);

  if (result == NULL) {
    fprintf(stderr, "Solver failed.\n");
  } else {
    save_solver_summary(result, output_dir, instance_name);
    save_solution(result->primal_solution, problem->num_variables, output_dir,
                  instance_name, "_primal_solution.txt");
    save_solution(result->dual_solution, problem->num_constraints, output_dir,
                  instance_name, "_dual_solution.txt");
    fa_cp_result_free(result);
  }

  qp_problem_free(problem);
  free(instance_name);

  return 0;
}

int main(int argc, char *argv[]) { return run_fa_cp(argc, argv); }
