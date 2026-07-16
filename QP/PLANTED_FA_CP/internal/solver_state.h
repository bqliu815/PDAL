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
void initialize_step_size_and_primal_weight(fa_cp_solver_state_t *state,
                                            const fa_cp_parameters_t *params);

fa_cp_solver_state_t *
initialize_solver_state(const fa_cp_parameters_t *params,
                        const qp_problem_t *original_problem,
                        const rescale_info_t *rescale_info);
void fa_cp_solver_state_free(fa_cp_solver_state_t *state);
void rescale_info_free(rescale_info_t *info);
void update_obj_product(fa_cp_solver_state_t *state, double *primal_solution);
double compute_xQx(fa_cp_solver_state_t *state, double *primal_sol,
                   double *primal_obj_product);
#ifdef __cplusplus
}
#endif