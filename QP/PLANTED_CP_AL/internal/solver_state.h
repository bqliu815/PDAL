/*
Copyright 2025 Haihao Lu
Copyright 2026 Hongpei Li
Modified for the RHR-CP-AL release by Benqi Liu, 2026.

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
void initialize_step_size_and_primal_weight(cp_al_solver_state_t *state,
                                            const cp_al_parameters_t *params);

cp_al_solver_state_t *
initialize_solver_state(const cp_al_parameters_t *params,
                        const qp_problem_t *original_problem,
                        const rescale_info_t *rescale_info);
void cp_al_solver_state_free(cp_al_solver_state_t *state);
void rescale_info_free(rescale_info_t *info);
void update_obj_product(cp_al_solver_state_t *state, double *primal_solution);
double compute_xQx(cp_al_solver_state_t *state, double *primal_sol,
                   double *primal_obj_product);
#ifdef __cplusplus
}
#endif