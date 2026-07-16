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

#ifdef __cplusplus
extern "C"
{
#endif

    cp_al_result_t *optimize(const cp_al_parameters_t *params, lp_problem_t *original_problem);

#ifdef __cplusplus
}
#endif
