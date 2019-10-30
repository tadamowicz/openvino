// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if GNA_LIB_VER == 2
#define nLayerKind operation
#define intel_layer_kind_t gna_layer_operation
#define intel_gna_proc_t uint32_t
#endif
