/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_TRITON_MOVING_AVERAGE_CC_IMPL_H
#define INCLUDED_TORCHDSP_TRITON_MOVING_AVERAGE_CC_IMPL_H

#include "triton_model.h"
#include <torchdsp/triton_moving_average_cc.h>

namespace gr {
namespace torchdsp {

class triton_moving_average_cc_impl : public triton_moving_average_cc
{
private:
    triton_model model_;

public:
    triton_moving_average_cc_impl(
        const std::string& model_name,
        const std::string& triton_url,
        size_t max_batch_size,
        size_t items_per_batch,
        size_t tap_size);
    ~triton_moving_average_cc_impl();

    // Where all the action really happens
    int work(
        int noutput_items,
        gr_vector_const_void_star& input_items,
        gr_vector_void_star& output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_TRITON_MOVING_AVERAGE_CC_IMPL_H */
