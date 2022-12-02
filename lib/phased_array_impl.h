/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_PHASED_ARRAY_IMPL_H
#define INCLUDED_TORCHDSP_PHASED_ARRAY_IMPL_H

#include <torchdsp/phased_array.h>
#include <vector>

namespace gr {
namespace torchdsp {

class phased_array_impl : public phased_array
{
private:
    int d_num_elements;
    float d_sep_lambda;
    float d_angle_degrees;
    float d_scale_db;
    float d_scale_lin;

    std::vector<gr_complex> d_multiplier;

public:
    phased_array_impl(
        int num_elements,
        float sep_lambda,
        float angle_degrees,
        float scale_db);
    ~phased_array_impl();

    // Where all the action really happens
    int work(
        int noutput_items,
        gr_vector_const_void_star& input_items,
        gr_vector_void_star& output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_PHASED_ARRAY_IMPL_H */
