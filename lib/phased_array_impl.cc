/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "phased_array_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

using input_type = gr_complex;
using output_type = gr_complex;
phased_array::sptr phased_array::make(
    int num_elements,
    float sep_lambda,
    float angle_degrees,
    float scale_db) {
    return gnuradio::make_block_sptr<phased_array_impl>(
        num_elements, sep_lambda, angle_degrees, scale_db);
}


/*
 * The private constructor
 */
phased_array_impl::phased_array_impl(
    int num_elements,
    float sep_lambda,
    float angle_degrees,
    float scale_db)
    : gr::sync_block(
          "phased_array",
          gr::io_signature::make(1, 1, sizeof(gr_complex)),
          gr::io_signature::make(num_elements, num_elements, sizeof(gr_complex))),
      d_num_elements(num_elements),
      d_sep_lambda(sep_lambda),
      d_angle_degrees(angle_degrees),
      d_scale_db(scale_db) {

    static gr_complex j(0, 1);
    float theta = d_angle_degrees * M_PI / 180.0;
    d_scale_lin = pow(10.0, d_scale_db / 20.0);

    d_multiplier.resize(d_num_elements);
    for (int m = 0; m < d_num_elements; m++) {
        d_multiplier[m] =
            d_scale_lin * exp(((float)(-2.0 * M_PI * m * d_sep_lambda * sin(theta))) * j);
    }
}

/*
 * Our virtual destructor.
 */
phased_array_impl::~phased_array_impl() {}

int phased_array_impl::work(
    int noutput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items) {
    auto in = static_cast<const input_type*>(input_items[0]);
    


    for (int m = 0; m < d_num_elements; m++) {
        auto out = static_cast<output_type*>(output_items[m]);
        for (int s = 0; s < noutput_items; s++) {

            out[s] = in[s] * d_multiplier[m];
        }
    }

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace torchdsp */
} /* namespace gr */
