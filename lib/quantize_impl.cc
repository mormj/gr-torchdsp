/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "quantize_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

using input_type = gr_complex;
using output_type = gr_complex;
quantize::sptr quantize::make(size_t nbits) {
    return gnuradio::make_block_sptr<quantize_impl>(nbits);
}


/*
 * The private constructor
 */
quantize_impl::quantize_impl(size_t nbits)
    : gr::sync_block(
          "quantize",
          gr::io_signature::make(1, 1, sizeof(input_type)),
          gr::io_signature::make(1, 1, sizeof(output_type))),
      d_nbits(nbits) {}

/*
 * Our virtual destructor.
 */
quantize_impl::~quantize_impl() {}

int quantize_impl::work(
    int noutput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items) {
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    float mult = pow(2.0, (d_nbits-1));

    for (int n = 0; n < noutput_items; n++) {
        auto rr = real(in[n]) * mult;
        rr = (rr >= 0) ? rr + 0.5 : rr - 0.5;
        rr = (int)rr;
        rr /= mult;

        auto ii = imag(in[n]) * mult;
        ii = (ii >= 0) ? ii + 0.5 : ii - 0.5;
        ii = (int)ii;
        ii /= mult;

        out[n] = gr_complex(rr, ii);
    }


    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace torchdsp */
} /* namespace gr */
