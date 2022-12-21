/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "triton_moving_average_cc_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

using input_type = gr_complex;
using output_type = gr_complex;
triton_moving_average_cc::sptr triton_moving_average_cc::make(
    const std::string& model_name,
    const std::string& triton_url,
    size_t max_batch_size,
    size_t items_per_batch,
    size_t tap_size) {

    return gnuradio::make_block_sptr<triton_moving_average_cc_impl>(
        model_name, triton_url, max_batch_size, items_per_batch, tap_size);
}


/*
 * The private constructor
 */
triton_moving_average_cc_impl::triton_moving_average_cc_impl(
    const std::string& model_name,
    const std::string& triton_url,
    size_t max_batch_size,
    size_t items_per_batch,
    size_t tap_size)
    : gr::sync_block(
          "triton_moving_average_cc",
          gr::io_signature::make(1, 1, sizeof(input_type)),
          gr::io_signature::make(1, 1, sizeof(output_type))),

      model_(model_name, max_batch_size, triton_url) {
    set_output_multiple(items_per_batch - (tap_size - 1)); // hard-coded from config.pbtxt
    set_history(tap_size); // should come from exported model's taps in make_model.py
}

/*
 * Our virtual destructor.
 */
triton_moving_average_cc_impl::~triton_moving_average_cc_impl() {}

int triton_moving_average_cc_impl::work(
    int noutput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items) {

    std::vector<const char*> in_ptrs(1);
    std::vector<char*> out_ptrs(1);


    // we can't do infer_batch here because the reduction in
    // output size due to the 'valid' conv_1d is based on history
    // not applied per batch
    // said differently, the input ptr doesn't jump by the input
    // size between batches
    // for now, just do a bunch of infers in a for loop

    auto batch_size = noutput_items / this->output_multiple();

    for (int i = 0; i < batch_size; i++) {

        auto in = static_cast<const gr_complex*>(input_items[0]) + i * output_multiple();
        auto out = static_cast<gr_complex*>(output_items[0]) + i * output_multiple();

        std::vector<const char*> in_ptrs(1);
        in_ptrs[0] = reinterpret_cast<const char*>(in);

        std::vector<char*> out_ptrs(1);
        out_ptrs[0] = reinterpret_cast<char*>(out);

        model_.infer(in_ptrs, out_ptrs);
    }


    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace torchdsp */
} /* namespace gr */
