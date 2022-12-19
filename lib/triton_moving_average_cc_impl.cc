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
    unsigned int tap_size) {

    auto model = triton_model::make(model_name, 256, triton_url);

    if (model == nullptr)
        throw std::runtime_error("Could not instantiate triton_model");

    return gnuradio::make_block_sptr<triton_moving_average_cc_impl>(model, tap_size);
}


/*
 * The private constructor
 */
triton_moving_average_cc_impl::triton_moving_average_cc_impl(
    std::unique_ptr<triton_model>& model,
    unsigned int tap_size)
    : gr::sync_block(
          "triton_moving_average_cc",
          gr::io_signature::make(1, 1, sizeof(input_type)),
          gr::io_signature::make(1, 1, sizeof(output_type))),

      model_(std::move(model)) {
    set_output_multiple(1024 - (tap_size-1)); // hard-coded from config.pbtxt
    set_history(tap_size);     // should come from exported model's taps in make_model.py
}

/*
 * Our virtual destructor.
 */
triton_moving_average_cc_impl::~triton_moving_average_cc_impl() {}

int triton_moving_average_cc_impl::work(
    int noutput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items) {

    std::vector<const char*> in_ptrs;
    in_ptrs.push_back(static_cast<const char*>(input_items[0]));

    std::vector<char*> out_ptrs;
    out_ptrs.push_back(static_cast<char*>(output_items[0]));

    // num_items_per_patch is fixed.
    auto batch_size = noutput_items / this->output_multiple();
    model_->infer_batch(in_ptrs, out_ptrs, batch_size);

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace torchdsp */
} /* namespace gr */
