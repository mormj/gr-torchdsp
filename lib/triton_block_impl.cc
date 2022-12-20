/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "triton_block_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

using input_type = gr_complex;
using output_type = gr_complex;

triton_block::sptr triton_block::make(
    const std::string& model_name,
    const size_t max_batch_size,
    const std::string& triton_url,
    const std::vector<int>& input_sizes,
    const std::vector<int>& output_sizes) {

    // We ask Triton for what the input signature is if one is not providided.
    // We sometimes need to provide one because complexf is not supported in Triton.
    return gnuradio::make_block_sptr<triton_block_impl>(
        model_name, max_batch_size, triton_url, input_sizes, output_sizes);
}

/*
 * The private constructor
 */
triton_block_impl::triton_block_impl(
    const std::string& model_name,
    const size_t max_batch_size,
    const std::string& triton_url,
    const std::vector<int>& input_sizes,
    const std::vector<int>& output_sizes)
    : gr::sync_block(
          "triton_block",
          gr::io_signature::make(0, 0, 0),
          gr::io_signature::make(0, 0, 0)),
      model_(
          model_name,
          max_batch_size,
          triton_url) // this is invoked after calling sync_block constructor.
{
    set_input_signature(gr::io_signature::makev(
        1, -1, input_sizes.size() == 0 ? model_.get_input_signature() : input_sizes));
    set_output_signature(gr::io_signature::makev(
        1, -1, output_sizes.size() == 0 ? model_.get_output_signature() : output_sizes));

    _items_per_inference = model_.get_output_sizes()[0];
    _single_item_size = output_sizes[0]; // model_.get_output_signature()[0];

    set_output_multiple(_items_per_inference / _single_item_size);
    
    std::cout << "Instantiated block" << std::endl;
}

/*
 * Our virtual destructor.
 */
triton_block_impl::~triton_block_impl() {}

int triton_block_impl::work(
    int noutput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items) {

    std::vector<const char*> in_ptrs;
    for (const auto& item : input_items)
        in_ptrs.push_back(static_cast<const char*>(item));

    std::vector<char*> out_ptrs;
    for (const auto& item : output_items)
        out_ptrs.push_back(static_cast<char*>(item));

    // num_items_per_patch is fixed.
    auto num_items_per_batch = _items_per_inference / _single_item_size;
    auto batch_size = noutput_items / num_items_per_batch;

    model_.infer_batch(in_ptrs, out_ptrs, batch_size);


    // std::cout << fmt::format("noutput_items: {}, batch_size: {}, num_items_per_batch:
    // {}", noutput_items, batch_size, num_items_per_batch) << std::endl;; auto in0 =
    // static_cast<const gr_complex*>(input_items[0]); auto out =
    // static_cast<gr_complex*>(output_items[0]); for (int i=0; i< noutput_items; i++) {
    //     out[i] = in[i];
    // }

    return noutput_items;
}

} // namespace torchdsp
} /* namespace gr */
