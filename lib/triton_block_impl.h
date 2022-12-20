/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_TRITON_BLOCK_IMPL_H
#define INCLUDED_TORCHDSP_TRITON_BLOCK_IMPL_H

#include "shm_utils.h"
#include <http_client.h>
#include <torchdsp/triton_block.h>
#include "triton_model.h"

namespace tc = triton::client;


namespace gr {
namespace torchdsp {

class triton_block_impl : public triton_block
{
private:
    triton_model model_;

    size_t _single_item_size;
    size_t _items_per_inference;

public:
    triton_block_impl(
        const std::string& model_name,
        const size_t max_batch_size,
        const std::string& triton_url,
        const std::vector<int>& input_sizes,
        const std::vector<int>& output_sizes);
    ~triton_block_impl();

    // Where all the action really happens
    int work(
        int noutput_items,
        gr_vector_const_void_star& input_items,
        gr_vector_void_star& output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_TRITON_BLOCK_IMPL_H */
