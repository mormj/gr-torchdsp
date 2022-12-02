/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_QUANTIZE_IMPL_H
#define INCLUDED_TORCHDSP_QUANTIZE_IMPL_H

#include <torchdsp/quantize.h>

namespace gr {
namespace torchdsp {

class quantize_impl : public quantize
{
private:
    size_t d_nbits;

public:
    quantize_impl(size_t nbits);
    ~quantize_impl();

    // Where all the action really happens
    int work(
        int noutput_items,
        gr_vector_const_void_star& input_items,
        gr_vector_void_star& output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_QUANTIZE_IMPL_H */
