/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_TRITON_MOVING_AVERAGE_CC_H
#define INCLUDED_TORCHDSP_TRITON_MOVING_AVERAGE_CC_H

#include <gnuradio/sync_block.h>
#include <torchdsp/api.h>

namespace gr {
namespace torchdsp {

/*!
 * \brief <+description of block+>
 * \ingroup torchdsp
 *
 */
class TORCHDSP_API triton_moving_average_cc : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<triton_moving_average_cc> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of torchdsp::triton_moving_average_cc.
     *
     * To avoid accidental use of raw pointers, torchdsp::triton_moving_average_cc's
     * constructor is in a private implementation
     * class. torchdsp::triton_moving_average_cc::make is the public interface for
     * creating new instances.
     */
    static sptr make(
        const std::string& model_name,
        const std::string& triton_url,
        unsigned int tap_size);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_TRITON_MOVING_AVERAGE_CC_H */
