/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_QUANTIZE_H
#define INCLUDED_TORCHDSP_QUANTIZE_H

#include <gnuradio/sync_block.h>
#include <torchdsp/api.h>

namespace gr {
namespace torchdsp {

/*!
 * \brief <+description of block+>
 * \ingroup torchdsp
 *
 */
class TORCHDSP_API quantize : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<quantize> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of torchdsp::quantize.
     *
     * To avoid accidental use of raw pointers, torchdsp::quantize's
     * constructor is in a private implementation
     * class. torchdsp::quantize::make is the public interface for
     * creating new instances.
     */
    static sptr make(size_t nbits);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_QUANTIZE_H */
