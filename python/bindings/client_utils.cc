/*
 * Copyright 2022 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <http_client.h>
#include <rapidjson/document.h>

namespace tc = triton::client;

void bind_client_utils(py::module& m) {

    m.def("load_model", [](const std::string& triton_url, const std::string& model_name){
        std::unique_ptr<tc::InferenceServerHttpClient> client;
        tc::Error err;

        err = tc::InferenceServerHttpClient::Create(&client, triton_url, false);
        bool is_live;
        err = client->IsServerLive(&is_live);
        if (!(err.IsOk() && is_live))
        {
            throw std::runtime_error("Triton Server is not live");
        }
        bool is_ready;
        err = client->IsServerReady(&is_ready);
        if (!(err.IsOk() && is_ready))
        {
            throw std::runtime_error("Triton Server is not ready");
        }
        err =  client->LoadModel(model_name);
        if (!(err.IsOk()))
        {
            throw std::runtime_error("Unabled to Load Model");
        }
    });

    m.def("unload_model", [](const std::string& triton_url, const std::string& model_name){
        std::unique_ptr<tc::InferenceServerHttpClient> client;
        tc::Error err;

        err = tc::InferenceServerHttpClient::Create(&client, triton_url, false);
        bool is_live;
        err = client->IsServerLive(&is_live);
        if (!(err.IsOk() && is_live))
        {
            throw std::runtime_error("Triton Server is not live");
        }
        bool is_ready;
        err = client->IsServerReady(&is_ready);
        if (!(err.IsOk() && is_ready))
        {
            throw std::runtime_error("Triton Server is not ready");
        }
        err =  client->UnloadModel(model_name);
        if (!(err.IsOk()))
        {
            throw std::runtime_error("Unabled to Load Model");
        }
    });

    m.def("get_loaded_models", [](const std::string& triton_url){
        std::unique_ptr<tc::InferenceServerHttpClient> client;
        tc::Error err;

        err = tc::InferenceServerHttpClient::Create(&client, triton_url, false);
        bool is_live;
        err = client->IsServerLive(&is_live);
        if (!(err.IsOk() && is_live))
        {
            throw std::runtime_error("Triton Server is not live");
        }
        bool is_ready;
        err = client->IsServerReady(&is_ready);
        if (!(err.IsOk() && is_ready))
        {
            throw std::runtime_error("Triton Server is not ready");
        }

        std::string model_repository_index;
        err = client->ModelRepositoryIndex(&model_repository_index);
        if (!(err.IsOk()))
        {
            throw std::runtime_error("Unabled to Get Repository Contents");
        }
        std::cout << model_repository_index << std::endl;
        // rapidjson::Document json_metadata;
        // json_metadata.Parse(model_repository_index.c_str());
        // for (auto& value : json_metadata.GetArray()) {
        //     std::cout << value << std::endl;
        // }

    });

    m.def("unload_all_models", [](const std::string& triton_url){
        std::unique_ptr<tc::InferenceServerHttpClient> client;
        tc::Error err;

        err = tc::InferenceServerHttpClient::Create(&client, triton_url, false);
        bool is_live;
        err = client->IsServerLive(&is_live);
        if (!(err.IsOk() && is_live))
        {
            throw std::runtime_error("Triton Server is not live");
        }
        bool is_ready;
        err = client->IsServerReady(&is_ready);
        if (!(err.IsOk() && is_ready))
        {
            throw std::runtime_error("Triton Server is not ready");
        }

        std::string model_repository_index;
        err = client->ModelRepositoryIndex(&model_repository_index);
        if (!(err.IsOk()))
        {
            throw std::runtime_error("Unabled to Get Repository Contents");
        }
        rapidjson::Document json_metadata;
        json_metadata.Parse(model_repository_index.c_str());
        for (auto& value : json_metadata.GetArray()) {
             client->UnloadModel(value["name"].GetString());
        }

    });


}
