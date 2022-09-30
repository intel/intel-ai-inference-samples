# Copyright (c) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import json
import numpy as np

import openvino.runtime as ov
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModel
from transformers.onnx.convert import export
from transformers.models.bert import BertOnnxConfig, BertConfig

from pathlib import Path

def make_model(core, model_name, input_shape, device):
    print('Optimizing model in OpenVINO:')
    print(f"{{ origin: '{model_name}', input shape: {input_shape} }}")

    out_dir = Path(__file__).parent / '__modcache__'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model Configuration
    config = BertConfig()
    onnx_config = BertOnnxConfig(config)

    # Download & Convert PyTorch model into ONNX
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    onnx_path = out_dir / "model.onnx"
    export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)

    # Compile OpenVINO model
    model_ir = core.read_model(model=onnx_path)

    for input_layer in model_ir.inputs:
        in_shape = input_layer.partial_shape
        in_shape[0] = ov.Dimension()
        in_shape[1] = ov.Dimension()
        model_ir.reshape({input_layer: in_shape})
    model = core.compile_model(model_ir, device)

    return model

def compute_batch_set(full_batch, batches):
    if batches is None or len(batches) == 0:
        return [full_batch,]

    batches = sorted(batches, reverse=True)
    batch_set = []
    rest_batch = full_batch
    for batch in batches:
        batch_set += [batch] * (rest_batch // batch)
        rest_batch %= batch
        if rest_batch == 0:
            break

    return batch_set

def execute_model(models, inputs, batches, dynamic_shape):
    input_ids = inputs[0]
    attention_masks = inputs[1] if len(inputs) > 1 else None
    token_type_ids = inputs[2] if len(inputs) > 2 else None

    # Join all inputs into 1 torch.Tensor
    all_input_ids = np.concatenate(input_ids, axis=0)

    all_attention_masks = np.concatenate(attention_masks, axis=0) if attention_masks is not None else np.ones(all_input_ids.shape, dtype=all_input_ids.dtype)
    all_token_types = np.concatenate(token_type_ids, axis=0) if token_type_ids is not None else np.zeros(all_input_ids.shape, dtype=all_input_ids.dtype)

    # Split combined inputs into batch set
    full_batch = all_input_ids.shape[0]

    if not dynamic_shape:
        batches = models.keys()

    splits = compute_batch_set(full_batch, batches)

    # numpy.split() workaround - 2-elements result if splits list has 1 element
    if len(splits) > 1:
      indicies = []
      acc =  0
      for s in splits[:-1]:
          acc += s
          indicies.append(acc)

      splitted_input_ids = np.split(all_input_ids, indicies, axis=0)
      splitted_attention_masks = np.split(all_attention_masks, indicies, axis=0)
      splitted_token_types = np.split(all_token_types, indicies, axis=0)
    else:
      splitted_input_ids = [all_input_ids]
      splitted_attention_masks = [all_attention_masks]
      splitted_token_types = [all_token_types]

    # Execute the model
    model_outputs = []
    for i in range(len(splits)):
        in_0 = splitted_input_ids[i]
        in_1 = splitted_attention_masks[i]
        in_2 = splitted_token_types[i]
        inp = {"input_ids" : in_0, "attention_mask" : in_1, "token_type_ids" : in_2}

        model = models[0 if dynamic_shape else splits[i]]
        outs = model(inp)
        tensors = list(outs.values())
        out = tensors[1]
        
        if (len(out.shape) != 2): 
          print(f'Output {list(outs.keys())[1]} shape {out.shape} does not match [x,y]')
          exit()

        model_outputs.append(out)

    # Re-combine results
    full_output = np.concatenate(model_outputs, axis=0)

    if len(input_ids) > 1:
        acc = 0
        indicies = []
        for x in input_ids[:-1]:
            acc += x.shape[0]
            indicies.append(acc)
        return np.split(full_output, indicies, axis=0)
    else:
        return [full_output]

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        self.device = "CPU"

        # Get INPUT0 configuration
        input0_config = pb_utils.get_input_config_by_name(
            self.model_config, "INPUT0")
        seq_length = input0_config['dims'][0]

        self.inputs_num = len(self.model_config['input'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        self.batches = []
        self.dynamic_shape = True

        parameters = self.model_config['parameters']

        if 'origin' in parameters:
          origin = parameters['origin']['string_value']
        else:
          raise pb_utils.TritonModelException("Origin model name should be defined")

        if 'batches' in parameters:
          self.batches = json.loads(parameters['batches']['string_value'])

        if 'dynamic_shape' in parameters:
          self.dynamic_shape = json.loads(parameters['dynamic_shape']['string_value'])

        self.core = ov.Core()
        self.models_cpu = dict()
        # dynamic shapes supported in fp32 mode
        if self.dynamic_shape:
          input_shape = [1, seq_length]
          self.models_cpu[0] = make_model(self.core, origin, input_shape, self.device)
        else:
          if seq_length <= 0:
            raise pb_utils.TritonModelException("Dynamic shapes switched off but input size is not defined")

          if self.batches is None or len(self.batches) == 0:
            self.batches = [1]

          for batch in self.batches:
            input_shape = [batch, seq_length]
            self.models_cpu[batch] = make_model(self.core, origin, input_shape, self.device)

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        # Make the list of inputs in form of ndarrays
        inputs_0 = []
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
            inputs_0.append(in_0)

        inputs = [inputs_0]

        # Attention_mask
        if self.inputs_num > 1:
          inputs_1 = []
          for request in requests:
              # Get INPUT1
              in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1").as_numpy()
              inputs_1.append(in_1)
          inputs.append(inputs_1)

        # Token_type_ids
        if self.inputs_num > 2:
          inputs_2 = []
          for request in requests:
              # Get INPUT2
              in_2 = pb_utils.get_input_tensor_by_name(request, "INPUT2").as_numpy()
              inputs_2.append(in_2)
          inputs.append(inputs_2)

        outputs = execute_model(self.models_cpu, inputs, self.batches, self.dynamic_shape)

        # Convert model outputs to triton responses
        responses = []
        for cur_bert_output in outputs:
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", cur_bert_output.astype(self.output0_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
