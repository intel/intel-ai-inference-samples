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

import torch
from torch.utils import dlpack
import triton_python_backend_utils as pb_utils
import intel_extension_for_pytorch as ipex
import json

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
        self.device = torch.device('cpu')
               
        print(f"\n{{ device name : {self.device}}}\n")
        
        # Get INPUT0 configuration
        input0_config = pb_utils.get_input_config_by_name(
            self.model_config, "input__0")
        
        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT__0")
       
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
       
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        model = model.eval()
        image_s = torch.rand(1,3,224,224)
        
        print('Optimizing model in IPEX:')
        try:
          model = ipex.optimize(model)  
          with torch.no_grad():
              model = torch.jit.trace(model, image_s)
              self.model = torch.jit.freeze(model)
        except Exception as e: print(e)
        
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
        # Make the list of inputs in form of torch.Tensor
        print('Start of execute')
        inputs = []
        responses = []
        for request in requests:
            # Get INPUT0        
            in_0_cpu = pb_utils.get_input_tensor_by_name(request, "input__0")
            in_0_cpu = torch.Tensor(in_0_cpu.as_numpy()).to(self.device)
            print('\nINPUT : ', in_0_cpu)
            
            with torch.no_grad():
                result = self.model(in_0_cpu)
            out_tensor_0 = pb_utils.Tensor("OUTPUT__0", result.cpu().numpy())
                        
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)                      
        
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
