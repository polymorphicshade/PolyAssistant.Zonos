#
# TODO: figure out how to solve this dependancy issue:
#
# zonos  | Traceback (most recent call last):
# zonos  |   File "/app/gradio_interface.py", line 6, in <module>
# zonos  |     from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
# zonos  |   File "/app/zonos/model.py", line 10, in <module>
# zonos  |     from zonos.autoencoder import DACAutoencoder
# zonos  |   File "/app/zonos/autoencoder.py", line 5, in <module>
# zonos  |     from transformers.models.dac import DacModel
# zonos  |   File "/opt/conda/lib/python3.11/site-packages/transformers/utils/import_utils.py", line 2045, in __getattr__
# zonos  |     module = self._get_module(self._class_to_module[name])
# zonos  |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# zonos  |   File "/opt/conda/lib/python3.11/site-packages/transformers/utils/import_utils.py", line 2075, in _get_module
# zonos  |     raise e
# zonos  |   File "/opt/conda/lib/python3.11/site-packages/transformers/utils/import_utils.py", line 2073, in _get_module
# zonos  |     return importlib.import_module("." + module_name, self.__name__)
# zonos  |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# zonos  |   File "/opt/conda/lib/python3.11/importlib/__init__.py", line 126, in import_module
# zonos  |     return _bootstrap._gcd_import(name[level:], package, level)
# zonos  |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# zonos  |   File "/opt/conda/lib/python3.11/site-packages/transformers/models/dac/modeling_dac.py", line 26, in <module>
# zonos  |     from ...modeling_utils import PreTrainedModel
# zonos  |   File "/opt/conda/lib/python3.11/site-packages/transformers/modeling_utils.py", line 61, in <module>
# zonos  |     from .integrations.flash_attention import flash_attention_forward
# zonos  |   File "/opt/conda/lib/python3.11/site-packages/transformers/integrations/flash_attention.py", line 5, in <module>
# zonos  |     from ..modeling_flash_attention_utils import _flash_attention_forward, flash_attn_supports_top_left_mask
# zonos  |   File "/opt/conda/lib/python3.11/site-packages/transformers/modeling_flash_attention_utils.py", line 36, in <module>
# zonos  |     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
# zonos  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# zonos  |   File "/opt/conda/lib/python3.11/site-packages/flash_attn/__init__.py", line 3, in <module>
# zonos  |     from flash_attn.flash_attn_interface import (
# zonos  |   File "/opt/conda/lib/python3.11/site-packages/flash_attn/flash_attn_interface.py", line 15, in <module>
# zonos  |     import flash_attn_2_cuda as flash_attn_gpu
# zonos  | ImportError: /opt/conda/lib/python3.11/site-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
# zonos exited with code 1
#
# (problem since 6/15/25)
#

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
RUN pip install uv
RUN pip install Flask

RUN apt update && \
    apt install -y espeak-ng && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . ./

RUN uv pip install --system -e . && uv pip install --system -e .[compile]
