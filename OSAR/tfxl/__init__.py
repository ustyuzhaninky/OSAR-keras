# coding=utf-8
# Copyright 2020 Konstantin Ustyuzhanin.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
name = 'tfxl'

from OSAR.tfxl.scale import *
from OSAR.tfxl.memory import *
from OSAR.tfxl.pos_embed import *
from .rel_bias import *
from OSAR.tfxl.rel_multi_head import *
from OSAR.tfxl.loader import *
from OSAR.tfxl.transformer_xl import *
from OSAR.tfxl.sequence import *
from OSAR.tfxl.softmax import *
from OSAR.tfxl.embedding import *
from OSAR.tfxl.feed_forward import *
from OSAR.tfxl.layer_normalization import *
