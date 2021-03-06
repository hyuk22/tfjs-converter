/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';

import {NamedTensorsMap} from '../../data/index';
import {ExecutionContext} from '../../executor';
import {Node} from '../index';

import {OpExecutor} from './types';
import {getParamValue} from './utils';

export let executeOp: OpExecutor = (node: Node, tensorMap: NamedTensorsMap,
                                    context: ExecutionContext):
                                       tfc.Tensor[] => {
  switch (node.op) {
    case 'resizeBilinear': {
      const images =
          getParamValue('images', node, tensorMap, context) as tfc.Tensor;
      const size = getParamValue('size', node, tensorMap, context) as number[];
      const alignCorners =
          getParamValue('alignCorners', node, tensorMap, context) as boolean;
      return [tfc.image.resizeBilinear(
          images as tfc.Tensor3D | tfc.Tensor4D, [size[0], size[1]],
          alignCorners)];
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'image';
