# WearMask3D
# Copyright 2021 Hanjo Kim and Minsoo Kim. All rights reserved.
# http://github.com/jhh37/wearmask3d
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: rlakswh@gmail.com      (Hanjo Kim)
#         devkim1102@gmail.com   (Minsoo Kim)

import json
from mask_functions import batch_fit_masks
from misc_utils import get_total_file_list, list_split
import multiprocessing


def main_process():
    # get configurations
    with open('config.json') as json_file:
        configs = json.load(json_file)

    dataset_path = configs["srcPath"]
    save_dataset_path = configs["dstPath"]
    overwrite_bool = configs["overwrite"]
    num_procs = configs["numProcs"]
    num_gpus = configs["numGpus"]

    print(f'Overwrite is {overwrite_bool}')

    # get whole file path
    total_file_list, _ = get_total_file_list(dataset_path, save_dataset_path)
    print(f'Total no. of images:{len(total_file_list)}')

    # remain file path
    _, processed_list = get_total_file_list(save_dataset_path, dataset_path)

    if overwrite_bool:
        remaining_file_list = total_file_list
    else:
        remaining_file_list = list(set(total_file_list) - set(processed_list))

    # list split in to N process
    split_lists = list_split(remaining_file_list, num_procs)
    print(f'No. of remaining images:{len(remaining_file_list)}')

    procs = []
    for idx, subj_list in enumerate(split_lists):
        cuda_device = idx % num_gpus
        proc = multiprocessing.Process(target=batch_fit_masks, args=(configs, subj_list, cuda_device))
        proc.daemon = True
        procs.append(proc)
        proc.start()

    # wait until whole subprocess to be end
    for proc in procs:
        proc.join()


if __name__ == '__main__':
    main_process()
