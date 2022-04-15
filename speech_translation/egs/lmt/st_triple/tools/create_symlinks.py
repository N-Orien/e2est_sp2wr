"""
Organize multilingual data to prepare for training
"""
import os
import re
import shutil
import json
import subprocess
import argparse

SPLITS = ['train_sp', 'dev', 'test']

def get_info(use_lid=False, use_joint_dict=False):
    
    prefix = 'dict1'
    if not use_joint_dict:
        prefix = 'dict2'
    
    return prefix


def create_data_links_jsons(input_dir, output_dir, 
                            use_lid=False, use_joint_dict=True,
                            nbpe_src=8000, nbpe_wrsrc=8000, nbpe=8000):
    """
    Create symbolic links to save jsons in the following structure: 
        output_dir/tgt_langs/use_${prefix}/src${nbpe_src}_wrsrc{nbpe_wrsrc}_tgt${nbpe}/${split}/${lang_pair}.json
        where: 
            - ${split} is "train_sp", "dev", "tst-COMMON", or "tst-HE"
            - ${lang_pair} is "en-de", "en-es", etc.
    """
    prefix = get_info(use_lid=use_lid, use_joint_dict=use_joint_dict)
    output_dir = os.path.join(output_dir, f'use_{prefix}', f'src{nbpe_src}_wrsrc{nbpe_wrsrc}_tgt{nbpe}')

    for s in SPLITS:
        os.makedirs(os.path.join(output_dir, s), exist_ok=True)
        if use_joint_dict:
            fname = f'data_{prefix}_bpe{nbpe}_tc.json'
        else:
            raise NotImplementedError
            fname = f'data_{prefix}_bpe_src{nbpe_src}lc.rm_tgt{nbpe}tc.json'
        src = os.path.join(input_dir, f'{s}.en', "deltafalse", fname)
        dst = os.path.join(output_dir, s, 'ja-en.json')
        print('{} -> {}'.format(src, dst))
        subprocess.call(["ln", "-s", src, dst])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='./dump', type=str, 
                        help='Path to directory where features are saved')
    parser.add_argument('--output-dir', type=str,
                        help='Path to directory to save symlinks')
    parser.add_argument('--use-lid', action='store_true',
                        help='Use language ID in the target sequence')
    parser.add_argument('--use-joint-dict', action='store_true',
                        help='Use joint dictionary for source and target')
    parser.add_argument('--nbpe', type=int, default=8000)
    parser.add_argument('--nbpe-src', type=int, default=8000)
    parser.add_argument('--nbpe-wrsrc', type=int, default=8000)

    args = parser.parse_args()

    create_data_links_jsons(args.input_dir, args.output_dir, 
                            use_lid=args.use_lid,
                            use_joint_dict=args.use_joint_dict,
                            nbpe=args.nbpe,
                            nbpe_src=args.nbpe_src,
                            nbpe_wrsrc=args.nbpe_wrsrc)


if __name__ == "__main__":
    main()
