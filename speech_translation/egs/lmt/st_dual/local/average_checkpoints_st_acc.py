#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np


def main():
    if args.log is not None:
        with open(args.log) as f:
            logs = json.load(f)
        val_scores = []
        for log in logs:
            if "validation/main/acc" in log.keys():
                val_scores += [[log["epoch"], log["validation/main/acc"]]]
        if len(val_scores) == 0:
            raise ValueError("`validation/main/acc` is not found in log.")
        val_scores = np.array(val_scores)

        if args.num == 1:
            idx_max = np.argmax(val_scores, axis=0)[-1]
            max_val_score = val_scores[idx_max]
            print(f"best val score: {max_val_score[1]}")
            print(f"best val score reached at ep: {int(max_val_score[0])}")
            last = [os.path.join(os.path.dirname(args.snapshots[0]), f"snapshot.ep.{int(max_val_score[0])}")]
        elif args.num > 1:
            sort_idx = np.argsort(val_scores[:, -1])
            sorted_val_scores = val_scores[sort_idx][::-1]
            print("best val scores = " + str(sorted_val_scores[:args.num, 1]))
            print("selected epochs = " + str(sorted_val_scores[:args.num, 0].astype(np.int64)))
            last = [os.path.dirname(args.snapshots[0]) + "/snapshot.ep.%d" % (
                int(epoch)) for epoch in sorted_val_scores[:args.num, 0]]
        else:
            raise NotImplementedError
    else:
        last = sorted(args.snapshots, key=os.path.getmtime)
        last = last[-args.num:]
    avg = None

    existing_last = []
    for path in last:
        if os.path.exists(path):
            existing_last.append(path)
    last = existing_last
    args.num = len(last)
    print(f"averaging over {args.num} checkpoints: {[s.split('/')[-1] for s in last]}")

    if args.backend == 'pytorch':
        import torch

        if len(last) == 1:
            print(f"Get the model with best val BLEU")
            avg = torch.load(last[0], map_location=torch.device("cpu"))["model"]

        elif len(last) > 1:
            # sum
            for path in last:
                states = torch.load(path, map_location=torch.device("cpu"))["model"]
                if avg is None:
                    avg = states
                else:
                    for k in avg.keys():
                        avg[k] += states[k]

            # average
            for k in avg.keys():
                if avg[k] is not None:
                    avg[k] /= args.num
        else:
            raise NotImplementedError
        
        # Save trans model
        print(f"Save trans_model to {args.out}")
        torch.save(avg, args.out)

    elif args.backend == 'chainer':
        # sum
        for path in last:
            states = np.load(path)
            if avg is None:
                keys = [x.split('main/')[1] for x in states if 'model' in x]
                avg = dict()
                for k in keys:
                    avg[k] = states['updater/model:main/{}'.format(k)]
            else:
                for k in keys:
                    avg[k] += states['updater/model:main/{}'.format(k)]
        # average
        for k in keys:
            if avg[k] is not None:
                avg[k] /= args.num
        np.savez_compressed(args.out, **avg)
        os.rename('{}.npz'.format(args.out), args.out)  # numpy save with .npz extension
    else:
        raise ValueError('Incorrect type of backend')


def get_parser():
    parser = argparse.ArgumentParser(description='average models from snapshot')
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--backend", default='chainer', type=str)
    parser.add_argument("--log", default=None, type=str, nargs="?")
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main()
