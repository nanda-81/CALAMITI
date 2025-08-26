#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
from modules.fusion import FusionNetwork

def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='FusionNetwork')
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--data-name', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='../')
    parser.add_argument('--pretrained-model', type=str, default=None)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args(args)

    # initialize model
    trainer = FusionNetwork(pretrained_model = args.pretrained_model,
                            gpu = args.gpu)

    trainer.load_dataset(dataset_dir = args.dataset_dir,
                         data_name = args.data_name,
                         batch_size = args.batch_size)

    trainer.initialize_training(out_dir = args.out_dir,
                                lr = args.lr)

    trainer.train(epochs = args.epochs)

if __name__ == '__main__':
    main()
