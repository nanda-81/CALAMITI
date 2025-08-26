#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
from modules.model import CALAMITI

def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='CALAMITI: unsupervised MR harmonization')
    parser.add_argument('--dataset-dirs', type=str, nargs='+', required=True)
    parser.add_argument('--data-names', type=str, nargs='+', required=True)
    parser.add_argument('--orientations', type=str, nargs='+', default=None)
    parser.add_argument('--out-dir', type=str, default='../')
    parser.add_argument('--pretrained-model', type=str, default=None)
    parser.add_argument('--no-shuffle-theta', default=True, action='store_false')
    parser.add_argument('--beta-dim', type=int, default=4)
    parser.add_argument('--theta-dim', type=int, default=2)
    parser.add_argument('--initial-temp', type=float, default=1.0)
    parser.add_argument('--anneal-rate', type=float, default=2e-4)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--fine-tune', default=False, action='store_true')
    args = parser.parse_args(args)

    # initialize model
    trainer = CALAMITI(beta_dim = args.beta_dim,
                       theta_dim = args.theta_dim,
                       train_sample = 'st_gumbel_softmax',
                       valid_sample = 'argmax',
                       pretrained_model = args.pretrained_model,
                       initial_temp = args.initial_temp,
                       anneal_rate = args.anneal_rate,
                       gpu = args.gpu,
                       fine_tune = args.fine_tune)

    trainer.load_dataset(dataset_dirs = args.dataset_dirs,
                         data_names = args.data_names,
                         orientations = args.orientations,
                         batch_size = args.batch_size)

    trainer.initialize_training(out_dir = args.out_dir,
                                lr = args.lr)

    trainer.train(epochs=args.epochs, shuffle_theta=args.no_shuffle_theta)

if __name__ == '__main__':
    main()
