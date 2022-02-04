import os
import argparse

import numpy as np
from loguru import logger

import data_loader as dl
import bias_measure
import utils


def calc_bias_scores(target_words, bps, embed_model):
    bias_scores = [bias_measure.batch_db_wa_bias(target_words, bps, embed_model).tolist(),
                   bias_measure.batch_ripa_bias(target_words, bps, embed_model).tolist(),
                   bias_measure.batch_nbm_bias(target_words, bps,
                                               embed_model).tolist()]  # n_bias_metrics, n_target_words, n_pairs
    return bias_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_folder", type=str)
    parser.add_argument("--vocab_path", type=str)
    parser.add_argument('--embed_type', type=str, choices=['sgns', 'glove'])
    parser.add_argument("--base_pair_type", type=str, choices=['gender', 'sexual_orientation'])
    parser.add_argument("--bias_score_path", type=str)
    args = parser.parse_args()

    bias_measures = ['db_wa', 'ripa', 'nbm']

    # load word lists
    vocab_path = args.vocab_path
    embed_folder = args.embed_folder
    bias_score_path = args.bias_score_path
    embed_type = args.embed_type
    os.makedirs(os.path.dirname(bias_score_path), exist_ok=True)

    vocab = dl.load_vocab(vocab_path)
    vocab.discard('<unk>')
    vocab.discard('<raw_unk>')

    bp_type = args.base_pair_type

    if bp_type == 'gender':
        bps = [tuple(gbp) for gbp in dl.load_gbp() if gbp[0] in vocab and gbp[1] in vocab]
    else:
        bps = [tuple(sbp) for sbp in dl.load_sbp() if sbp[0] in vocab and sbp[1] in vocab]

    bias_scores = []
    model_names = []
    if embed_type == 'sgns':
        embed_paths = [os.path.join(embed_folder, embed_path)
                       for embed_path in os.listdir(embed_folder) if embed_path.endswith('.model')]
        for embed_idx, embed_path in enumerate(embed_paths):
            logger.info(f'Processing embedding model #{embed_idx+1} out of {len(embed_paths)}')
            embed_model = dl.load_gensim_sgns(embed_path)
            bias_scores.append(calc_bias_scores(list(vocab), bps, embed_model))
            model_names.append(os.path.basename(embed_path))

    else:
        embed_paths = [os.path.join(embed_folder, embed_path)
                       for embed_path in os.listdir(embed_folder) if embed_path.endswith('.txt')]

        for embed_idx, embed_path in enumerate(embed_paths):
            logger.info(f'Processing embedding model #{embed_idx+1} out of {len(embed_paths)}')
            embed_model = dl.load_glove(embed_path, vocab_path=vocab_path)
            bias_scores.append(calc_bias_scores(list(vocab), bps, embed_model))
            model_names.append(os.path.basename(embed_path))
    # save
    bias_scores = np.moveaxis(np.array(bias_scores), 0, 3)  # measure, target, gbp, embed
    utils.write_pickle([[bias_measures, list(vocab), bps, model_names], bias_scores], bias_score_path)


if __name__ == '__main__':
    main()
