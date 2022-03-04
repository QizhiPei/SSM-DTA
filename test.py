import logging
import os
import sys
import argparse

from fairseq.models.roberta import RobertaModel
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
import numpy as np 


def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    logger = logging.getLogger(__name__)

    roberta = RobertaModel.from_pretrained(
        args.checkpoint.split('/')[0],
        checkpoint_file=args.checkpoint.split('/')[1],
        data_name_or_path=args.data_bin
    )

    roberta.cuda()
    roberta.eval()
    gold, pred = [], []

    i = 0
    with open(f'{args.test_data}/test.can.re') as mol_in:
        with open(f'{args.test_data}/test.pro.addspace') as pro_in:
            with open(f'{args.test_data}/test.label') as label_in:
                with open(args.output_fn, 'w') as out_f:
                    out_f.write("prediction\ttarget\n")
                    for target in label_in:
                        target = target.strip()
                        sent1 = mol_in.readline().strip()
                        sent2 = pro_in.readline().strip()
                        tokens_0, tokens_1 = roberta.myencode_separate(sent1, sent2)
                        if len(tokens_0) > 512:
                            tokens_0 = tokens_0[:512]
                        if len(tokens_1) > 1024:
                            tokens_1 = tokens_1[:1024]
                        # Extract the last layer's features
                        # Use new classification head to predict
                        predictions = roberta.myextract_features_separate(tokens_0, tokens_1)
                        out_f.write(str(predictions.item()) + '\t' + target + '\n')
                        gold.append(float(target))
                        pred.append(predictions.item())
                        i += 1

    print('Test size: ', i)      
    print('MSE: ', mean_squared_error(gold, pred))
    print('RMSE: ', np.sqrt(mean_squared_error(gold, pred))) 
    print('Pearson: ', pearsonr(gold, pred))
    print('C-index: ', concordance_index(gold, pred))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data-bin', type=str, default=None)
    parser.add_argument('--test-data', type=str, default=None)
    parser.add_argument('--output-fn', type=str, default=None)
    args = parser.parse_args()
    main(args)