import logging
import os
import sys
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
import numpy as np

from fairseq.models.roberta import RobertaModel
from torch.nn.utils.rnn import pad_sequence

def main(args):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    roberta = RobertaModel.from_pretrained(
        os.path.split(args.checkpoint)[0],
        checkpoint_file=os.path.split(args.checkpoint)[1],
        data_name_or_path=args.data_bin,
    )

    roberta.cuda()
    roberta.eval()
    bsz = args.batch_size


    total = len(open(args.input_mol_fn, 'r').readlines())
    pbar = tqdm(total=total, desc='Predicting')
    
    with open(f'{args.input_mol_fn}', 'r') as mol_in, open(f'{args.input_pro_fn}', 'r') as pro_in, open(args.output_fn, 'w') as out_f:
        batch_mol_buf = []
        batch_pro_buf = []
        for i, mol_pro in enumerate(zip(mol_in, pro_in)):
            mol, pro = mol_pro
            if ((i+1) % bsz == 0) or ((i+1) == total):
                tmp_mol, tmp_pro = roberta.myencode_separate(mol.strip(), pro.strip())
                batch_mol_buf.append(tmp_mol[:512])
                batch_pro_buf.append(tmp_pro[:1024])
                tokens_0 = pad_sequence(batch_mol_buf, batch_first=True, padding_value=1)
                tokens_1 = pad_sequence(batch_pro_buf, batch_first=True, padding_value=1)
                predictions = roberta.myextract_features_separate(tokens_0, tokens_1)
                for result in predictions:
                    out_f.write(f'{str(result.item())}\n')
                batch_mol_buf.clear()
                batch_pro_buf.clear()
                pbar.update(1)
            else:
                tmp_mol, tmp_pro = roberta.myencode_separate(mol.strip(), pro.strip())
                batch_mol_buf.append(tmp_mol[:512])
                batch_pro_buf.append(tmp_pro[:1024])
                pbar.update(1)
                continue
    
    pbar.close()
    if args.mode == 'eval':
        assert args.input_label_fn is not None
        pred = [float(line.strip()) for line in open(args.output_fn, 'r').readlines()]
        gold = [float(line.strip()) for line in open(args.input_label_fn, 'r').readlines()]
        print('MSE:', mean_squared_error(gold, pred))
        print('RMSE:', np.sqrt(mean_squared_error(gold, pred))) 
        print('Pearson:', pearsonr(gold, pred))
        print('C-index:', concordance_index(gold, pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data-bin', type=str, default=None)
    parser.add_argument('--input-mol-fn', type=str, default=None)
    parser.add_argument('--input-pro-fn', type=str, default=None)
    parser.add_argument('--input-label-fn', type=str, default=None)
    parser.add_argument('--output-fn', type=str, default='/tmp/tmp.txt')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'predict'])
    args = parser.parse_args()
    main(args)