import re
import io
import argparse
from tqdm import tqdm
import multiprocessing
from rdkit import Chem


def rm_map_number(smiles):
    t = re.sub(':\d*', '', smiles)
    return t


def canonicalize(smiles):
    try:
        smiles, keep_atommap = smiles
        if not keep_atommap:
            smiles = rm_map_number(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        else:
            return Chem.MolToSmiles(mol)
    except:
        return None


def main(args):
    input_fn = args.fn
    def lines():
        with io.open(input_fn, 'r', encoding='utf8', newline='\n') as srcf:
            for line in srcf.readlines():
                yield line.strip(), args.keep_atommapnum

    results = []
    total = len(io.open(input_fn, 'r', encoding='utf8', newline='\n').readlines())

    pool = multiprocessing.Pool(args.workers)
    for res in tqdm(pool.imap(canonicalize, lines(), chunksize=100000), total=total):
        if res is not None:
            results.append('{}\n'.format(res))

    if args.output_fn is None:
        output_fn = '{}.can'.format(input_fn)
    else:
        output_fn = args.output_fn
    io.open(output_fn, 'w', encoding='utf8', newline='\n').writelines(results)
    print('{}/{}'.format(len(results), total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', type=str)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-fn', type=str, default=None)
    parser.add_argument('--keep-atommapnum', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
