import re
import io
import argparse
from tqdm import tqdm
import multiprocessing

def addspace(pro):
    return ' '.join(list(pro))

def main(args):
    input_fn = args.fn
    def lines():
        with io.open(input_fn, 'r', encoding='utf8', newline='\n') as srcf:
            for line in srcf:
                yield line.strip()

    results = []
    total = len(io.open(input_fn, 'r', encoding='utf8', newline='\n').readlines())

    pool = multiprocessing.Pool(args.workers)
    for res in tqdm(pool.imap(addspace, lines(), chunksize=100000), total=total):
        if res:
            results.append('{}\n'.format(res))

    if args.output_fn is None:
        output_fn = '{}.pro.addspace'.format(input_fn)
    else:
        output_fn = args.output_fn
    io.open(output_fn, 'w', encoding='utf8', newline='\n').writelines(results)
    print('{}/{}'.format(len(results), total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', type=str)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-fn', type=str, default=None)
    args = parser.parse_args()
    main(args)
