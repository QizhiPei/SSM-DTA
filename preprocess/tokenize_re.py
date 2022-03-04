import re
import io
import argparse
from tqdm import tqdm
import multiprocessing


def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    try:
        assert re.sub('\s+', '', smi) == ''.join(tokens)
    except:
        return ''

    return ' '.join(tokens)


def main(args):
    input_fn = args.fn
    def lines():
        with io.open(input_fn, 'r', encoding='utf8', newline='\n') as srcf:
            for line in srcf:
                yield line.strip()

    results = []
    total = len(io.open(input_fn, 'r', encoding='utf8', newline='\n').readlines())

    pool = multiprocessing.Pool(args.workers)
    for res in tqdm(pool.imap(smi_tokenizer, lines(), chunksize=100000), total=total):
        if res:
            results.append('{}\n'.format(res))

    if args.output_fn is None:
        output_fn = '{}.bpe'.format(input_fn)
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
