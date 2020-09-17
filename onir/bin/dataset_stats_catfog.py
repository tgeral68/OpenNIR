from tqdm import tqdm
import pandas as pd
import onir


def main():
    logger = onir.log.easy()

    context = onir.injector.load({
        'vocab': onir.vocab,
        'dataset': onir.datasets
    })

    vocab = context['vocab']
    dataset = context['dataset']
    logger.debug(f'vocab: {vocab.config}')
    logger.debug(f'dataset: {dataset.config}')

    #dataset.init()

    num_docs = onir.datasets.num_docs(dataset)
    num_queries = onir.datasets.num_queries(dataset)
    num_qrels = onir.datasets.qrels(dataset)
    print("Table 1:")
    print(f"{num_docs}\t{num_queries}\t{num_qrels}")
    
    

if __name__ == '__main__':
    main()
