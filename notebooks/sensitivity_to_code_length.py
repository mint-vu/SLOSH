from utils.experiment import Experiment
import json

datasets = ['point_mnist', 'modelnet40', 'oxford']
ks = [4, 8, 16]
ref_funcs = ['rand']
hash_code_lengths = [2 ** i for i in range(4, 11)]

for dataset in datasets:
    if dataset == 'point_mnist':
        ref_size = 150
        n_slice = 8

    elif dataset == 'modelnet40':
        ref_size = 512
        n_slice = 16

    elif dataset == 'oxford':
        ref_size = 384
        n_slice = 128

    print('hash code lengths:', hash_code_lengths)

    val_results = []

    for k in ks:
        for ref in ref_funcs:
            for l in hash_code_lengths:
                exp = Experiment(dataset, 'swe', 'faiss-lsh', mode='validation', random_state=0,
                                 k=k, ref_size=336, code_length=l, num_slices=n_slice, ref_func=ref)
                exp.test()
                report = exp.get_exp_report()
                val_results.append({'dataset': dataset,
                                    'k': k,
                                    'reference': ref,
                                    'num_slices': n_slice,
                                    'code_length': l,
                                    'precision_k': report['precision_k'],
                                    'acc': report['acc']})
                print(val_results[-1])

    with open(f'json/{dataset}_code_length.json', 'w') as f:
        json.dump(val_results, f)
