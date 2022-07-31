from utils.experiment import Experiment
import json

datasets = ['point_mnist', 'modelnet40', 'oxford']
ks = [4, 8, 16]
ref_funcs = ['rand']
hash_code_length = [2 ** i for i in range(4, 11)]

for dataset in datasets:
    if dataset == 'point_mnist':
        ref_size = 150
        n_slices = [2 ** i for i in range(10)]

    elif dataset == 'modelnet40':
        ref_size = 512
        n_slices = [2 ** i for i in range(8)]

    elif dataset == 'oxford':
        ref_size = 384
        n_slices = [2 ** i for i in range(8)]

    print('slices:', n_slices)
    print('hash code lengths:', hash_code_length)

    val_results = []
    for s in n_slices:
        for l in hash_code_length:
            for k in ks:
                for ref in ref_funcs:
                    exp = Experiment(dataset, 'swe', 'faiss-lsh', mode='validation', random_state=0,
                                     k=k, ref_size=336, code_length=l, num_slices=s, ref_func=ref)
                    exp.test()
                    report = exp.get_exp_report()
                    val_results.append({'dataset': dataset,
                                        'k': k,
                                        'reference': ref,
                                        'num_slices': s,
                                        'code_length': l,
                                        'precision_k': report['precision_k'],
                                        'acc': report['acc']})
                    print(f'num_slices: {s}')
                    print(val_results[-1])

    with open(f'json/{dataset}_slices.json', 'w') as f:
        json.dump(val_results, f)