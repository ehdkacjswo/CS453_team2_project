import os
import argparse
import glob
import pprint
from evaluation.gen_src.gen import generate
from evaluation.evaluator import Evaluator

print_bar_single = '-' * 60
print_bar_double = '=' * 60

def print_result(result):
    print(print_bar_single)
    for file_name, seed_fr in result.filefunc_result.items():
        print(file_name)
        print(print_bar_single)
        for seed, fr in seed_fr.items():
            print(seed)
            pprint.pprint(fr, width=10)
        print(print_bar_single)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, help='File name in manual_src to run.', default=None)
    parser.add_argument('-g', '--gen_src_count', type=int, help='Number of automatically generated source to run.', default=0)
    parser.add_argument('-n', '--num_seeds', type=int, help='Number of seeds that will run per source file.', default=2)
    args = parser.parse_args()

    seeds = [10 * s for s in range(args.num_seeds)]
    evaluator = Evaluator(seeds, 'evaluation/manual_src', 'evaluation/__genned')

    if args.file_name is not None:
        evaluator.eval_with_manual_src_file(args.file_name)
    elif args.gen_src_count > 0:
        for _ in range(args.gen_src_count):
            evaluator.eval_with_gen_src()
    else:
        evaluator.eval_with_manual_src_all()

    dnn_approx_result, dnn_result, vanila_result = evaluator.get_all_results()

    print('All results.')
    print('DNN + Approx')
    print_result(dnn_approx_result)
    print('DNN')
    print_result(dnn_result)
    print('Vanila')
    print_result(vanila_result)

    dnn_approx_avg, dnn_avg, vanila_avg = evaluator.get_all_avg_results()

    print(print_bar_double)
    print('Overall avg results.')
    print('DNN + Approx')
    print(dnn_approx_avg)
    print('DNN')
    print(dnn_avg)
    print('Vanila')
    print(vanila_avg)
    
        

if __name__ == "__main__":
    main()
