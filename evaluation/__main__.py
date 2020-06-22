import os
import argparse
import glob
from evaluation.gen_src.gen import generate
from evaluation.evaluator import Evaluator
import deepga_tool

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, help='File name in manual_src to run.', default=None)
    parser.add_argument('-g', '--gen_src_count', type=int, help='Number of automatically generated source to run.', default=0)
    parser.add_argument('-n', '--run_per_file', type=int, help='Number of runs per single test file.', default=1)
    args = parser.parse_args()

    evaluator = Evaluator(args.run_per_file, 'evaluation/manual_src', '__genned_123456789.py')

    if args.file_name is not None:
        evaluator.eval_with_manual_src_file(args.file_name)
    elif args.gen_src_count > 0:
        for _ in range(args.gen_src_count):
            evaluator.eval_with_gen_src()
    else:
        evaluator.eval_with_manual_src_all()
        

if __name__ == "__main__":
    main()