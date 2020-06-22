import os
import argparse
import glob
from evaluation.gen_src.gen import generate
import deepga_tool

PYTHON_CMD = 'python'
RUN_PER_FILE = 1  # Mutable by args

MANUAL_SRC_DIR = 'evaluation/manual_src'
GENNED_FNAME = '__genned_123456789.py'

def run_test_generator(py_file_path):
    cmd = '{} -m deepga_tool {}'.format(PYTHON_CMD, py_file_path)
    for _ in range(RUN_PER_FILE):
        os.system(cmd)

def eval_with_gen_src():
    genned_src = generate()
    print("=" * 20)
    print(genned_src)
    print("=" * 20)
    with open(GENNED_FNAME, 'w') as genned_file:
        genned_file.write(genned_src)
    run_test_generator(GENNED_FNAME)
    os.remove(GENNED_FNAME)

def eval_with_manual_src_all():
    manual_src_py_files = os.path.join(MANUAL_SRC_DIR, '*.py')
    for manual_path in [os.path.join(MANUAL_SRC_DIR, os.path.basename(whole_path)) for whole_path in glob.glob(manual_src_py_files)]:
        run_test_generator(manual_path)

def eval_with_manual_src_file(file_name):
    manual_path = os.path.join(MANUAL_SRC_DIR, file_name)
    run_test_generator(manual_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, help='File name in manual_src to run.', default=None)
    parser.add_argument('-g', '--gen_src_count', type=int, help='Number of automatically generated source to run.', default=0)
    parser.add_argument('-n', '--run_per_file', type=int, help='Number of runs per single test file.', default=1)
    args = parser.parse_args()

    RUN_PER_FILE = args.run_per_file

    if args.file_name is not None:
        eval_with_manual_src_file(args.file_name)
    elif args.gen_src_count > 0:
        for _ in range(args.gen_src_count):
            eval_with_gen_src()
    else:
        eval_with_manual_src_all()
        

if __name__ == "__main__":
    main()