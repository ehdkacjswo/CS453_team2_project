import os
import argparse
import glob
from gen_src.srcgen import src_gen

PYTHON_CMD = 'python'
RUN_PER_FILE = 1

def run_test_generator(py_file_path):
    main_path = os.path.join(os.getcwd(), 'main.py')
    cmd = '{} {} {}'.format(PYTHON_CMD, main_path, py_file_path)
    for _ in range(RUN_PER_FILE):
        os.system(cmd)

def eval_with_gen_src():
    gen_src_dir = 'gen_src'
    genned_path = os.path.join(gen_src_dir, 'genned.py')
    genned_src = src_gen()
    print("=" * 20)
    print(genned_src)
    print("=" * 20)
    with open(genned_path, 'w') as genned_file:
        genned_file.write(genned_src)
    run_test_generator(genned_path)
    os.remove(genned_path)

def eval_with_manual_src_all():
    manual_src_py_files = os.path.join('manual_src', '*.py')
    for manual_path in [os.path.join('manual_src', os.path.basename(whole_path)) for whole_path in glob.glob(manual_src_py_files)]:
        run_test_generator(manual_path)

def eval_with_manual_src_file(file_name):
    manual_path = os.path.join('manual_src', file_name)
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
    else:
        eval_with_manual_src_all()
        for _ in range(args.gen_src_count):
            eval_with_gen_src()

if __name__ == "__main__":
    main()