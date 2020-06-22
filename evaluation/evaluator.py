import os
import argparse
import glob
from evaluation.gen_src.gen import generate
from deepga_tool.test_generator import TestGenerator

class Evaluator:
    def __init__(self, run_per_file, manual_src_dir, genned_file_name):
        self.run_per_file = run_per_file
        self.manual_src_dir = manual_src_dir
        self.genned_file_name = genned_file_name

    def _run_test_generator(self, py_file_path):
        cmd = 'python -m deepga_tool {}'.format(py_file_path)
        for _ in range(self.run_per_file):
            os.system(cmd)

    def eval_with_gen_src(self):
        genned_src = generate()
        print("=" * 20)
        print(genned_src)
        print("=" * 20)
        with open(self.genned_file_name, 'w') as genned_file:
            genned_file.write(genned_src)
        self._run_test_generator(self.genned_file_name)
        os.remove(self.genned_file_name)

    def eval_with_manual_src_all(self):
        manual_src_py_files = os.path.join(self.manual_src_dir, '*.py')
        for manual_path in [os.path.join(self.manual_src_dir, os.path.basename(whole_path)) for whole_path in glob.glob(manual_src_py_files)]:
            self._run_test_generator(manual_path)

    def eval_with_manual_src_file(self, file_name):
        manual_path = os.path.join(self.manual_src_dir, file_name)
        self._run_test_generator(manual_path)