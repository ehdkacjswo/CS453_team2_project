import os
import argparse
import glob
from evaluation.gen_src.gen import generate
from deepga_tool.test_generator import TestGenerator

class Evaluator:
    def __init__(self, run_per_file, manual_src_dir, genned_src_dir):
        self.run_per_file = run_per_file
        self.manual_src_dir = manual_src_dir
        self.genned_src_dir = genned_src_dir
        if not os.path.isdir(self.genned_src_dir):
            os.mkdir(genned_src_dir)

        self.gen_count = 0
        self.test_generator = TestGenerator()  # Default settings.

        self.dnn_approx_result = {}
        self.dnn_result = {}
        self.vanila_result = {}

    def _run_test_generator(self, target_file_path):
        self.dnn_approx_result[target_file_path] = self.test_generator.test_file(target_file_path, True, True)
        self.dnn_result[target_file_path] = self.test_generator.test_file(target_file_path, True, False)
        self.vanila_result[target_file_path] = self.test_generator.test_file(target_file_path, False, False)

    @staticmethod
    def _calc_avg(result):
        generation = 0
        coverage = 0
        elapsed_time = 0

        for file_rt in result.values():
            for fun_rt in file_rt:
                generation += fun_rt['execution']
                coverage += fun_rt['coverage']
                elapsed_time += fun_rt['elapsed_time']
        return {
            'execution': generation / len(result),
            'coverage': coverage / len(result),
            'elapsed_time': elapsed_time / len(result),
        }

    def eval_with_gen_src(self):
        genned_src = generate()

        genned_file_path = os.path.join(self.genned_src_dir, '__genned_{}.py'.format(self.gen_count))
        self.gen_count += 1

        with open(genned_file_path, 'w') as genned_file:
            genned_file.write(genned_src)
        self._run_test_generator(genned_file_path)

    def eval_with_manual_src_all(self):
        manual_src_py_files = os.path.join(self.manual_src_dir, '*.py')
        for manual_path in [os.path.join(self.manual_src_dir, os.path.basename(whole_path)) for whole_path in glob.glob(manual_src_py_files)]:
            self._run_test_generator(manual_path)

    def eval_with_manual_src_file(self, file_name):
        manual_path = os.path.join(self.manual_src_dir, file_name)
        self._run_test_generator(manual_path)

    def get_all_results(self):
        return self.dnn_approx_result, self.dnn_result, self.vanila_result

    def get_avg_results(self):
        return self._calc_avg(self.dnn_approx_result), self._calc_avg(self.dnn_result), self._calc_avg(self.vanila_result)
