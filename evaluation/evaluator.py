import os
import argparse
import glob
from evaluation.gen_src.gen import generate
from deepga_tool.test_generator import TestGenerator

class Evaluator:
    class Result:
        def __init__(self):
            self.filefunc_result = {}

        def store_result(self, target_file_path, seed, test_generator_result):
            for func_name, func_result in test_generator_result.items():
                filefunc = '{}_{}'.format(target_file_path, func_name)
                if filefunc not in self.filefunc_result:
                    self.filefunc_result[filefunc] = {}
                self.filefunc_result[filefunc][seed] = func_result

        def calc_seed_avg(self):
            for filefunc, result in self.filefunc_result.items():
                seed_count = len(result)
                generation_avg = 0
                coverage_avg = 0
                elasped_time_avg = 0

                for seed, func_result in result.items():
                    generation_avg += func_result.generation
                    coverage_avg += func_result.coverage
                    elasped_time_avg += func_result.elasped_time

                generation_avg /= seed_count
                coverage_avg /= seed_count
                elasped_time_avg /= seed_count

                self.filefunc_result[filefunc]['avg'] = TestGenerator.FuncResult(
                        generation_avg, coverage_avg, elasped_time_avg)

        # This is valid since #seeds is always uniform for single Evaluator object.
        def calc_all_avg(self):
            generation_avg = 0
            coverage_avg = 0
            elasped_time_avg = 0
            all_count = len(self.filefunc_result)

            for filefunc, result in self.filefunc_result.items():
                generation_avg += result['avg'].generation
                coverage_avg += result['avg'].coverage
                elasped_time_avg += result['avg'].elasped_time

            generation_avg /= all_count
            coverage_avg /= all_count
            elasped_time_avg /= all_count

            return TestGenerator.FuncResult(generation_avg, coverage_avg, elasped_time_avg)

    def __init__(self, seeds, manual_src_dir, genned_src_dir):
        self.seeds = seeds
        self.manual_src_dir = manual_src_dir
        self.genned_src_dir = genned_src_dir
        if not os.path.isdir(self.genned_src_dir):
            os.mkdir(genned_src_dir)

        self.gen_count = 0
        self.test_generator = TestGenerator()  # Default settings.

        self.dnn_approx_result = self.Result()
        self.dnn_result = self.Result()
        self.vanila_result = self.Result()

    def _run_test_generator(self, target_file_path):
        for seed in self.seeds:
            print(target_file_path, seed)
            self.test_generator.set_seed(seed)
            self.dnn_approx_result.store_result(target_file_path, seed,
                    self.test_generator.test_file(target_file_path, True, True))
            self.dnn_result.store_result(target_file_path, seed,
                    self.test_generator.test_file(target_file_path, True, False))
            self.vanila_result.store_result(target_file_path, seed,
                    self.test_generator.test_file(target_file_path, False, False))

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
        self.dnn_approx_result.calc_seed_avg()
        self.dnn_result.calc_seed_avg()
        self.vanila_result.calc_seed_avg()
        return self.dnn_approx_result, self.dnn_result, self.vanila_result

    def get_all_avg_results(self):
        return (self.dnn_approx_result.calc_all_avg(),
                self.dnn_result.calc_all_avg(),
                self.vanila_result.calc_all_avg())
