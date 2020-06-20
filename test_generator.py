import ast
import astor
import sys
import os
import copy
import math
import time
import importlib
import random as rand
from ast_helper import find_num, find_if, name_len, branch

from ga.ga_helper import in_test, add_test
from ga.selection import save_sel
from ga.crossover import doam_cross
from ga.mutation import doam_mut

from dnn.model import MLP
from dnn.nn_train import guided_mutation, train, forward
import torch
import torch.optim as optim


class TestGenerator:
    def __init__(self,
                 p, gen, pm_percent,
                 niter, lr, no_cuda, step_size, seed, model_dir):
        self.p = p
        self.save_p = int(math.floor(p * 0.1))
        self.gen = gen

        self.func_file = 'branch_dist_print'
        self.br_file = 'br_dist'

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        self.niter = niter
        self.lr = lr
        self.no_cuda = no_cuda
        self.seed = seed
        self.model_dir = model_dir

        self.args = self.Args(pm_percent / 100.0, 1, 1, device, step_size, niter)

        rand.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Class contains arguments for other functions
    class Args:
        def __init__(self, pm, alpha, beta, device, step_size, niter):
            self.pm = pm
            self.alpha = alpha
            self.beta = beta
            self.device = device
            self.step_size = step_size
            self.niter = niter
            self.use_dnn = True
            self.k = 1

    # Generarte input from function ast
    def gen_input(self, func):
        rt = []
        arg_num = len(func.args.args)
        special = list(set(find_num(func.body) + [0, 1, -1]))

        while len(rt) < self.p:
            inp = []

            for j in range(arg_num):
                if rand.random() <= 0.2:
                    inp.append(rand.choice(special))
                else:
                    inp.append(rand.randint(-10000, 10000))

            add_test(rt, inp)

        return special, rt

    # Analyze the fitness output
    def get_result(self):
        f = open(self.br_file, "r")
        br_data = f.readlines()
        f.close()

        # Maps branch id to branch distance
        # Positive id : true branch, Negative id : false branch
        # Passed branches : negative distance
        br_dict = {}

        for data in br_data:
            br_id, br_dist = [float(x) for x in data.split(" ")]

            # Passed branch has non-poisitve branch distance
            # Otherwise, not passed
            new_data = [(br_id, br_dist), (-br_id, self.args.k - br_dist)]

            for tup in new_data:
                item = br_dict.get(tup[0])

                if item is None or item > tup[1]:
                    br_dict[tup[0]] = tup[1]

        br_fit = {}

        # For each leaves, find fitness
        for leaf_ind, lvl_dict in self.leaf_index.items():
            for ind, lvl in sorted(lvl_dict.items(), key=lambda tup: tup[1]):
                dist = br_dict.get(ind)

                if dist is not None:
                    if dist > 0:
                        br_fit[leaf_ind] = lvl + float(dist) / (dist + 1)
                        break

                    else:
                        if lvl == 0:
                            br_fit[leaf_ind] = -1

                        # Parent passed but did not reach its child
                        else:
                            br_fit[leaf_ind] = lvl

                        break

            # None itself or ancestors visited
            if not leaf_ind in br_fit:
                br_fit[leaf_ind] = len(lvl_dict) + 1

        return br_fit

    # Learn new test and train dnn
    # Returns true if
    def test_and_learn(self, new_test):
        new_output = []
        sol_found = False

        # Initialize input and fitness for dnn
        if self.args.use_dnn:
            dnn_inp = {}
            dnn_fit = {}

            for leaf_ind in self.leaf_index:
                dnn_inp[leaf_ind] = []
                dnn_fit[leaf_ind] = []

        for inp in new_test:
            # Don't print anything
            with self.HiddenPrint():
                with open(self.br_file, 'w') as br_report:
                    try:
                        self.method(br_report, *inp)
                    except:
                        # Exception detected, do nothing
                        pass

            new_output.append((inp, self.get_result()))

            # Check whether the solution is found
            for leaf_ind in list(self.leaf_index):
                # When the test is found for leaf
                if new_output[-1][1][leaf_ind] < 0:
                    del self.leaf_index[leaf_ind]
                    self.leaf_cover.append(leaf_ind)

                    print(leaf_ind, inp)

                    if not bool(self.leaf_index):
                        sol_found = True
                    break

            if sol_found:
                break

            if self.args.use_dnn:
                for leaf_ind in self.leaf_index:
                    dnn_inp[leaf_ind].append(new_output[-1][0])
                    dnn_fit[leaf_ind].append([new_output[-1][1][leaf_ind]])

        # Number of leaves that dnn converges (loss < 0.1)
        conv_leaf = 0

        if self.args.use_dnn and (not sol_found):
            for leaf_ind in self.leaf_index:
                epoch, loss = train(dnn_inp[leaf_ind], dnn_fit[leaf_ind], leaf_ind, self.args)

                if loss < 0.2:
                    conv_leaf += 1

        return new_output, conv_leaf, sol_found

    # Approximate fitness using model
    def approx(self, new_test):
        new_output = []

        for test in new_test:
            fit_dict = {}

            for leaf_ind in self.leaf_index:
                fit_dict[leaf_ind] = forward(test, leaf_ind, self.args)

            new_output.append((test, fit_dict))

        return new_output

    # Helps to suppress print
    class HiddenPrint:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    # Main part tests, evolves test cases
    def test_func(self, root_copy, body_ind, test_file_name, func_file_name):
        func = root_copy.body[body_ind]

        if not isinstance(func, ast.FunctionDef):
            return

        func_name = func.name
        func.name = self.new_func_name

        # Needs no argument
        if not func.args.args:
            return

        branch.br_list = [None]
        find_if(func.body, 0, self.args, True)

        # No branches found
        if len(branch.br_list) == 1:
            return

        # Generate input
        special, new_test = self.gen_input(func)

        # Change function name and Import original function
        func.name = self.new_func_name
        root_copy.body.insert(0, ast.ImportFrom(module=test_file_name[:-3].replace(
            '/', '.'), names=[ast.alias(name=func_name, asname=None)], level=0))
        func.args.args.insert(0, ast.Name(id=self.args.file_name))

        # Write changed code on new file
        code = astor.to_source(root_copy)
        source_file = open(func_file_name, 'w')
        source_file.write(code)
        source_file.close()

        # Get index of leaf branches (ind, app_lvl)
        self.leaf_index = {}

        for cur_br in branch.br_list[1:]:
            # At least one of branches is leaf
            if cur_br.reach and ((not cur_br.true) or (not cur_br.false)):
                app_lvl = 1
                lvl_dict = {}
                next_ind = cur_br.parent

                # Add parents till the root
                while next_ind != 0:
                    lvl_dict[next_ind] = app_lvl
                    app_lvl += 1
                    next_ind = branch.br_list[abs(next_ind)].parent

                # Branch without child branch is leaf
                if not cur_br.true:
                    pos_dict = copy.deepcopy(lvl_dict)
                    pos_dict[cur_br.ind] = 0
                    self.leaf_index[cur_br.ind] = pos_dict
                if not cur_br.false:
                    neg_dict = copy.deepcopy(lvl_dict)
                    neg_dict[-cur_br.ind] = 0
                    self.leaf_index[-cur_br.ind] = neg_dict

        # Used for final printing
        leaf_index_copy = copy.deepcopy(self.leaf_index)

        # DNN init
        if self.args.use_dnn:
            self.args.input_dim = len(func.args.args) - 1
            self.args.dnn = {}

        # Branch fitness output with(test, output)
        output = {}
        last_best = {}

        for leaf_ind in self.leaf_index:
            output[leaf_ind] = []

            # (Best fitness, number of generation passed) for each leaves
            last_best[leaf_ind] = [-1, 0]

            if self.args.use_dnn:
                model = MLP(self.args.input_dim).to(self.args.device)
                opt = optim.AdamW(model.parameters(), lr=0.01)
                self.args.dnn[leaf_ind] = (model, opt)

        # Import revised code
        module = importlib.import_module(func_file_name[:-3].replace('/', '.'))
        self.method = getattr(module, self.new_func_name)

        # Index of coverd leaves
        self.leaf_cover = []
        sol_found = False

        # Return value (gen, # of br, # of br passed)
        rt = []

        print(self.leaf_index.keys())

        # Population initialization
        new_output, conv_leaf, sol_found = self.test_and_learn(new_test)

        if sol_found:
            return [0, 2 * (len(branch.br_list) - 1), 2 * (len(branch.br_list) - 1)]

        for leaf_ind in self.leaf_index:
            save_sel(output, new_output, leaf_ind, self.p, self.save_p)
            output[leaf_ind].sort(key=lambda data: data[1][leaf_ind])

        use_approx = True if conv_leaf == len(self.leaf_index) else False
        approx_gen = 0

        for i in range(self.gen):
            use_approx = False
            print(i)

            print(use_approx)

            #use_approx = True
            new_test = []
            last_test_num = 0
            pop_per_leaf = (self.p + len(self.leaf_index) - 1) / len(self.leaf_index)

            # Create new test cases
            for leaf_ind in self.leaf_index:
                # Generate test case until p tests
                while len(new_test) - last_test_num < pop_per_leaf:
                    children = doam_cross(output[leaf_ind], leaf_ind, special, self.args)

                    # Check whether the child is already in test case
                    for child in children:
                        child_found = False

                        for sub_leaf_ind in self.leaf_index:
                            if in_test([out[0] for out in output[sub_leaf_ind]], child):
                                child_found = True
                                break

                        if not child_found:
                            add_test(new_test, child)

                last_test_num = len(new_test)

            '''if use_approx:
                approx_gen += 1
                deep_leaf = False
                fit_leaf = 0

                new_output = self.approx(new_test)
                
                for leaf_ind in self.leaf_index:
                    org_best = output[leaf_ind][0][1][leaf_ind]
                    last_best = org_best if approx_gen == 1 else output[leaf_ind][self.save_p][1][leaf_ind]

                    # Save first save_p tests
                    save_sel(output, new_output, leaf_ind, self.p, self.save_p)
                    cur_best = output[leaf_ind][self.save_p][1][leaf_ind]

                    # Found deeper case
                    if cur_best <= math.floor(org_best):
                        deep_leaf = True

                    # Fitness converges
                    #print(leaf_ind, cur_best, last_best)
                    if cur_best >= last_best or abs(cur_best - last_best) < 1e-1:
                        fit_leaf += 1

                print(deep_leaf, fit_leaf, len(self.leaf_index))

                # Found deeper case or fitness converges for every leaves
                # Or last generation
                if deep_leaf or fit_leaf == len(self.leaf_index) or i == self.gen - 1:
                    use_approx = False
                    new_test = []
                    last_test_num = 0

                    for leaf_ind in self.leaf_index:
                        for child in [out[0] for out in output[leaf_ind][self.save_p:]]:
                            child_found = False

                            for sub_leaf_ind in self.leaf_index:
                                if in_test([out[0] for out in output[sub_leaf_ind][:self.save_p]], child):
                                    child_found = True
                                    break

                            if not child_found:
                                add_test(new_test, child)

                            if len(new_test) - last_test_num >= pop_per_leaf:
                                break

                        last_test_num = len(new_test)
                        output[leaf_ind] = output[leaf_ind][:self.save_p]'''

            if not use_approx:
                approx_gen = 0

                new_output, conv_leaf, sol_found = self.test_and_learn(new_test)

                if sol_found or i == self.gen - 1:
                    rt.append(i + 1)
                    break

                # Number of leaves that fitness converges
                fit_leaf = 0

                # Select population
                for leaf_ind in self.leaf_index:
                    save_sel(output, new_output, leaf_ind, self.p, self.save_p)
                    last_best = output[leaf_ind][0][1][leaf_ind]

                    output[leaf_ind].sort(key=lambda data: data[1][leaf_ind])
                    cur_best = output[leaf_ind][0][1][leaf_ind]

                    '''# Fitness doesn't change much
                    if abs(last_best - cur_best) < 1e-1:
                        fit_leaf += 1'''

                '''if conv_leaf == len(self.leaf_index) or fit_leaf == len(self.leaf_index):
                    use_approx = True'''

        # Set of braches coverd
        br_pass = set()

        for leaf_ind, lvl_dict in leaf_index_copy.items():
            # Solution found for leaf
            if leaf_ind in self.leaf_cover:
                for parent in lvl_dict:
                    br_pass.add(parent)

            else:
                test = output[leaf_ind][0][0]
                best_lvl = int(math.ceil(output[leaf_ind][0][1][leaf_ind]))

                for parent, lvl in lvl_dict.items():
                    # Add when only it's visited
                    if lvl >= best_lvl:
                        br_pass.add(parent)

        rt.append(2 * (len(branch.br_list) - 1))
        rt.append(len(br_pass))

        # Delete trashes
        del module
        del self.method

        if os.path.exists(func_file_name):
            os.remove(func_file_name)
        if os.path.exists(self.br_file):
            os.remove(self.br_file)

        return rt

    def test_file(self, test_file_name, use_dnn):
        root = astor.code_to_ast.parse_file(test_file_name)

        # Generate unused variable name to avoid shadowing
        var_len = name_len(root.body) + 1

        self.args.file_name = 'f' * var_len
        self.args.temp_name = 't' * var_len
        self.args.lambda_arg = 'x' * var_len
        self.new_func_name = 'f' * (var_len + 1)

        self.args.use_dnn = use_dnn

        rt = []

        # Test for each functions
        for ind in range(len(root.body)):
            if isinstance(root.body[ind], ast.FunctionDef):
                time_start = time.time()
                rt.append(self.test_func(copy.deepcopy(root), ind,
                                         test_file_name, self.func_file + str(ind) + '.py'))
                rt[-1].append(time.time() - time_start)

        # [generation, total branch, passed branch, time]
        return rt
