import argparse
from deepga_tool.test_generator import TestGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('py_file', type=str, help='Input python function file')
    parser.add_argument(
        '--p', type=int, help='Number of population')
    parser.add_argument('--gen', type=int,
                        help='Number of generation')
    parser.add_argument(
        '--pm', type=int, help='Probability of mutation in percentage')

    # Arguments for our deep learning framework
    parser.add_argument('--niter', type=int,
                        help="Number of iteration to be optimized")
    parser.add_argument('--lr', type=float,
                        help="Learning for optimizer")
    parser.add_argument('--no_cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--step_size', type=float,
                        help="Step size for guided gradient descent")
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument(
        '--model-dir', help='model save directory')

    args = parser.parse_args()
    test_generator = TestGenerator(
        p=args.p, gen=args.gen, pm_percent=args.pm,
        niter=args.niter, lr=args.lr, no_cuda=args.no_cuda,
        step_size=args.step_size, seed=args.seed, model_dir=args.model_dir
    )

    # Experiment with dnn or not
    print('testing with dnn and approx')
    print(test_generator.test_file(args.py_file, True, True), '\n')
    
    print('testing with dnn and no approx')
    print(test_generator.test_file(args.py_file, True, False), '\n')

    print('testing without dnn')
    print(test_generator.test_file(args.py_file, False, False))


if __name__ == "__main__":
    main()
