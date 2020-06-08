import argparse
from input_generator import InputGenerator

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('py_file', type=str, help='Input python function file')
	parser.add_argument('--p', type=int, help='Number of population', default=100)
	parser.add_argument('--gen', type=int, help='Number of generation', default=1000)
	parser.add_argument('--pm', type=int, help='Probability of mutation in percentage', default=20)
	parser.add_argument('--ps', type=int, help='Percentage of population saved <= 50', default = 10)
	parser.add_argument('--alpha', type=int, help='Alpha of gamma distribution', default=1)
	parser.add_argument('--beta', type=int, help='Beta of gamma distribution', default=1)
	parser.add_argument('--func', type=str, help='Name of revised python file', default='branch_dist_print')
	parser.add_argument('--br', type=str, help='Name of branch distance file', default='br_dist')

	# Arguments for our deep learning framework
	parser.add_argument('--niter', type=int, help="Number of iteration to be optimized", default=10000)
	parser.add_argument('--lr', type=float, help="Learning for optimizer", default=1e-2)
	parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
	parser.add_argument('--step_size',type=float, help="Step size for guided gradient descent", default=0.1)
	parser.add_argument('--seed', type=int, help='random seed', default=2)
	parser.add_argument('--model-dir', help='model save directory', default='./ckpt')

	args = parser.parse_args()
	input_generator = InputGenerator(
		args.p, args.gen, args.pm, 
		args.niter, args.lr, args.no_cuda,
		args.step_size, args.seed, args.model_dir
	)
	# Experiment with dnn or not
	print('testing with dnn')
	print(input_generator.test_file(args.py_file, True), '\n')

	print('testing without dnn')
	print(input_generator.test_file(args.py_file, False))

if __name__ == "__main__":
    main()
