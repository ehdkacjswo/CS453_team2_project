import sys
import subprocess

i = 0
for i in range(10):
    subprocess.call([sys.executable, 'main.py', 'input/function2.py', '--seed', str(i*50)])