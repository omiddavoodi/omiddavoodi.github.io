#!/bin/sh
python3 handledataset.py
scp main.py worker1:~/Desktop/md-gan/
scp main.py worker2:~/Desktop/md-gan/
scp main.py worker3:~/Desktop/md-gan/
scp worker0-tokeep.txt worker1:~/Desktop/md-gan/
scp worker1-tokeep.txt worker2:~/Desktop/md-gan/
scp worker2-tokeep.txt worker3:~/Desktop/md-gan/
ssh worker1 'python3 Desktop/md-gan/copy_data.py'
ssh worker2 'python3 Desktop/md-gan/copy_data.py'
ssh worker3 'python3 Desktop/md-gan/copy_data.py'
python3 copy_data.py
mpirun -np 4 --hostfile hostfile venv/bin/python main.py

