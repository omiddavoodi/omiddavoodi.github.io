import os, shutil

lst = os.listdir('mnist_png/training2')

for i in lst:
	shutil.rmtree('mnist_png/training2/' + i)
	os.mkdir('mnist_png/training2/' + i)

f = open('worker0-tokeep.txt', 'r')
s = f.read()
f.close()

t = s.split('\n')

for i in t:
	shutil.copyfile(i, i.replace('training', 'training2'))
	print(i + " copied")