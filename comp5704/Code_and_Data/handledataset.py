import os,random

numWorkers = 3
skew = 0.0

maxRate = 1.0/numWorkers

classLst = os.listdir('mnist_png/training')


skews = [[maxRate for i in range(len(classLst))] for i in range(numWorkers)]

for i in range(len(classLst)):
	rateChange = skew*maxRate
	if (i < (len(classLst) // numWorkers) * numWorkers):
		skews[i%numWorkers][i] += rateChange
		for j in range(numWorkers):
			if (j != i%numWorkers):
				skews[j][i] -= rateChange/(numWorkers-1)

# for i in skews:
# 	print(i)

files = [os.listdir('mnist_png/training/' + classLst[i]) for i in range(len(classLst))]

print([(len(files[i]),classLst[i]) for i in range(len(files))])
print([len(files[i]) for i in range(len(files))])


to_keep = [[[] for j in range(len(classLst))] for i in range(numWorkers)]

for j in range(len(classLst)):
	base = 0
	base2 = 0
	for i in range(numWorkers):
		base2 = base
		base += skews[i][j]

		r1 = int(base2*len(files[j]))
		r2 = int(base*len(files[j]))

		for k in range(r1, r2):
			to_keep[i][j].append(files[j][k])



sums = [0 for i in range(len(classLst))]
for i in to_keep:
	for j in range(len(i)):
		sums[j] += len(i[j])

num_total = sum(sums)

extra = num_total % numWorkers

while (extra > 0):
	z = random.randrange(0, numWorkers)
	y = random.randrange(0, len(classLst))
	x = random.randrange(0, len(to_keep[z][y]))
	to_keep[z][y].pop(x)
	extra -= 1




while(True):
	sums = [0 for i in range(len(classLst))]
	for i in to_keep:
		for j in range(len(i)):
			sums[j] += len(i[j])

	min_total = sum(sums)
	min_worker = 0
	max_total = 0
	max_worker = 0

	for ii in range(len(to_keep)):
		i = to_keep[ii]
		#print([len(j) for j in i])
		total = sum([len(j) for j in i])
		#print(total)
		if (total > max_total):
			max_total = total
			max_worker = ii
		if (total < min_total):
			min_total = total
			min_worker = ii

	if (min_total == max_total): break

	z = max_worker
	y = random.randrange(0, len(classLst))
	x = random.randrange(0, len(to_keep[z][y]))
	u = to_keep[z][y].pop(x)
	to_keep[min_worker][y].append(u)
	#extra -= 1

#print (min_total, min_worker)
#print (max_total, max_worker)

class_totals = [0 for i in range(len(classLst))]

for ii in range(len(to_keep)):
	i = to_keep[ii]
	print([len(j) for j in i])
	for j in range(len(i)):
		class_totals[j] += len(i[j])
	total = sum([len(j) for j in i])
	print(total)

print(class_totals)
print([len(files[i]) for i in range(len(files))])


for i in range(len(to_keep)):
	f = open('worker' + str(i) + '-tokeep.txt', 'w')
	s = ""
	for j in range(len(to_keep[i])):
		for k in to_keep[i][j]:
			s += 'mnist_png/training/' + classLst[j] + '/' + k + '\n'

	f.write(s[:len(s)-1])
	f.close()
