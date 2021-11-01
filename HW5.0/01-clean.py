import json

filename = ['Children of the Frost.txt', 'Musical Memories.txt', 'Poirot Investigates.txt']

x = []
y = []


for fn in filename:

	with open('raw/'+fn, 'r') as fd:
		content = fd.read().strip().split('\n\n')
	fd.close()

	content = content[30:]
	content = [x.replace('\n', ' ') for x in content]
	content = [x.split() for x in content]
	content = [x for x in content if len(x) >= 20]
	content = [' '.join(x) for x in content]

	label = fn.split('.')[0]

	x += content
	y += [label] * len(content)

data = {'x': x, 'y': y}

with open('data.json', 'w') as fw:
	json.dump(data, fw)
fw.close()