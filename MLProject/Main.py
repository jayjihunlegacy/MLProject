
def import_data():
	filename = 'data_refined.csv'
	with open(filename, 'r') as f:
		input = f.readlines()

	legend = input[0].replace('\n','').split(',')
	instruction = input[1].replace('\n','').split(',')

	input=input[2:]
	# listize all data.
	listized = [line.replace('\n','').split(',') for line in input]

	# remove unnecessary attributes.
	removed=[]
	for record in listized:
		newrecord = []
		for idx,attr in enumerate(record):
			if instruction[idx]!='no':
				newrecord.append(attr)
		removed.append(newrecord)

	legend = list(filter(lambda attr: instruction[legend.index(attr)] != 'no',legend))
	instruction = list(filter(lambda inst: inst!='no',instruction))
			

	# classify 'categorical attributes'
	for idx, attr in enumerate(legend):
		if attr=='combined_shot_type':
			categories=['Bank Shot', 'Dunk', 'Hook Shot', 'Jump Shot', 'Layup', 'Tip Shot']
			pass
		elif attr=='shot_type':
			categories=['2PT Field Goal', '3PT Field Goal']
			pass
		elif attr=='shot_zone_area':
			categories=['Back']
			pass
		elif attr=='shot_zone_basic':
			pass
		elif attr=='shot_zone_range':
			pass
		elif attr=='opponent':
			pass
		else:
			print('Error')
	# split data into 'Train data' and 'Test data'
	train_data = list(filter(lambda record: record.count('')==0, removed))
	test_data = list(filter(lambda record: record.count('')!=0, removed))

	# split X and Y.
	index_y = instruction.index('label')
	train_x = [record[:index_y] + record[index_y+1:] for record in train_data]
	train_y = [record[index_y] for record in train_data]
	test_x = [record[:index_y] + record[index_y+1:] for record in test_data]



import_data()