from Classifiers import *
def import_data(numerize=True,numerize_category=False,process_attribute=True):
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
			
	if numerize_category:
		# classify 'categorical attributes'
		for idx, attr in enumerate(legend):
			if attr=='combined_shot_type':
				categories={'Bank Shot' : 1, 'Dunk' : 2, 'Hook Shot' : 3, 'Jump Shot' : 4, 'Layup' : 5, 'Tip Shot' : 6}
				found_so_far=10
			
			elif attr=='shot_type':
				categories={'2PT Field Goal' : 1, '3PT Field Goal' : 2}
				found_so_far=10
			
			elif attr=='shot_zone_area':
				categories={'Back Court(BC)':1, 'Left Side(L)':2, 'Left Side Center(LC)':3, 'Center(C)':4, 'Right Side Center(RC)':5, 'Right Side(R)':6}
				found_so_far=10
			
			elif attr=='shot_zone_basic':
				categories={'Above the Break 3':1, 'Mid-Range':2, 'Restricted Area':3, 'In The Paint (Non-RA)':4, 'Backcourt':5, 'Left Corner 3':6, 'Right Corner 3':7}
				found_so_far=10
			
			elif attr=='shot_zone_range':
				categories={'Less Than 8 ft.':1, '8-16 ft.':2, '16-24 ft.':3, '24+ ft.':4}
				found_so_far=10
			
			elif attr=='opponent':
				categories={}
				found_so_far=0			

			else:
				continue

			for idx_record, record in enumerate(removed):
				if record[idx] in categories.keys():
					removed[idx_record][idx]=categories[record[idx]]
				else:
					found_so_far+=1
					categories[record[idx]]=found_so_far
					removed[idx_record][idx]=found_so_far
	else:
		#delete all categorical attributes
		attribute_num = len(legend)
		for i in reversed(range(attribute_num)):
			if instruction[i] == 'categorical':
				legend.pop(i)
				instruction.pop(i)
				
				for index in range(len(removed)):
					removed[index].pop(i)
				
	if numerize:
		for idx in range(len(instruction)):
			if instruction[idx]=='numerical':
				for idx_record in range(len(removed)):
					removed[idx_record][idx] = float(removed[idx_record][idx])

	# pre-process 'process attributes'
	if process_attribute:
		for idx, attr in enumerate(legend):
			if attr=='game_id':
				prev=0
				found_so_far=0

				for idx_record, record in enumerate(removed):
					if record[idx]==prev:
						removed[idx_record][idx]=found_so_far
					else:
						found_so_far+=1
						prev=record[idx]
						removed[idx_record][idx]=found_so_far
			elif attr=='lon':
				for i in range(len(removed)):
					removed[i][idx]=float(119+float(removed[i][idx]))
			elif attr=='game_date':
				for idx_record, record in enumerate(removed):
					date=record[idx]
					year,month,day = date.split('-')
					year=int(year)
					month=int(month)
					day=int(day)
					removed[idx_record].append(year)
					removed[idx_record].append(month)
					removed[idx_record].append(day)
					removed[idx_record].remove(date)

		legend.remove('game_date')
		legend.append('year')
		legend.append('month')
		legend.append('day')
				
	# split data into 'Train data' and 'Test data'
	train_data = list(filter(lambda record: record.count('')==0, removed))
	test_data = list(filter(lambda record: record.count('')!=0, removed))

	# split X and Y.
	index_y = instruction.index('label')
	for i in range(len(train_data)):
		train_data[i][index_y] = int( train_data[i][index_y] )
	train_x = [record[:index_y] + record[index_y+1:] for record in train_data]
	train_y = [record[index_y] for record in train_data]
	test_x = [record[:index_y] + record[index_y+1:] for record in test_data]

	valid_fraction=0.9
	train_num=int(len(train_x)*valid_fraction)
	valid_x = train_x[train_num:]
	train_x = train_x[:train_num]

	valid_y = train_y[train_num:]
	train_y = train_y[:train_num]

	train_set = (train_x,train_y)
	valid_set = (valid_x,valid_y)
	
	return (train_set, valid_set, test_x)
	
def main():
	classifier = FirstClassifer()
	classifier.train(0.001, n_epoch = 20)
	score=classifier.valid()
	print('Accuracy :',score)

if __name__=='__main__':
	main()