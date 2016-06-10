import numpy as np
from math import sqrt
#action_type,combined_shot_type,game_event_id,game_id,lat,loc_x,loc_y,lon,minutes_remaining,period,playoffs,season,seconds_remaining,shot_distance,shot_made_flag,shot_type,shot_zone_area,shot_zone_basic,shot_zone_range,team_id,team_name,game_date,matchup,opponent,shot_id
class Classifier(object):
	'''
	Abstract class for every classifier.
	Every classifier should inherit 'Classifier'.
	'''
	def __init__(self):
		self.unnecessary=[]
		self.numericals=[]
		self.categoricals=[]

	def loaddata(self,print_legend):
		numerize=True
		process_attribute=True
		scaling=True
		filename = 'data_refined.csv'
		with open(filename, 'r') as f:
			input = f.readlines()

		#1. listize all data.
		legend = input[0].replace('\n','').split(',')
		input=input[1:]		
		removed = [line.replace('\n','').split(',') for line in input]

		#2. remove unnecessary attributes.
		for attr in self.unnecessary:
			idx=legend.index(attr)
			removed = [record[:idx] + record[idx+1:] for record in removed]
			legend.remove(attr)
					
		#3. pre-process 'process attributes'		
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

		if 'game_date' in legend:
			legend.remove('game_date')
			legend.append('year')
			legend.append('month')
			legend.append('day')

		#3-1. Coalsce minutes_remaining and seconds_remaining
		if 'minutes_remaining' in legend and 'seconds_remaining' in legend:
			m_idx = legend.index('minutes_remaining')
			s_idx = legend.index('seconds_remaining')
			for idx_record, record in enumerate(removed):
				seconds = int(record[s_idx])
				minutes = int(record[m_idx])
				removed[idx_record][s_idx]=seconds + 60 * minutes
				removed[idx_record].pop(m_idx)
			legend.remove('minutes_remaining')
				
			
		#4. Numerize String attributes.
		if numerize:
			for idx in range(len(legend)):
				if legend[idx] in self.numericals:
					for idx_record in range(len(removed)):
						removed[idx_record][idx] = float(removed[idx_record][idx])
		
		#5. Specific things!					
		removed, legend= self.loaddata_specific(removed, legend)
		
		#6. Scale the attributes.
		if scaling:
			n=len(removed)			
			for idx in range(len(legend)):				
				attr = legend[idx]
				if attr.startswith('_') or attr=='shot_made_flag':
					continue

				sum=0
				sqr_sum=0
				#6-1. get mean and variance.
				for record in removed:
					value = record[idx]

					if type(value) == str:
						print(attr)

					sum+=value
					sqr_sum+=value*value

				mean = sum/n
				variance = sqr_sum/n - mean*mean
				std=sqrt(variance)
				#6-2. normalize attributes for zero-mean, and unit-variance
				for idx_record,record in enumerate(removed):
					value=removed[idx_record][idx]
					value=value-mean
					value=value/std
					removed[idx_record][idx]=value
			

		if print_legend:
			print('='*10,'legend','='*10)
			[print(attr) if attr!='shot_made_flag' else None for attr in legend]
			print('='*25)
		#for i in range(len(legend)):
		#	if legend[i]=='shot_made_flag':
		#		continue
		#	print(legend[i], '\t\t\t',removed[0][i])

		#print(legend)
		#print(removed[0])
		#print(len(legend))
		#print(len(removed[0]))

		#=================End of pre-processing=================

		# split data into 'Train data' and 'Test data'
		train_data = list(filter(lambda record: record.count('')==0, removed))
		test_data = list(filter(lambda record: record.count(''), removed))

		# split X and Y.
		index_y = legend.index('shot_made_flag')
		for i in range(len(train_data)):
			train_data[i][index_y] = int( train_data[i][index_y] )
		train_x = [record[:index_y] + record[index_y+1:] for record in train_data]
		train_y = [record[index_y] for record in train_data]
		self.test_x = [record[:index_y] + record[index_y+1:] for record in test_data]

		valid_fraction=0.9
		train_num=int(len(train_x)*valid_fraction)
		self.valid_x = train_x[train_num:]
		self.train_x = train_x[:train_num]

		self.valid_y = np.array(train_y[train_num:])
		self.train_y = np.array(train_y[:train_num])
		print('Data loaded.')
	
	def export_answer(self):
		filename = 'sample_submission.csv'
		with open(filename, 'r') as f:
			tuples = f.readlines()

		firstline = tuples[0]
		tuples=tuples[1:]
		filename = 'try.csv'
		with open(filename, 'w') as f:
			f.write(firstline)
			for idx,line in enumerate(tuples):
				id = line.split(',')[0]
				f.write(id+','+str(self.test_y[idx])+'\n')