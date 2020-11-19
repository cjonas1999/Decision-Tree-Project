import math

class DecisionTree:
	def __init__(self, meta_filename, training_filename):
		self.attribute_names = []
		self.attributes = [] #the classifications are the last item in self.attributes
		self.training_set = []
		self.tree = None

		self.readMetaFile(meta_filename)
		self.createTrainingSet(training_filename)

		#given = ["" for _ in range(len(self.attributes)-1)]
		given = []
		self.tree = self.generateTree(given)


	def __str__(self):
		return "\n".join(self.treeStringHelper(self.tree, 0))


	def treeStringHelper(self, node, height):			
		lines = []
		lines.append("   "*height + self.attribute_names[node.val])

		for attr in self.attributes[node.val]: 
			lines.append("   "*height + "= " + attr)
			if node.children[attr] == None:
				lines.append("   "*(height+2) + node.classification[attr])
			else:
				lines = lines + self.treeStringHelper(node.children[attr], height+1)

		return lines


	def readMetaFile(self, meta_filename):
		meta_f = open(meta_filename, "r")
		i = 0
		for line in meta_f:
			n = line.strip().split(':')#attribute name is n[0]

			self.attributes.append(set())

			self.attribute_names.append(n[0])
			for val in n[1].split(','):
				self.attributes[i].add(val)

			i += 1

		meta_f.close()
		#print(self.attributes)


	def createTrainingSet(self, training_filename):
		train_f = open(training_filename, "r")
		for line in train_f:
			formatted_line = line.strip().split(',')
			self.training_set.append(formatted_line)
		train_f.close()

		#print(self.training_set)
    

	def generateTree(self, given):#given = array of tuples, of attribute index and value
		newNode = None

		#Calculate INFO(D) for classification
		info_class = 0
		total_rows = 0

		class_totals = {}
		for classification in self.attributes[-1]:
			class_totals[classification] = 0

		for i in range(len(self.training_set)):
			line = self.training_set[i]
			skip = False
			for g in given:
				if line[g[0]] != g[1]:
					skip = True

			if skip:
				continue
			else:
				class_totals[line[-1]] += 1
				total_rows += 1

		for count in class_totals.values():
			if count > 0:
				info_class -= count/total_rows * math.log(count/total_rows, 2)


		#Check if there are existing examples
		if total_rows == 0:
			return None


		#Calculate GAIN for each attribute
		gain_attr = []
		attr_counts = []
		class_counts = []

		i = 0
		for attr in self.attributes[:-1]:
			skip = False
			for g in given:
				if i == g[0]:
					skip = True
			if skip:
				i += 1
				gain_attr.append(0)
				attr_counts.append({})
				class_counts.append({})
				continue

			info = 0
			attr_count = {}
			class_count = {}
			for attr in self.attributes[i]:
				attr_count[attr] = 0
				class_count[attr] = {}

				for classification in self.attributes[-1]:
					class_count[attr][classification] = 0

			for line in self.training_set:
				skip_line = False
				for g in given:
					if line[g[0]] != g[1]:
						skip_line = True
				if skip_line:
					continue
				attr_count[line[i]] += 1
				class_count[line[i]][line[-1]] += 1


			for attr, attr_total in attr_count.items():
				if attr_total == 0:
					continue
				newinfo = 0
				for classification in self.attributes[-1]:
					ratio = class_count[attr][classification]/attr_total
					if ratio > 0:
						newinfo -= ratio * math.log(ratio, 2)

				info += newinfo * attr_total/total_rows


			attr_counts.append(attr_count)
			class_counts.append(class_count)

			gain_attr.append(info_class - info)

			i += 1

		maxIndex = gain_attr.index(max(gain_attr))
		newNode = self.TreeNode(maxIndex)

		#determine which can be classified or not
		notClassified = []
		for attr, class_count in class_counts[maxIndex].items():
			aboveZero = []
			for classification, count in class_count.items():
				if count > 0:
					aboveZero.append(classification)

			if len(aboveZero) == 1:
				newNode.classify(aboveZero[0], attr)
			else:
				notClassified.append(attr)

		for attr in notClassified:
			newGiven = given + [(maxIndex, attr)]
			childNode = self.generateTree(newGiven)
			if childNode == None:
				maxClass = max(class_totals, key=class_totals.get)
				newNode.classify(maxClass, attr)
			else:
				newNode.insert(childNode, attr)

		return newNode		


	def classifyFile(self, infile_name, outfile_name):
		inf = open(infile_name, "r")
		outf = open(outfile_name, "w+")
		for line in inf:
			indword = line.strip().split(',')
			classification = self.bestprobability(self.tree, indword)

			for word in indword[:-1]:
				outf.write(word)
				outf.write(",")
			outf.write(classification+'\n')
         
		inf.close()
		outf.close()
    
    
	def bestprobability(self, node, idword):
		name = self.attribute_names[node.val]
		index = self.attribute_names.index(name)
        
		for attr in self.attributes[node.val]:
			if(attr == idword[index]):
				#print(attr)
				if(node.children[attr] == None):
					return node.classification[attr]
				else:
					skip = True
					a_tt = attr
                    
					while(skip):
						node = node.children[a_tt]
						name = self.attribute_names[node.val]
						index = self.attribute_names.index(name)
                        
						for att in self.attributes[node.val]:
							a_tt = att
							if(att == idword[index]):
								#print(att)
								if(node.children[att] == None):
									return node.classification[att]
									skip = False

                    
                    
	def calculateAccuracy(self, filename):
		correct = 0
		incorrect = 0

		infile = open(filename, "r")

		for line in infile:
			splitline = line.strip().split(',')
			classification = self.bestprobability(self.tree, splitline)

			if classification == splitline[-1]:
				correct += 1
			else:
				incorrect += 1

		return correct/(correct + incorrect)



	class TreeNode:
		def __init__(self, val):
			self.val = val
			self.children = {}
			self.classification = {}

		def insert(self, newNode, edgeName):
			self.children[edgeName] = newNode
			self.classification[edgeName] = None

		def classify(self, classification, edgeName):
			self.children[edgeName] = None
			self.classification[edgeName] = classification

		def get(self, edgeName):
			return self.children[edgeName]



def main():
	running = True

	metafile = input("Enter meta file name:")
	trainingfile = input("Enter training file name:")
	d = DecisionTree(metafile, trainingfile)
	print(d)
	infile = input("Enter test file name:")
	outfile = input("Enter output file name:")
	d.classifyFile(infile, outfile)
	filename = input("Enter test file name:")
	print(d.calculateAccuracy(filename))
'''
	while running:
		choice = input("\n\n1 to train.\n2 to classify a file.\n3 to calculate accuracy.\n4 to exit\nEnter choice: ")

		if choice == "1":
			metafile = input("Enter meta file name:")
			trainingfile = input("Enter training file name:")
			b = DecisionTree(metafile, trainingfile)
			print(b)

		elif choice == "2":
			infile = input("Enter test file name:")
			outfile = input("Enter output file name:")
			b.classifyFile(infile, outfile)

		elif choice == "3":
			infile = input("Enter test file name:")
			accuracy = b.calculateAccuracy(infile)
			print(f"{infile} was {accuracy*100}% accurate")

		elif choice == "4":
			running = False
'''
main()