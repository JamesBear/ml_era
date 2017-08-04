
DATA_PATH = "ti7_data_0804.txt"
DEFAULT_FORMAT = "{:>12}  {:>12}  {:>12}"

def get_file_content(file_path):
    f = open(file_path, 'r')
    file_content = f.read()
    f.close()
    return file_content

def write_to_file(file_path, file_content):
    f = open(file_path, 'w')
    f.write(file_content)
    f.close()
	
def parse_first_round_data(path):
	result = []

	content = get_file_content(path)
	for line in content.splitlines():
		line = line.strip()
		if len(line) > 0:
			splitted = line.split(',')
			item = []
			item.append(splitted[0])
			numbers = splitted[1].split('-')
			item.append(int(numbers[0]))
			item.append(int(numbers[1]))
			item.append(splitted[2])
			result.append(item)
	
	return result
	
def add_or_create_score(score_dict, name, wins, loses):
	item = score_dict.get(name)
	if item == None:
		item = [0,0]
	item[0] += wins
	item[1] += loses
	score_dict[name] = item
	
	
def get_score_dict(data):
	score_dict = dict()
	for item in data:
		add_or_create_score(score_dict, item[0], item[1], item[2])
		add_or_create_score(score_dict, item[3], item[2], item[1])
	return score_dict
	
		
def dict_to_list(the_dict):
	list = []
	for k,v in the_dict.items():
		list.append([k, v])
	return list
	
def sort_score_dict(score_dict):
	sorted_list = dict_to_list(score_dict)
	sorted_list.sort(key = lambda item:item[1][0]*1000-item[1][1], reverse=True)
	return sorted_list
			
def print_score_list(score_list):
	for item in score_list:
		#print(item[0],'\t',item[1][0],'\t',item[1][1])
		print(DEFAULT_FORMAT.format(item[0], item[1][0], item[1][1]))
		
def main():
	print("\nData file name: " + DATA_PATH + "\n")
	data = parse_first_round_data(DATA_PATH)
	#print(data)
	score_dict = get_score_dict(data)
	#print(score_dict)
	sorted_list = sort_score_dict(score_dict)
	#print(sorted_list)
	print("Teams sorted by wins and loses:\n")
	print(DEFAULT_FORMAT.format("TEAM_NAME", "WINS", "LOSES"))
	print_score_list(sorted_list)
	
main()
input('Press enter to continue..')