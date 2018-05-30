
import sys

def gen(num_files):
	file = open("README.md", "w")
	file.write("# plots")
	file.write("\n")
	for x in reversed(range(1, num_files+1)):
		file.write("![Donut Project](images/im" + str(x) + ".png)")
		file.write("\n")
	file.close()

incrementor = int(str(sys.argv[1]))
gen(incrementor)
