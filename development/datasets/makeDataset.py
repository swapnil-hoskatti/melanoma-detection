import os

seg_img = os.listdir('seg')
new_csv = []

with open('features _with_labels.csv') as f:
	full_csv = f.read()
	for i, row in enumerate(full_csv.split('\n')):
		if i/200 == 0:
			print(f"{i+1} hua ab tak")
		img_label = row.split(',')[0]
		if img_label+'.jpg' in seg_img:
			new_csv += [row]

print("List Ready")

with open('new_database_with_labels.csv', 'w') as f:
	for row in new_csv:
		f.write("%s\n" % row)

print("Ho gaya ab")