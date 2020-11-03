import numpy as np
import time
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Calculate accuracies on overrepresented genera')
parser.add_argument("data_dir", help="Directory which contains numpy array data", type=str)
args = parser.parse_args()

true_cat = np.load(args.data_dir + 'overrep_y.npy')
names = np.load(args.data_dir + 'overrep_names.npy')

genus = []

for i in names:
	if ' ' in i:
		genus.append(i.split(' ')[3])
	else:
		genus.append(i.split('_')[0][1:])

genus = np.asarray(genus)
bgds_cat = np.argmax(np.load('DNN_overrep_softmax.npy'), axis=1)
plasclass_chr_cat = np.genfromtxt(args.data_dir + 'overrep_chr.plasclass', delimiter='\t', dtype=str)
plasclass_pl_cat = np.genfromtxt(args.data_dir + 'overrep_pl.plasclass', delimiter='\t', dtype=str)
plasclass_cat = np.vstack((plasclass_chr_cat, plasclass_pl_cat))
plasclass_cat = np.asarray(plasclass_cat[:,1], dtype=float) >= 0.5
plasflow_chr_cat = np.genfromtxt(args.data_dir + 'overrep_chr.plasflow_pred.tsv', skip_header=1, delimiter='\t', dtype=str)
plasflow_pl_cat = np.genfromtxt(args.data_dir + 'overrep_pl.plasflow_pred.tsv', skip_header=1, delimiter='\t', dtype=str)
plasflow_cat = np.vstack((plasflow_chr_cat, plasflow_pl_cat))
plasflow_cat = np.asarray([x.startswith('plasmid.') for x in plasflow_cat[:,5]])
print(np.unique(genus, return_counts=True))
gen_acc_data = []

print('\t'.join(['', 'PlasFlow', 'PlasFlow', 'PlasClass', 'PlasClass', 'BGDS', 'BGDS']))
print('\t'.join(['', 'Chrom.', 'Plasmid', 'Chrom.', 'Plasmid', 'Chrom.', 'Plasmid']))
print('\t'.join(['Genus', 'Acc. (%)', 'Acc. (%)', 'Acc. (%)', 'Acc. (%)', 'Acc. (%)', 'Acc. (%)']))

for gen in np.unique(genus):
	bgds_chr_acc = 1 - np.mean(bgds_cat[(true_cat==0) & (genus==gen)])
	bgds_pl_acc = np.mean(bgds_cat[(true_cat==1) & (genus==gen)])
	plasflow_chr_acc = 1 - np.mean(plasflow_cat[(true_cat==0) & (genus==gen)])
	plasflow_pl_acc = np.mean(plasflow_cat[(true_cat==1) & (genus==gen)])
	plasclass_chr_acc = 1 - np.mean(plasclass_cat[(true_cat==0) & (genus==gen)])
	plasclass_pl_acc = np.mean(plasclass_cat[(true_cat==1) & (genus==gen)])
	gen_acc_data.append([plasflow_chr_acc, plasflow_pl_acc, plasclass_chr_acc, plasclass_pl_acc, bgds_chr_acc, bgds_pl_acc])
	print('\t'.join([gen] + [str(np.round(x*100,2)) for x in gen_acc_data[-1]] ))

gen_acc_data = np.asarray(gen_acc_data)
sort_ind = np.asarray([1,9,0,6,8,2,5,3,4,7])
gen_acc_data = gen_acc_data[sort_ind]
PF_acc = np.mean(np.asarray(gen_acc_data[:,1:3], dtype=float), axis=1)
PC_acc = np.mean(np.asarray(gen_acc_data[:,3:5], dtype=float), axis=1)
my_acc = np.mean(np.asarray(gen_acc_data[:,5:], dtype=float), axis=1)

ind = np.arange(10) 
width = 0.2   
plt.figure(figsize=(10, 7))
plt.bar(ind, PF_acc, width, label='PlasFlow')
plt.bar(ind + width, PC_acc, width, label='PlasClass')
plt.bar(ind + 2*width, my_acc, width, label='DNN with GBDS')
plt.ylabel('Accuracy (%)')
plt.xlim(-0.7,10)
plt.ylim(50,100)
plt.xticks(ind + width, gen_acc_data[:,0], rotation='vertical')
plt.legend(loc=2, prop={'size': 10})
plt.gcf().subplots_adjust(bottom=0.25)
plt.savefig("overrep_accuracy_comparison.pdf")
