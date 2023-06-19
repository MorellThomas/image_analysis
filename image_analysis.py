#!/bin/python

from sys import modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.stats import normaltest
from scipy.stats import ttest_ind
import os
from skimage.io import imread
from skimage.filters.thresholding import threshold_otsu
from skimage.segmentation import watershed
import tifffile
import seaborn as sns
#import pickle

##########

"""
functions
"""


def spinning(n:int):
	"""
	print spinning / while waiting..
	"""
	
	if n % 4 == 0:
		print("\033[F-")
	elif n % 3 == 0:
		print("\033[F/")
	elif n % 2 == 0:
		print("\033[F|")
	else:
		print("\033[F\\")

def create_splits(n: int):
	"""
	return split tuple for a number of images, e.g.: 6 -> (3,2)
	if n is uneven: do for n + 1
	"""
	if n <= 4:
		return n, 1

	n = n if n % 2 == 0 else n + 1
	first = n // 2
	while n % first != 0:
		first -= 1

	return first, int(n / first)
	

def setup_files(filedir: str, suffix: str):
	"""
	return directory and names of files
	files need to be setup as individual tifs for each channel, otherwise change the code

	filedir as absolute path starting with /, otherwise assumes relative path
	"""

	filedir = os.path.join(os.getcwd(), filedir) if filedir[0] != "/" else filedir
	files = []
	files = [ file.name for file in os.scandir(filedir) if (file.is_file()) and (file.name[-len(suffix):] == suffix) ]
	files.sort()
	if files[0][0] != "1":
		raise Exception(f"file Error: file names should be numbered from 1 onwards, instead starts with '{files[0]}'")

	return filedir,files

def plot_channels(images: list, titles: list, maxs: list, cmaps=["gray", "Greens", "Reds"]):
	"""
	plot n images side-by-side with i different channels
	"""

	n = len(images)
	cols, rows = create_splits(n)
	plt.figure()
	for i,_ in enumerate(images):
		plt.subplot(rows, cols, i+1)
		plt.imshow(images[i], interpolation="none", cmap=cmaps[i], vmin=0, vmax=maxs[i])
		plt.title(titles[i])
		plt.colorbar(fraction=0.046, pad=0.04)
	plt.show()

def bg_detection(images):
	"""
	remove background of images via otsu_thresholding and filling morphological holes in the signal
	also initially labels cells
	"""
	sigma = 1
	img_smooth = [ ndi.gaussian_filter(img, sigma) for img in images ]

	thresh = [ threshold_otsu(img) for img in img_smooth ]
	signal_raw = [ img_smooth[i] > thresh[i] for i,_ in enumerate(img_smooth) ]

	signal = [ ndi.binary_fill_holes(img) for img in signal_raw ]

	cell_labels = [ ndi.label(sig)[0] for sig in signal ]
	n_cells = [ cells.max() for cells in cell_labels ]

	return img_smooth, signal, cell_labels, n_cells

def cleanup_labels(images: list, labels: list, ignore=0):
	"""
	cleans up labels for following constraints:
		no cells at image border
		no cells with oversaturated pixels
	"""

	border_mask = [ ndi.binary_dilation(np.zeros(cells.shape, dtype=bool), border_value=1) for cells in labels ]

	print(f"Cleaning up cell labels . . .")
	k = 0
	for i,channel in enumerate(labels[ignore:]):
		i += ignore
		for cell_ID in np.unique(channel):

			print(f"\033[FCleaning up cell labels . . . {str(round((k/(n_cells[1]+n_cells[2]))*100, 1)).zfill(2)}%")
			k += 1
			
			cell_mask = channel == cell_ID
			cell_border_overlap = np.logical_and(cell_mask, border_mask[i])
			n_overlap_pixels = np.sum(cell_border_overlap)

			if n_overlap_pixels > 0:
				labels[i][cell_mask] = 0	# delete cell at boundary
				continue

			#img_of_cell = np.logical_and(cell_mask, img_smooth[i])
			img_of_cell = cell_mask * images[i]
			maxI = ndi.maximum(img_of_cell)
			if maxI == 255:		# removing oversaturated cells
				labels[i][cell_mask] = 0
				continue

		for new_ID, cell_ID in enumerate(np.unique(labels[i])[ignore:]):
			labels[i][labels[i]==cell_ID] = new_ID + 1	# re-label

	print(f"\033[FCleaning up cell labels . . . 100.0%")
	return labels

def re_label(labels: list, ignore=0):
	"""
	re-labels the label list to ensure a continuous labeling
	"""

	for i,_ in enumerate(labels):
		for new_ID, cell_ID in enumerate(np.unique(labels[i])[ignore:]):
			labels[i][labels[i]==cell_ID] = new_ID + 1	# re-label

	n_cells = [ cells.max() for cells in labels ]

	return labels, n_cells

def check_nucleus_in_cytoplasm(labels: list, ignore=0):
	"""
	check if the nucleus signal has a corresponding cytoplasmic signal
	"""
	center_of_mass = {}
	for nuc_ID in np.unique(labels[2])[1:]:
		nucleus_mask = labels[2] == nuc_ID
		com_r, com_c = ndi.center_of_mass(labels[2], labels=labels[2], index=nuc_ID)
		center_of_mass[nuc_ID] = int(com_r), int(com_c)

		if labels[1][center_of_mass[nuc_ID][0]][center_of_mass[nuc_ID][1]] == 0:
			labels[2][nucleus_mask] = 0	# removing cells where a the nucleus has no corresponding cytoplasm

	labels,_ = re_label(labels, ignore=ignore)
	return labels

def ws_cytoplasm(img: list, signal: list, labels: list, index_cytoplasm: int, index_nuclei: int):
	"""
	expands nucleic labels to the cytoplasmic signal to create labeled cytoplasms
	"""
	mask_for_ws = signal[index_cytoplasm] > 0
	watershed_cytoplasm = watershed(img_smooth[index_cytoplasm], labels[index_nuclei], mask=mask_for_ws)	# re-create cytoplasm area according to nuclei labels and with the clean signal as a mask

	labels[index_cytoplasm] = watershed_cytoplasm
	n_cells = [ cells.max() for cells in labels ]

	return labels, n_cells


def check_area_of_cells(labels: list, ignore=0, index_cytoplasm=1, index_nucleus=2):
	"""
	excludes all cells where the area of the nucleus is bigger than the cytoplasm
	"""
	area = {}
	for i,channel in enumerate(labels[ignore:]):
		area[i] = []
		for cell_ID in np.unique(channel)[ignore:]:
			cell_mask = channel == cell_ID
			
			area[i].append(np.sum(cell_mask))

	for cell_ID in np.unique(labels[index_cytoplasm])[ignore:]:
		cyt_mask = labels[index_cytoplasm] == cell_ID
		nuc_mask = labels[index_nucleus] == cell_ID
		if area[0][int(cell_ID)-1] < area[1][int(cell_ID)-1]:
			labels[index_cytoplasm][cyt_mask] = 0
			labels[index_nucleus][nuc_mask] = 0
			continue

	return re_label(labels, ignore=ignore)


def calculate_results(sample_number: int, result: dict, conditions:list, labels:list, img: list, n_cells: list, ignore=0):
		"""
		calculate the mean intensity ratios (I_cyto / I_nucl)
		"""
		print(f"Calculating intesity results . . .")
		k = 0
		_results = []
		for i,channel in enumerate(labels[ignore:]):
			_results.append({"cell_id":[], "mean_I":[], "cell_area":[]})
			for cell_ID in np.unique(channel)[ignore:]:
				print(f"\033[FCalculating intensity results . . . {str(round((k/(n_cells[1]+n_cells[2]))*100, 1)).zfill(2)}%")
				k += 1
				cell_mask = channel == cell_ID
				
				_results[i]["cell_id"].append(cell_ID)
				_results[i]["mean_I"].append(np.mean(images[i+1], where=cell_mask))
			
		result[conditions[sample_number-1]] = []
		meanI_cyto = []
		meanI_nucl = []
		for s,_ in enumerate(_results[0]["mean_I"]):
			meanI_cyto.append(_results[0]["mean_I"][s])
			meanI_nucl.append(_results[1]["mean_I"][s])
			result[conditions[sample_number-1]].append(meanI_cyto[-1] / meanI_nucl[-1])

		print(f"\033[FCalculating intensity results . . . 100.0%")
		return result

def test_normally(results: list[list], mod=None):
	"""
	test if the data is normally distributed
	"""
	normally = [normaltest(I) for I in results] if mod == None else [ normaltest(mod(I)) for I in results ]
	normally = [ p[1] > 0.05 for p in normally ]
	return normally


def test_significance(results:list[list], mod=None):
	"""
	do t-test if data is normally distributed
	"""
	normally = test_normally(results) if type(mod) == None else test_normally(results, mod=mod)
	normal_indices = [ index for index,normal in enumerate(normally) if normal == True ]
	significance = []
	for i,index in enumerate(normal_indices):
		if i < len(normal_indices) - 1:
			_,ttest = ttest_ind(results[index], results[normal_indices[i+1]]) if mod == None else ttest_ind(mod(results[index]), mod(results[normal_indices[i+1]]))
			if ttest >= 0.05:
				signf = "ns"
			elif (ttest < 0.05) and (ttest >= 0.01):
				signf = "*"
			elif (ttest < 0.01) and (ttest >= 0.001):
				signf = "**"
			else:
				signf = "***"
			significance.append([signf, index, normal_indices[i+1]])
	return significance


def add_significance(maximum: float, columns: list[int], sig: str, n: int):
	"""
	add significance markers to the plot
	"""
	delta = maximum * 0.05
	offset = delta
	plt.plot([columns[0], columns[0], columns[1], columns[1]], [maximum + n * offset, maximum + delta + n * offset, maximum + delta + n * offset, maximum + n * offset], lw=1.5, c="k")
	plt.text((columns[0]+columns[1])*0.5, maximum+delta + n * offset, sig, ha="center", va="bottom", color="k")


def plot_bar_scatter(data, title: str, ylabel: str, xticks: list[str], significance=False, scale={}, palette="vlag", theme="ticks", scatter_alpha=0.7):
	"""
	creates a boxplot with scatter. data has to be as a list. If using pd.DataFrame, you may have to remove the xticklabels

	significance has to be in the format [ "**", 2, 3], or nested lists of the same style

	scale has to be a dictionary with keys 'x' and/or 'y' containing the scaling of the axis
	"""
	sns.set_theme(style=theme)

	if significance and (type(significance[0]) == list):
		for n, correlation in enumerate(significance):
			maxi = np.max([ np.max(col) for i,col in enumerate(data)])
			add_significance(maxi, correlation[1:], correlation[0], n)

	ax = sns.boxplot(data=data, width=0.6, palette=palette)
	ax.set_xticklabels(xticks)
	if scale.get("y"):
		ax.set_yscale(scale["y"])
	if scale.get("x"):
		ax.set_xscale(scale["x"])
	plt.ylabel(ylabel)
	plt.title(title)
	sns.stripplot(data=data, size=4, palette="dark:0.3", linewidth = 0, alpha=scatter_alpha)
	plt.show()

##########



if __name__ == "__main__":
	"""
	main code
	"""
	files_dir, files_names = setup_files("sample_data", ".tif")
	conditions = ["Control 1", "Control 2", "Condition 1", "Condition 2", "Condition 3", "Condition 4"]	# names of samples
	n = 3	# sample number to start with
	starting_n = n
	ignore_channel = 1	# ignore channel 1 (index 0) for analysing
	stats_for_plot = {}
	while [ file for file in files_names if file[:len(str(n))] == str(n) ] != []:

		images = [ imread(os.path.join(files_dir, file)) for file in files_names if file[:len(str(n))] == str(n) ]

		print(f"Analyzing sample {conditions[n-1]}")
		
		img_smooth, signal, labels, n_cells = bg_detection(images)


		labels = cleanup_labels(images, labels, ignore=ignore_channel)

		labels = check_nucleus_in_cytoplasm(labels, ignore=ignore_channel)

		labels, n_cells = ws_cytoplasm(img_smooth, signal, labels, 1, 2)

		labels, n_cells = check_area_of_cells(labels, ignore=ignore_channel)
		
		stats_for_plot = calculate_results(n, stats_for_plot, conditions, labels, images, n_cells, ignore=ignore_channel)
		n += 1


	stats_for_plot = [ IoI for IoI in stats_for_plot.values() ]

	significance = test_significance(stats_for_plot, mod=np.log)

	print("Done!")

	plot_bar_scatter(stats_for_plot, "Whole cell to nuclear intensity ratio", "Intensity ratio", conditions[starting_n-1:], significance=significance)
