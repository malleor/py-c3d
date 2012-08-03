import matplotlib.pyplot as plt
from lmj import c3d
import string as s

class C3DContent:
	def __init__(self, path):
		# load the c3d
		self.reader = c3d.Reader(open(path, 'rb'))
		
		# get data
		self.video, self.analog = zip(*self.reader.read_frames())
		
		# get marker labels
		labels_raw = self.reader.group('POINT').params['LABELS']
		cols, rows = labels_raw.dimensions
		self.labels = []
		for i in xrange(rows):
			self.labels.append(s.strip(labels_raw.bytes[cols*i:cols*(i+1)], ' '))

	def plot(self, marker_index, fig=None, limits=None, mstyle=None):
		'''Plots a video signal of given index.'''
		label, marker = self.getmarker(marker_index)
		if limits is not None:
			marker = marker[limits[0]:limits[1]]
		plt.figure(fig or 'Marker '+label)
		for i in xrange(4):
			plt.subplot(4, 1, i+1)
			plt.cla()
			plt.plot([v[i] for v in marker], marker=mstyle)

	def getmarker(self, index_or_label):
		'''Retreives a sequence of a specific video signal.'''
		index = -1
		if type(index_or_label) == str:
			index = self.labels.index(index_or_label)
		elif type(index_or_label) == int:
			index = index_or_label
		else:
			raise TypeError('\'index_or_label\' should be a marker index or its label')
		return (self.labels[index], [f[index] for f in self.video])
