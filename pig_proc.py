import matplotlib.pyplot as plt
from lmj import c3d

class C3DContent:
	def __init__(self, path):
		reader = c3d.Reader(open(path, 'rb'))
		self.video, self.analog = zip(*reader.read_frames())

	def plot(self, marker_index, fig=None, limits=None, mstyle=None):
		'''Plots a video signal of given index.'''
		marker = self.getmarker(marker_index)
		if limits is not None:
			marker = marker[limits[0]:limits[1]]
		plt.figure(fig)
		for i in range(4):
			plt.subplot(4, 1, i+1)
			plt.cla()
			plt.plot([v[i] for v in marker], marker=mstyle)

	def getmarker(self, marker_index):
		'''Retreives a sequence of a specific video signal.'''
		return [f[marker_index] for f in self.video]
	