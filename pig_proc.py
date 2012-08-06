import matplotlib.pyplot as plt
from lmj import c3d
import string as s
import os
import struct
import numpy

class ForcePlates(object):
	def __init__(self, group):
		self.num_plates = abs(struct.unpack('h', group.params['USED'].bytes)[0])
		# corners
		corners_param = group.params['CORNERS']
		corners_flat = struct.unpack('f'*3*4*self.num_plates, corners_param.bytes)
		self.corners = numpy.reshape(corners_flat, (self.num_plates, 4, 3))
		# origin
		origin_param = group.params['ORIGIN']
		origin_flat = struct.unpack('f'*3*self.num_plates, origin_param.bytes)
		self.origin = numpy.reshape(origin_flat, (self.num_plates, 3))
	
	def write(self, group):
		pass
	
	def plot(self, fig=None):
		f = plt.figure(fig or 'Force plates'); plt.clf()
		a = f.gca(projection='3d')
		colors = map(lambda x: x, 'rbgcmyk'[::-1])
		for i in xrange(self.num_plates):
			c = colors.pop()
			# corners
			x, y, z = zip(*self.corners[i])
			a.scatter(x, y, z, c=c)
			# origin
			x, y, z = self.origin[i]
			a.scatter(x, y, z, c=c)
		plt.show()

class C3DContent:
	def __init__(self, path):
		from copy import deepcopy
		import logging
		
		log = logging.FileHandler(path+'_read.log', 'wt')
		log.setLevel(logging.DEBUG)
		logging.root.addHandler(log)
		
		# load the c3d
		self.file = path
		self._reader = c3d.Reader(open(path, 'rb')) # TODO: remove from self
		
		# get a copy of the header
		self.header = deepcopy(self._reader.header)
		
		# get a copy of parameters
		self.groups = deepcopy(self._reader._groups)
		
		# get a copy of video and analog data
		self.video, self.analog = zip(*self._reader.read_frames())
		
		# get marker labels
		labels_raw = self.getgroup('POINT').params['LABELS']
		label_len, num_markers = labels_raw.dimensions
		self.labels = {}
		for i in xrange(num_markers):
			label = labels_raw.bytes[label_len*i:label_len*(i+1)].strip(' ')
			if C3DContent._is_marker(label):
				self.labels[label] = i
		
		# get force plates info
		fp_group = self.getgroup('FORCE_PLATFORM')
		self.force_plates = ForcePlates(fp_group)
		
		logging.root.removeHandler(log)
	
	def getgroup(self, name):
		return self.groups.get(name.upper(), None)

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
	
	def save(self, path):
		import logging
		
		log = logging.FileHandler(path+'_write.log', 'wt')
		log.setLevel(logging.DEBUG)
		logging.root.addHandler(log)
		
		# prepare output
		handle = open(path, 'wb')
		writer = c3d.Writer(handle)
		
		# write metadata
		writer.header = self.header
		writer._groups = self.groups
		writer.write_metadata()
		handle.flush()
		
		# write video and analog data
		frames = zip(self.video, self.analog)
		writer.write_frames(frames)
		
		logging.root.removeHandler(log)

	def getmarker(self, index_or_label):
		'''Retreives a sequence of a specific video signal.'''
		index = -1
		label = ''
		if type(index_or_label) == str:
			try:
				label = filter(C3DContent._label_matches(index_or_label), self.labels.keys())[0]
			except IndexError:
				raise NameError('label \'%s\' does not match any of stored markers:\n%s' % (label, self.labels.keys()))
			try:
				index = self.labels[label]
			except KeyError:
				raise NameError('no marker of name \'%s\'' % label) # this should already be covered!
		elif type(index_or_label) == int:
			index = index_or_label
			try:
				label = [k for k,v in self.labels.iteritems() if v==index][0]
			except IndexError:
				raise IndexError('no marker of index %d' % index)
		else:
			raise TypeError('\'index_or_label\' should be a marker index or its label')
		
		return (label, [f[index] for f in self.video])

	def getmarkers(self):
		for index in self.labels.itervalues():
			yield self.getmarker(index)
	
	@staticmethod
	def _label_matches(base_label):
		base_label = base_label.split(':')[-1]
		base_label = base_label.lower()
		return lambda tested_label: tested_label.lower() == base_label or \
			tested_label.split(':')[-1].lower() == base_label
	
	@staticmethod
	def _is_marker(label):
		label = label.split(':')[-1]
		return len(label) in xrange(4,6) and label == label.upper()


def load(path):
	ext = os.path.splitext(path)[1].lower()
	if ext != '.c3d':
		raise ValueError('expected .c3d path, got '+ext)
	
	return C3DContent(path)

def save(content, dirpath, limits=None, separate=True, markers=None):
	'''Saves c3d trajectories into a txt file.
	
	content   - C3DContent instance with trajectories
	dirpath   - directory path to be written into
	limits    - tuple (beg,end) for trajectories cropping (default: no cropping)
	separate  - write each trajectory into a separate file? (default: yes)
	markers   - list of markers to be written (default: all)
	'''
	if not os.path.isdir(dirpath):
		dirpath = os.path.split(dirpath)[0]
	
	make_path = lambda filename: os.path.join(dirpath, filename+'.txt')
	h = open(make_path('all_markers'), 'wt') if not separate else None
	
	for label, marker in content.getmarkers():
		if markers is not None and len(filter(C3DContent._label_matches(label), markers)) == 0:
			continue
		
		filename = s.split(label, ':')[-1] if separate else None
		h = open(make_path(filename), 'wt') if separate else h
		
		marker = marker[limits[0]:limits[1]] if limits else marker
		
		for pt in marker:
			if pt[3] != -1.0:
				h.write('%f %f %f\n' % (pt[0],pt[1],pt[2]))
	
	markers_str = 'all markers' if markers is None else 'markers:\n' + str(markers) + '\n'
	output_str = 'directory:\n' + dirpath if separate else 'file:\n' + h.name
	print 'Written', markers_str, 'to', output_str
