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
		# load the c3d
		self.file = path
		self.reader = c3d.Reader(open(path, 'rb'))
		
		# get data
		self.video, self.analog = zip(*self.reader.read_frames())
		
		# get marker labels
		labels_raw = self.reader.group('POINT').params['LABELS']
		label_len, num_markers = labels_raw.dimensions
		self.labels = {}
		for i in xrange(num_markers):
			label = s.strip(labels_raw.bytes[label_len*i:label_len*(i+1)], ' ')
			if C3DContent._is_marker(label):
				self.labels[label] = i
		
		# get force plates info
		fp_group = self.reader.group('FORCE_PLATFORM')
		self.force_plates = ForcePlates(fp_group)

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
		for label in self.labels.keys():
			yield self.getmarker(label)
	
	@staticmethod
	def _label_matches(base_label):
		base_label = s.split(base_label, ':')[-1]
		base_label = s.lower(base_label)
		return lambda tested_label: s.lower(tested_label) == base_label or \
			s.lower(s.split(tested_label, ':')[-1]) == base_label
	
	@staticmethod
	def _is_marker(label):
		label = s.split(label, ':')[-1]
		return len(label) in xrange(4,6) and label == s.upper(label)


def load(path):
	ext = s.lower(os.path.splitext(path)[1])
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
