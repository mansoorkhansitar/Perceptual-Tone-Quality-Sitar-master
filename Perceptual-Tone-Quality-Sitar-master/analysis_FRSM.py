###########################################################################################################################################
#### Importing libraries ####
import os,sys
from sklearn.cluster import KMeans
import numpy as np
from essentia import standard; from essentia.standard import *
import matplotlib.pyplot as plt
###########################################################################################################################################
#### Function to detect onset ####
def onset_det(filei,audio,rate):
	list_spec=[] 
	list_flux = []

	for frame in FrameGenerator(audio, frameSize=4096, hopSize=2048, startFromZero=True):
		spec = Spectrum()(Windowing(type='hann')(frame))
		spec = np.log(1 + 1000*spec)
		list_spec.append(spec)

	list_flux.append(sum(np.array(list_spec[0])))
	for i in range(1, len(list_spec)):
		diff = np.array(list_spec[i]) - np.array(list_spec[i-1])
		diff = (diff + abs(diff))/2
		list_flux.append(sum(diff))
	return int(0.1*fs)+(list_flux.index(max(list_flux)))*2048 #return onset position
###########################################################################################################################################
#### Function to extract envelope and calculate duration of decay ####
def exp_env(audio, step):
  step = int(step)
  audio = np.abs(audio)
  envelope = []; env_x = []
  for i in range(0, len(audio), step):
      env_x += [i+np.argmax(audio[i:i+step])]
      envelope += [np.max(audio[i:i+step])]
  ##Uncomment below line to see envelope overlaid on audio
# plt.plot(range(0,len(audio)), audio,'g'); plt.plot(env_x, envelope,'r'); plt.show()
  env_x=np.array(env_x)
  envelope = np.array(envelope)
  start = env_x[np.where(envelope==envelope.max())[0]]
  locs = np.where(envelope<0.1*envelope.max())[0]
  if len(locs)<1:
    stop1 = env_x[-1]
  else:
    stop1 = env_x[locs[np.where(locs > np.where(envelope==envelope.max())[0][0])][0]]
  locs = np.where(envelope<0.01*envelope.max())[0] #duration from max amplitude to 10% of max
  if len(locs)<1:
    stop2 = env_x[-1]
  else:
    stop2 = env_x[locs[np.where(locs > np.where(envelope==envelope.max())[0][0])][0]]
  return stop1 #can also return 'envelope and env_x' if needed
###########################################################################################################################################
#### Main ####
fs=44100
labels=[] #populate with names of all files processed below
data_folder = './data1/' # main folder containing three sub-folders as below; place this script in the folder containing 'data_folder'
dirs1 = ['data_ma', 'data_sa', 'data_pa']
i=0;
lim_l=0; lim_h=20000 #min and max frequencies within which to compute the features
for dir1 in dirs1: 
	dataset=[]
	dirs=os.listdir(data_folder+dir1)
	for diri in dirs:
		if diri == 'Neither Band Nor Khula':
		  continue
		files = os.listdir(data_folder+dir1+'/'+diri)
		if diri=='Band Jawari':
			n_band=len(files)
		#Iterate over each file in 'data_sa', 'data_ma' and 'data_pa' directories
		for filei in files:
			labels.append(diri+', '+filei)
			audio = EasyLoader(filename=data_folder+dir1+'/'+diri+'/'+filei)() #load the audio
			start=int(onset_det(filei,audio,fs)) # get onset

			##Computing features
			##1. Temporal - Duration of decay (uncomment below line to use)
#			decay_dur=(exp_env(audio[start-int(0.1*fs):],0.1*fs))/fs # get duration of decay

			##2. Spectral - High to low-frequency energy ratio; Spectral centroid; Spectral Flux
			list_spec=[] # array to hold the STFT values
			centroids=[]; hilo=[]; list_flux=[]
			for frame in FrameGenerator(audio[start:start+int(2.*fs)], frameSize=4096, hopSize=2048, startFromZero=True):
				#can add zero-padding here
				spec = Spectrum()(Windowing(type='hann')(frame))
				list_spec.append(spec)
				centroids.append(lim_l+Centroid(range=lim_h-lim_l)(spec[lim_l*4096/fs:lim_h*4096/fs]))
				hilo.append(np.log10(sum(spec[5000*4096/fs:16000*4096/fs]**2)/sum(spec[0*4096/fs:5000*4096/fs]**2)))

			##Uncomment 5 lines below to compute spectral flux; skipping it to run code faster
#			list_flux.append((sum(np.array(list_spec[0]))))
#			for x in range(1, len(list_spec)):
#				diff = np.array(list_spec[x]) - np.array(list_spec[x-1])
#				diff = (diff + abs(diff))/2
#				list_flux.append(sum(diff))

			##Uncomment one of the four lines below based on feature needed
#			dataset.append(decay_dur)
#			dataset.append([np.array(hilo).mean()])
			dataset.append([np.array(centroids).mean()])
#			dataset.append([np.array(sum(list_flux)/len(list_flux)).mean()])

	kmeans = KMeans(n_clusters=2, init='random').fit(dataset) #k-means clustering using a single feature

	##Plotting
	##set the value of feature name according to feature chosen
	feature_name='Spectral Centroid (Hz)'
	plt.ylabel(feature_name)
	plt.scatter(i*np.ones(n_band),dataset[:n_band],marker="x")
	plt.scatter((i+5)*np.ones(len(dataset)-n_band),dataset[n_band:],facecolors='none') 
	plt.plot(range(i-1,i+8), np.ones(9)*kmeans.cluster_centers_.mean(), linestyle='--')
	plt.title('Band vs Khula Plot for first 3 strings of Sitar')
	plt.text(i+0.5,kmeans.cluster_centers_.mean()+50,str(round(kmeans.cluster_centers_.mean(),2)),fontsize=9)
	plt.xticks([0,5,10,15,20,25], ('Band(46)', 'Khula(20)','Band(6)', 'Khula(48)','Band(14)', 'Khula(31)') )
	i=i+10;
plt.xlabel('First String(Ma)              Second String(Sa)              Third String(Pa)',position = (0.5,5));
plt.show()
###########################################################################################################################################
