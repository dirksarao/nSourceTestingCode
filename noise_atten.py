import csv
from ctypes import sizeof
from turtle import shape
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
import statistics as stat
import math
from scipy.stats import norm

f_4_8_ch0 = open('/home/dirk/adc_raw_data/noise_4_8db_atten/adc_raw_data_ch0.csv')
f_4_8_ch1 = open('/home/dirk/adc_raw_data/noise_4_8db_atten/adc_raw_data_ch1.csv')
f_4_8_ch2 = open('/home/dirk/adc_raw_data/noise_4_8db_atten/adc_raw_data_ch2.csv')
f_4_8_ch3 = open('/home/dirk/adc_raw_data/noise_4_8db_atten/adc_raw_data_ch3.csv')

f_4_ch0 = open('/home/dirk/adc_raw_data/noise_4dB_atten/adc_raw_data_ch0.csv')
f_4_ch1 = open('/home/dirk/adc_raw_data/noise_4dB_atten/adc_raw_data_ch1.csv')
f_4_ch2 = open('/home/dirk/adc_raw_data/noise_4dB_atten/adc_raw_data_ch2.csv')
f_4_ch3 = open('/home/dirk/adc_raw_data/noise_4dB_atten/adc_raw_data_ch3.csv')

f_none_ch0 = open('/home/dirk/adc_raw_data/noise_no_atten/adc_raw_data_ch0.csv')
f_none_ch1 = open('/home/dirk/adc_raw_data/noise_no_atten/adc_raw_data_ch1.csv')
f_none_ch2 = open('/home/dirk/adc_raw_data/noise_no_atten/adc_raw_data_ch2.csv')
f_none_ch3 = open('/home/dirk/adc_raw_data/noise_no_atten/adc_raw_data_ch3.csv')

f_none_skiplna_ch0 = open('/home/dirk/noiseSourceMeasurements/noisesourcemeasurements/adc_raw_data_ch0_no_atten_noise_only.csv')
f_none_skiplna_ch1 = open('/home/dirk/noiseSourceMeasurements/noisesourcemeasurements/adc_raw_data_ch1_no_atten_noise_only.csv')
f_none_skiplna_ch2 = open('/home/dirk/noiseSourceMeasurements/noisesourcemeasurements/adc_raw_data_ch2_no_atten_noise_only.csv')
f_none_skiplna_ch3 = open('/home/dirk/noiseSourceMeasurements/noisesourcemeasurements/adc_raw_data_ch3_no_atten_noise_only.csv')

f_4_skiplna_ch0 = open('/home/dirk/noiseSourceMeasurements/4dB_enabled/adc_raw_data_ch0_noise_only_4dB_atten.csv')
f_4_skiplna_ch1 = open('/home/dirk/noiseSourceMeasurements/4dB_enabled/adc_raw_data_ch1_noise_only_4dB_atten.csv')
f_4_skiplna_ch2 = open('/home/dirk/noiseSourceMeasurements/4dB_enabled/adc_raw_data_ch2_noise_only_4dB_atten.csv')
f_4_skiplna_ch3 = open('/home/dirk/noiseSourceMeasurements/4dB_enabled/adc_raw_data_ch3_noise_only_4dB_atten.csv')

f_4_8_skiplna_ch0 = open('/home/dirk/noiseSourceMeasurements/4dB_8dB_enabled/adc_raw_data_ch0_noise_only_4dB_8dB_atten.csv')
f_4_8_skiplna_ch1 = open('/home/dirk/noiseSourceMeasurements/4dB_8dB_enabled/adc_raw_data_ch0_noise_only_4dB_8dB_atten.csv')
f_4_8_skiplna_ch2 = open('/home/dirk/noiseSourceMeasurements/4dB_8dB_enabled/adc_raw_data_ch0_noise_only_4dB_8dB_atten.csv')
f_4_8_skiplna_ch3 = open('/home/dirk/noiseSourceMeasurements/4dB_8dB_enabled/adc_raw_data_ch0_noise_only_4dB_8dB_atten.csv')

f_sine_test = open('/home/dirk/adc_raw_data/adc_data/adc_raw_data_ch0.csv')

rows = []
rows1 = []
rows2 = []
rows3 = []

rows4 = []
rows5 = []
rows6 = []
rows7 = []

rows8 = []
rows9 = []
rows10 = []
rows11 = []

rows12 = []

rows13 = []
rows14 = []
rows15 = []
rows16 = []

rows17 = []
rows18 = []
rows19 = []
rows20 = []

rows21 = []
rows22 = []
rows23 = []
rows24 = []

#Sine Test

for row12 in csv.reader(f_sine_test):
    rows12.append(row12)

ydata = np.array(rows12).reshape(4096,)
ydata_sine = np.zeros((4096,1))
ydata_sine = ydata

#4_8dB attenuation

#Ch0
for row in csv.reader(f_4_8_ch0):
    rows.append(row)

ydata = np.array(rows).reshape(4096,)
ydata_4_8 = np.zeros((4096,4))
ydata_4_8[:, 0] = ydata

#Ch1
for row1 in csv.reader(f_4_8_ch1):
    rows1.append(row1)

ydata = np.array(rows1).reshape(4096,)
ydata_4_8[:, 1] = ydata

#Ch2
for row2 in csv.reader(f_4_8_ch2):
    rows2.append(row2)

ydata = np.array(rows2).reshape(4096,)
ydata_4_8[:, 2] = ydata

#Ch3
for row3 in csv.reader(f_4_8_ch3):
    rows3.append(row3)

ydata = np.array(rows3).reshape(4096,)
ydata_4_8[:, 3] = ydata


#4dB attenuation

#Ch0
for row4 in csv.reader(f_4_ch0):
    rows4.append(row4)

ydata = np.array(rows4).reshape(4096,)
ydata_4 = np.zeros((4096,4))
ydata_4[:, 0] = ydata

#Ch1
for row5 in csv.reader(f_4_ch1):
    rows5.append(row5)

ydata = np.array(rows5).reshape(4096,)
ydata_4[:, 1] = ydata

#Ch2
for row6 in csv.reader(f_4_ch2):
    rows6.append(row6)

ydata = np.array(rows6).reshape(4096,)
ydata_4[:, 2] = ydata

#Ch3
for row7 in csv.reader(f_4_ch3):
    rows7.append(row7)

ydata = np.array(rows7).reshape(4096,)
ydata_4[:, 3] = ydata


#no attenuation

#Ch0
for row8 in csv.reader(f_none_ch0):
    rows8.append(row8)

ydata = np.array(rows8).reshape(4096,)
ydata_none = np.zeros((4096,4))
ydata_none[:, 0] = ydata

#Ch1
for row9 in csv.reader(f_none_ch1):
    rows9.append(row9)

ydata = np.array(rows9).reshape(4096,)
ydata_none[:, 1] = ydata

#Ch2
for row10 in csv.reader(f_none_ch2):
    rows10.append(row10)

ydata = np.array(rows10).reshape(4096,)
ydata_none[:, 2] = ydata

#Ch3
for row11 in csv.reader(f_none_ch3):
    rows11.append(row11)

ydata = np.array(rows11).reshape(4096,)
ydata_none[:, 3] = ydata

#no attenuation skipped lna

#Ch0
for row13 in csv.reader(f_none_skiplna_ch0):
    rows13.append(row13)

ydata = np.array(rows13).reshape(4096,)
ydata_none_skiplna = np.zeros((4096,4))
ydata_none_skiplna[:, 0] = ydata

#Ch1
for row14 in csv.reader(f_none_skiplna_ch1):
    rows14.append(row14)

ydata = np.array(rows14).reshape(4096,)
ydata_none_skiplna[:, 1] = ydata

#Ch2
for row15 in csv.reader(f_none_skiplna_ch2):
    rows15.append(row15)

ydata = np.array(rows15).reshape(4096,)
ydata_none_skiplna[:, 2] = ydata

#Ch3
for row16 in csv.reader(f_none_skiplna_ch3):
    rows16.append(row16)

ydata = np.array(rows16).reshape(4096,)
ydata_none_skiplna[:, 3] = ydata

#4dB attenuation skipped lna

#Ch0
for row17 in csv.reader(f_4_skiplna_ch0):
    rows17.append(row17)

ydata = np.array(rows17).reshape(4096,)
ydata_4_skiplna = np.zeros((4096,4))
ydata_4_skiplna[:, 0] = ydata

#Ch1
for row18 in csv.reader(f_4_skiplna_ch1):
    rows18.append(row18)

ydata = np.array(rows18).reshape(4096,)
ydata_4_skiplna[:, 1] = ydata

#Ch2
for row19 in csv.reader(f_4_skiplna_ch2):
    rows19.append(row19)

ydata = np.array(rows19).reshape(4096,)
ydata_4_skiplna[:, 2] = ydata

#Ch3
for row20 in csv.reader(f_4_skiplna_ch3):
    rows20.append(row20)

ydata = np.array(rows20).reshape(4096,)
ydata_4_skiplna[:, 3] = ydata

#4_8dB attenuation skipped lna

#Ch0
for row21 in csv.reader(f_4_8_skiplna_ch0):
    rows21.append(row21)

ydata = np.array(rows21).reshape(4096,)
ydata_4_8_skiplna = np.zeros((4096,4))
ydata_4_8_skiplna[:, 0] = ydata

#Ch1
for row22 in csv.reader(f_4_8_skiplna_ch1):
    rows22.append(row22)

ydata = np.array(rows22).reshape(4096,)
ydata_4_8_skiplna[:, 1] = ydata

#Ch2
for row23 in csv.reader(f_4_8_skiplna_ch2):
    rows23.append(row23)

ydata = np.array(rows23).reshape(4096,)
ydata_4_8_skiplna[:, 2] = ydata

#Ch3
for row24 in csv.reader(f_4_8_skiplna_ch3):
    rows24.append(row24)

ydata = np.array(rows24).reshape(4096,)
ydata_4_8_skiplna[:, 3] = ydata


#sampling rate
sr = 2.8*10**9

#fft calcs

X_0a = fft(ydata_4_8[:, 0])
X_1a = fft(ydata_4_8[:, 1])
X_2a = fft(ydata_4_8[:, 2])
X_3a = fft(ydata_4_8[:, 3])

X_0b = fft(ydata_4[:, 0])
X_1b = fft(ydata_4[:, 1])
X_2b = fft(ydata_4[:, 2])
X_3b = fft(ydata_4[:, 3])

X_0c = fft(ydata_none[:, 0])
X_1c = fft(ydata_none[:, 1])
X_2c = fft(ydata_none[:, 2])
X_3c = fft(ydata_none[:, 3])

X_sine = fft(ydata_sine)

N = len(X_0a)
n = np.arange(N)
T = N/sr
freq = n/T

# #plotting

# #Sine Test

# plt.plot(range(len(ydata_sine)), ydata_sine)
# plt.title("raw sine data")
# plt.show()

# plt.plot(freq, np.abs(X_sine))
# plt.title("fft of sine data")
# plt.show()

# plt.hist(ydata_sine, bins=100)
# plt.title("Histogram plot on raw sine data")
# plt.show()

# plt.hist(np.abs(X_sine), bins=100)
# plt.title("Histogram plot on fft of sine data")
# plt.show()

# #4_8dB_attenuation
# fig, axs = plt.subplots(2,2)

# plt.subplots_adjust(left=0.118,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.85, 
#                     wspace=0.4, 
#                     hspace=0.5)

# fig.suptitle("4 and 8dB attenuation (fft)") 
# axs[0,0].plot(freq, np.abs(X_0a))
# axs[0,0].set_title('Channel 0')
# axs[0,0].set_ylabel('Amplitude') 
# axs[0,0].set_xlabel('Frequency(Hz)')
# axs[0,1].plot(freq, np.abs(X_1a))
# axs[0,1].set_title('Channel 1')
# axs[0,1].set_ylabel('Amplitude') 
# axs[0,1].set_xlabel('Frequency(Hz)')
# axs[1,0].plot(freq, np.abs(X_2a))
# axs[1,0].set_title('Channel 2')
# axs[1,0].set_ylabel('Amplitude') 
# axs[1,0].set_xlabel('Frequency(Hz)')
# axs[1,1].plot(freq, np.abs(X_3a))
# axs[1,1].set_title('Channel 3')
# axs[1,1].set_ylabel('Amplitude') 
# axs[1,1].set_xlabel('Frequency(Hz)')

# #4_8dB_attenuation
# fig, axs = plt.subplots(2,2,)

# plt.subplots_adjust(left=0.118,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.85, 
#                     wspace=0.4, 
#                     hspace=0.5)

# fig.suptitle("4 and 8dB attenuation (time domain)")
# axs[0,0].plot(range(len(ydata_4_8[:, 0])), ydata_4_8[:, 0])
# axs[0,0].set_title('Channel 0')
# axs[0,0].set_ylabel('Amplitude') 
# axs[0,0].set_xlabel('Time(s)')
# axs[0,1].plot(range(len(ydata_4_8[:, 0])), ydata_4_8[:, 1])
# axs[0,1].set_title('Channel 1')
# axs[0,1].set_ylabel('Amplitude') 
# axs[0,1].set_xlabel('Time(s)')
# axs[1,0].plot(range(len(ydata_4_8[:, 0])), ydata_4_8[:, 2])
# axs[1,0].set_title('Channel 2')
# axs[1,0].set_ylabel('Amplitude') 
# axs[1,0].set_xlabel('Time(s)')
# axs[1,1].plot(range(len(ydata_4_8[:, 0])), ydata_4_8[:, 3])
# axs[1,1].set_title('Channel 3')
# axs[1,1].set_ylabel('Amplitude') 
# axs[1,1].set_xlabel('Time(s)')

# #4dB_attenuation
# fig, axs = plt.subplots(2,2)
# fig.suptitle("4dB attenuation")
# axs[0,0].plot(freq, np.abs(X_0b))
# axs[0,0].set_title('Channel 0')
# axs[0,1].plot(freq, np.abs(X_1b))
# axs[0,1].set_title('Channel 1')
# axs[1,0].plot(freq, np.abs(X_2b))
# axs[1,0].set_title('Channel 2')
# axs[1,1].plot(freq, np.abs(X_3b))
# axs[1,1].set_title('Channel 3')

# #no_attenuation
# fig, axs = plt.subplots(2,2)
# fig.suptitle("No attenuation")
# axs[0,0].plot(freq, np.abs(X_0c))
# axs[0,0].set_title('Channel 0')
# axs[0,1].plot(freq, np.abs(X_1c))
# axs[0,1].set_title('Channel 1')
# axs[1,0].plot(freq, np.abs(X_2c))
# axs[1,0].set_title('Channel 2')
# axs[1,1].plot(freq, np.abs(X_3c))
# axs[1,1].set_title('Channel 3')

#no_attenuation
fig, axs = plt.subplots(2,2)
fig.suptitle("No attenuation")
axs[0,0].plot(freq, ydata_none[:, 0])
axs[0,0].set_title('Channel 0')
axs[0,1].plot(freq, ydata_none[:, 0])
axs[0,1].set_title('Channel 1')
axs[1,0].plot(freq, ydata_none[:, 0])
axs[1,0].set_title('Channel 2')
axs[1,1].plot(freq, ydata_none[:, 0])
axs[1,1].set_title('Channel 3')

#plt.show()

#no_attenuation
fig, axs = plt.subplots(2,2)
fig.suptitle("No attenuation skipped lna")
axs[0,0].plot(freq, ydata_none_skiplna[:, 0])
axs[0,0].set_title('Channel 0')
axs[0,1].plot(freq, ydata_none_skiplna[:, 1])
axs[0,1].set_title('Channel 1')
axs[1,0].plot(freq, ydata_none_skiplna[:, 2])
axs[1,0].set_title('Channel 2')
axs[1,1].plot(freq, ydata_none_skiplna[:, 3])
axs[1,1].set_title('Channel 3')

#plt.show()

#Gaussian calcs

sdevs = np.zeros((4,3))
means = np.zeros((4,3))
# f_gaus_4_8 = np.zeros((1000,4))
# x = np.arange(-4096/2,4096/2)/(4096/2)

#4_8dB_attenuation

sdevs[0,0] = stat.stdev(abs(X_0a))
sdevs[1,0] = stat.stdev(abs(X_1a))
sdevs[2,0] = stat.stdev(abs(X_2a))
sdevs[3,0] = stat.stdev(abs(X_3a))

means[0,0] = stat.mean(abs(X_0a))
means[1,0] = stat.mean(abs(X_1a))
means[2,0] = stat.mean(abs(X_2a))
means[3,0] = stat.mean(abs(X_3a))

#4dB_attenuation

sdevs[0,1] = stat.stdev(abs(X_0b))
sdevs[1,1] = stat.stdev(abs(X_1b))
sdevs[2,1] = stat.stdev(abs(X_2b))
sdevs[3,1] = stat.stdev(abs(X_3b))

means[0,1] = stat.mean(abs(X_0b))
means[1,1] = stat.mean(abs(X_1b))
means[2,1] = stat.mean(abs(X_2b))
means[3,1] = stat.mean(abs(X_3b))

#no_attenuation

sdevs[0,2] = stat.stdev(abs(X_0c))
sdevs[1,2] = stat.stdev(abs(X_1c))
sdevs[2,2] = stat.stdev(abs(X_2c))
sdevs[3,2] = stat.stdev(abs(X_3c))

means[0,2] = stat.mean(abs(X_0c))
means[1,2] = stat.mean(abs(X_1c))
means[2,2] = stat.mean(abs(X_2c))
means[3,2] = stat.mean(abs(X_3c))

print("Standard deviation of 4 and 8dB attenuation signals:")
print("Channel 0:", sdevs[0][0])
print("Channel 1:", sdevs[1][0])
print("Channel 1:", sdevs[2][0])
print("Channel 3:", sdevs[3][0])
print("\n")

print("Standard deviation of 4dB attenuation signals:")
print("Channel 0:", sdevs[0][1])
print("Channel 1:", sdevs[1][1])
print("Channel 1:", sdevs[2][1])
print("Channel 3:", sdevs[3][1])
print("\n")

print("Standard deviation signals with no attenuation:")
print("Channel 0:", sdevs[0][2])
print("Channel 1:", sdevs[1][2])
print("Channel 1:", sdevs[2][2])
print("Channel 3:", sdevs[3][2])
print("\n")

# for i in range(4):
#     for j in range(1000):
#         f_gaus_4_8[j][i] = (1/(sdevs[i][0]*math.sqrt(2*math.pi)))*np.exp(-((j-means[i][0])**2)/((2*sdevs[i][0])**2))


#4_8dB_attenuation
#hist seperates the range into several bins and return the number of instances in each bin

#4_8dB_attenuation
# fig, axs = plt.subplots(2,2)

# plt.subplots_adjust(left=0.118,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.85, 
#                     wspace=0.4, 
#                     hspace=0.5)

# fig.suptitle("Histogtam Test: 4 and 8dB attenuation (time domain)")
# axs[0,0].hist(ydata_4_8[:, 0], bins=50)
# axs[0,0].set_title('Channel 0')
# axs[0,0].set_ylabel('Bin Frequency')
# axs[0,0].set_xlabel('Amplitude')
# axs[0,1].hist(ydata_4_8[:, 1], bins=50)
# axs[0,1].set_title('Channel 1')
# axs[0,1].set_ylabel('Bin Frequency') 
# axs[0,1].set_xlabel('Amplitude')
# axs[1,0].hist(ydata_4_8[:, 2], bins=50)
# axs[1,0].set_title('Channel 2')
# axs[1,0].set_ylabel('Bin Frequency') 
# axs[1,0].set_xlabel('Amplitude')
# axs[1,1].hist(ydata_4_8[:, 3], bins=50)
# axs[1,1].set_title('Channel 3')
# axs[1,1].set_ylabel('Bin Frequency') 
# axs[1,1].set_xlabel('Amplitude')

# #4dB_attenuation
# fig, axs = plt.subplots(2,2)

# plt.subplots_adjust(left=0.118,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.85, 
#                     wspace=0.4, 
#                     hspace=0.5)

# fig.suptitle("Histogram Test: 4dB attenuation (time domain)")
# axs[0,0].hist(ydata_4[:, 0], bins=50)
# axs[0,0].set_title('Channel 0')
# axs[0,0].set_ylabel('Bin Frequency') 
# axs[0,0].set_xlabel('Amplitude')
# axs[0,1].hist(ydata_4[:, 0], bins=50)
# axs[0,1].set_title('Channel 1')
# axs[0,1].set_ylabel('Bin Frequency') 
# axs[0,1].set_xlabel('Amplitude')
# axs[1,0].hist(ydata_4[:, 0], bins=50)
# axs[1,0].set_title('Channel 2')
# axs[1,0].set_ylabel('Bin Frequency') 
# axs[1,0].set_xlabel('Amplitude')
# axs[1,1].hist(ydata_4[:, 0], bins=50)
# axs[1,1].set_title('Channel 3')
# axs[1,1].set_ylabel('Bin Frequency') 
# axs[1,1].set_xlabel('Amplitude')

#no_attenuation
fig, axs = plt.subplots(4,2,figsize=(10, 10))

plt.subplots_adjust(left=0.118,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.90, 
                    wspace=0.4, 
                    hspace=0.5)

fig.suptitle("Histogtam Test: No attenuation (time domain)")
axs[0,0].hist(ydata_none[:, 0], bins=50)
axs[0,0].set_title('Amplifier: Ch 0')
axs[0,0].set_ylabel('Bin Frequency') 
axs[0,0].set_xlabel('Amplitude')
axs[1,0].hist(ydata_none[:, 1], bins=50)
axs[1,0].set_title('Amplifier: Ch 1')
axs[1,0].set_ylabel('Bin Frequency') 
axs[1,0].set_xlabel('Amplitude')
axs[2,0].hist(ydata_none[:, 2], bins=50)
axs[2,0].set_title('Amplifier: Ch 2')
axs[2,0].set_ylabel('Bin Frequency') 
axs[2,0].set_xlabel('Amplitude')
axs[3,0].hist(ydata_none[:, 3], bins=50)
axs[3,0].set_title('Amplifier: Ch 3')
axs[3,0].set_ylabel('Bin Frequency') 
axs[3,0].set_xlabel('Amplitude')

axs[0,1].hist(ydata_none_skiplna[:, 0], bins=10)
axs[0,1].set_title('Coupler: Ch 0')
axs[0,1].set_ylabel('Bin Frequency') 
axs[0,1].set_xlabel('Amplitude')
axs[1,1].hist(ydata_none_skiplna[:, 1], bins=10)
axs[1,1].set_title('Coupler: Ch 1')
axs[1,1].set_ylabel('Bin Frequency') 
axs[1,1].set_xlabel('Amplitude')
axs[2,1].hist(ydata_none_skiplna[:, 2], bins=10)
axs[2,1].set_title('Coupler: Ch 2')
axs[2,1].set_ylabel('Bin Frequency') 
axs[2,1].set_xlabel('Amplitude')
axs[3,1].hist(ydata_none_skiplna[:, 3], bins=10)
axs[3,1].set_title('Coupler: Ch 3')
axs[3,1].set_ylabel('Bin Frequency') 
axs[3,1].set_xlabel('Amplitude')

plt.subplots

print(ydata_none[:, 0].shape)
print(ydata_none_skiplna[:, 0].shape)

#no_attenuation skip lna
fig, axs = plt.subplots(2,2)

plt.subplots_adjust(left=0.118,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.85, 
                    wspace=0.4, 
                    hspace=0.5)

fig.suptitle("Histogtam Test: No attenuation skipped lna (time domain)")
axs[0,0].hist(ydata_none_skiplna[:, 0], bins=10)
axs[0,0].set_title('Coupler: Ch 0')
axs[0,0].set_ylabel('Bin Frequency') 
axs[0,0].set_xlabel('Amplitude')
axs[0,1].hist(ydata_none_skiplna[:, 1], bins=10)
axs[0,1].set_title('Coupler: Ch 1')
axs[0,1].set_ylabel('Bin Frequency') 
axs[0,1].set_xlabel('Amplitude')
axs[1,0].hist(ydata_none_skiplna[:, 2], bins=10)
axs[1,0].set_title('Coupler: Ch 2')
axs[1,0].set_ylabel('Bin Frequency') 
axs[1,0].set_xlabel('Amplitude')
axs[1,1].hist(ydata_none_skiplna[:, 3], bins=10)
axs[1,1].set_title('Coupler: Ch 3')
axs[1,1].set_ylabel('Bin Frequency') 
axs[1,1].set_xlabel('Amplitude')

#4dB_attenuation skip lna
fig, axs = plt.subplots(2,2)

plt.subplots_adjust(left=0.118,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.85, 
                    wspace=0.4, 
                    hspace=0.5)

fig.suptitle("Histogtam Test: 4dB attenuation skipped lna (time domain)")
axs[0,0].hist(ydata_4_skiplna[:, 0], bins=10)
axs[0,0].set_title('Channel 0')
axs[0,0].set_ylabel('Bin Frequency') 
axs[0,0].set_xlabel('Amplitude')
axs[0,1].hist(ydata_4_skiplna[:, 1], bins=10)
axs[0,1].set_title('Channel 1')
axs[0,1].set_ylabel('Bin Frequency') 
axs[0,1].set_xlabel('Amplitude')
axs[1,0].hist(ydata_4_skiplna[:, 2], bins=10)
axs[1,0].set_title('Channel 2')
axs[1,0].set_ylabel('Bin Frequency') 
axs[1,0].set_xlabel('Amplitude')
axs[1,1].hist(ydata_4_skiplna[:, 3], bins=10)
axs[1,1].set_title('Channel 3')
axs[1,1].set_ylabel('Bin Frequency') 
axs[1,1].set_xlabel('Amplitude')

#4_8dB_attenuation skip lna
fig, axs = plt.subplots(2,2)

plt.subplots_adjust(left=0.118,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.85, 
                    wspace=0.4, 
                    hspace=0.5)

fig.suptitle("Histogtam Test: 4 and 8dB attenuation skipped lna (time domain)")
axs[0,0].hist(ydata_4_8_skiplna[:, 0], bins=10)
axs[0,0].set_title('Channel 0')
axs[0,0].set_ylabel('Bin Frequency') 
axs[0,0].set_xlabel('Amplitude')
axs[0,1].hist(ydata_4_8_skiplna[:, 1], bins=10)
axs[0,1].set_title('Channel 1')
axs[0,1].set_ylabel('Bin Frequency') 
axs[0,1].set_xlabel('Amplitude')
axs[1,0].hist(ydata_4_8_skiplna[:, 2], bins=10)
axs[1,0].set_title('Channel 2')
axs[1,0].set_ylabel('Bin Frequency') 
axs[1,0].set_xlabel('Amplitude')
axs[1,1].hist(ydata_4_8_skiplna[:, 3], bins=10)
axs[1,1].set_title('Channel 3')
axs[1,1].set_ylabel('Bin Frequency') 
axs[1,1].set_xlabel('Amplitude')





plt.show()
