## EX-04 Name: NAVEEN K (212223060184)
# ASK & FSK
# Aim
Write a simple Python program for the modulation and demodulation of ASK and FSK.
# Tools required
VS Code
# ASK Program
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Butterworth low-pass filter for demodulation
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# Parameters
fs = 1000
f_carrier = 50
bit_rate = 10
T = 1  # Duration of signal in seconds

t = np.linspace(0, T, int(fs * T), endpoint=False)

# Message signal (binary data)
bits = np.random.randint(0, 2, bit_rate)
bit_duration = fs // bit_rate
message_signal = np.repeat(bits, bit_duration)

# Carrier signal
carrier = np.sin(2 * np.pi * f_carrier * t)

# ASK Modulation
ask_signal = message_signal * carrier

# ASK Demodulation
demodulated = ask_signal * carrier
filtered_signal = butter_lowpass_filter(demodulated, f_carrier, fs)
decoded_bits = (filtered_signal[::bit_duration] > 0.25).astype(int)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Message Signal (Binary)', color='b')
plt.title('Message Signal')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, carrier, label='Carrier Signal', color='g')
plt.title('Carrier Signal')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, ask_signal, label='ASK Modulated Signal', color='r')
plt.title('ASK Modulated Signal')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.step(np.arange(len(decoded_bits)), decoded_bits, label='Decoded Bits', color='r', marker='x')
plt.title('Decoded Bits')
plt.grid(True)

plt.tight_layout()
plt.show()
```
# Output Waveform for ASK
<img width="1200" height="645" alt="ask" src="https://github.com/user-attachments/assets/c764d68a-522f-4f9b-8431-748b4245dc94" />

# FSK Program
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Butterworth low-pass filter for demodulation
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# Parameters
fs = 1000       # Sampling frequency
f1 = 30         # Frequency for bit = 0
f2 = 70         # Frequency for bit = 1
bit_rate = 10   # Bits per second
T = 1           # Duration in seconds

t = np.linspace(0, T, int(fs * T), endpoint=False)

# Generate random binary bits
bits = np.random.randint(0, 2, bit_rate)
bit_duration = fs // bit_rate
message_signal = np.repeat(bits, bit_duration)

# Carrier signals for bit 0 and bit 1
carrier_f1 = np.sin(2 * np.pi * f1 * t)
carrier_f2 = np.sin(2 * np.pi * f2 * t)

# FSK Modulation
fsk_signal = np.zeros_like(t)
for i, bit in enumerate(bits):
    start = i * bit_duration
    end = start + bit_duration
    freq = f2 if bit else f1
    fsk_signal[start:end] = np.sin(2 * np.pi * freq * t[start:end])

# FSK Demodulation (Coherent)
ref_f1 = np.sin(2 * np.pi * f1 * t)
ref_f2 = np.sin(2 * np.pi * f2 * t)

# Multiply with reference and apply low-pass filter
corr_f1 = butter_lowpass_filter(fsk_signal * ref_f1, f2, fs)
corr_f2 = butter_lowpass_filter(fsk_signal * ref_f2, f2, fs)

# Energy detection for each bit period
decoded_bits = []
for i in range(bit_rate):
    start = i * bit_duration
    end = start + bit_duration
    energy_f1 = np.sum(corr_f1[start:end] ** 2)
    energy_f2 = np.sum(corr_f2[start:end] ** 2)
    decoded_bits.append(1 if energy_f2 > energy_f1 else 0)

decoded_bits = np.array(decoded_bits)
demodulated_signal = np.repeat(decoded_bits, bit_duration)

# Plotting
plt.figure(figsize=(12, 12))

plt.subplot(6, 1, 1)
plt.plot(t, message_signal, color='b')
plt.title('Message Signal')
plt.grid(True)

plt.subplot(6, 1, 2)
plt.plot(t, carrier_f1, color='g')
plt.title('Carrier Signal for bit = 0 (f1)')
plt.grid(True)

plt.subplot(6, 1, 3)
plt.plot(t, carrier_f2, color='r')
plt.title('Carrier Signal for bit = 1 (f2)')
plt.grid(True)

plt.subplot(6, 1, 4)
plt.plot(t, fsk_signal, color='m')
plt.title('FSK Modulated Signal')
plt.grid(True)

plt.subplot(6, 1, 5)
plt.plot(t, demodulated_signal, color='k')
plt.title('Demodulated Signal')
plt.grid(True)

plt.tight_layout()
plt.show()
```

# Output Waveform for FSK
<img width="1536" height="754" alt="fsk" src="https://github.com/user-attachments/assets/e76c4079-73ed-4f23-95c8-efaafee972fb" />

# Results
 The experiment of modulation and demodulation of ASK and FSK was successfully executed.
