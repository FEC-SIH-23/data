#This code is focused to generate a BCH encoded string dataset in csv 
#format to train RNN model for PSID SIH1447. This code utilizes Libraries
#galois for BCH encoding. It generates a random data encodes it using BCH#modulates it using BPSK modulation technique, adds White Gaussian Noise #and then pads the signal to desired length and writes the signal as samp#les in csv file.
# use [$ pip install numpy galois] to install the required libraries.
import galois
import numpy as np
import csv

def generate_random_data(n):
    return galois.GF2.Random(n)

def encode_bch(data, n, k):
    bch = galois.BCH(n, k)
    return bch.encode(data)

def bpsk_modulate(data):
    return [-1 if bit == 0 else 1 for bit in data]

def add_awgn(signal, snr_dB):
    noise_std_dev = 10 ** (-snr_dB / 20)
    awgn = np.random.normal(0, noise_std_dev, len(signal))
    noisy_signal = np.where(np.array(signal) + awgn > 0, 1, 0)
    return noisy_signal.tolist()

def pad_signal(signal, length):
    return signal + [0] * (length - len(signal))

def generate_samples(n, k, num_samples, snr_dB, pad_length):
    samples = []
    for _ in range(num_samples):
        # Generate random binary data
        data = generate_random_data(k)

        # Encode using BCH
        encoded_data = encode_bch(data, n, k)

        # Modulate using BPSK
        modulated_data = bpsk_modulate(encoded_data)

        # Add AWGN
        noisy_data = add_awgn(modulated_data, snr_dB)

        # Pad the signal
        padded_data = pad_signal(noisy_data, pad_length)

        samples.append(padded_data)

    return samples

def save_to_csv(data, filename, encoding):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["padded_AWGN/BPSK/BCH_data_string", "encoding"])
        for line in data:
            line_str = ''.join(map(str, line))
            writer.writerow([line_str, encoding])

            
            
if __name__ == "__main__":
    n = 15
    k = 11
    num_samples = 100
    snr_dB = 2
    pad_length = 1024
    encoding = 4

    
    samples = generate_samples(n, k, num_samples, snr_dB, pad_length)
    save_to_csv(samples, 'bch_dataset(-2SNR, BL2048).csv', encoding)


    print("CSV created, download it now!")
    print("The current BCH coding used is BCH({}, {})".format(n, k))

