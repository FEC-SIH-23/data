#This code is focused to generate a BCH encoded string dataset in csv 
#format to train RNN model for PSID SIH1447. This code utilizes Libraries
#galois for BCH encoding. It generates a random data encodes it using BCH#modulates it using BPSK modulation technique, adds White Gaussian Noise #and then pads the signal to desired length and writes the signal as samp#les in csv file.
# use [$ pip install numpy galois] to install the required libraries.


# Importing libraries required, remove the comment on the next line to install on Google Colab and Kaggle..
!pip install galois numpy

import galois
import numpy as np
import csv

#Generates binary string of length "n"..
def generate_random_data(n):
    return galois.GF2.Random(n)

#Using Galois library, this function encodes the message generated by message() 
#and since we are applying 16QAM we need the message to be multiple of 4..
def encode_bch(data, n, k):
    bch = galois.BCH(n, k)
    encoded_data = bch.encode(data)
    return encoded_data[:k] 

#function to implement a 16QAM Modulator...
def generate_symbols_16qam():
    symbols_16qam = np.array([
        complex(-3, 3), complex(-1, 3), complex(1, 3), complex(3, 3),
        complex(-3, 1), complex(-1, 1), complex(1, 1), complex(3, 1),
        complex(-3, -1), complex(-1, -1), complex(1, -1), complex(3, -1),
        complex(-3, -3), complex(-1, -3), complex(1, -3), complex(3, -3),
    ])
    return symbols_16qam

def modulate_to_bits_16qam(bits, symbols):
    num_symbols = len(bits) // 4
    symbol_indices = [int(bits[i:i+4], 2) for i in range(0, len(bits), 4)]
    modulated_symbols = symbols[symbol_indices]

    modulated_bits = ''.join(format(int(np.real(symbol)), '04b') + format(int(np.imag(symbol)), '04b') for symbol in modulated_symbols)
    modulated_bits = modulated_bits.replace('-', '')

    return modulated_bits

#Adds AWGN on the modulated signal, simulating real world channels..
def add_awgn(signal, snr_dB):
    noise_std_dev = 10 ** (-snr_dB / 20)
    awgn = np.random.normal(0, noise_std_dev, len(signal))
    noisy_signal = np.where(np.array(signal) + awgn > 0, 1, 0)
    return noisy_signal.tolist()

#Pads signal by adding Zeroes to the desired length..
def pad_signal(signal, length):
    return signal + [0] * (length - len(signal))

#Generating samples..
def generate_samples_16qam(n_qam, k_qam, num_samples, snr_dB, pad_length, n_bch, k_bch):
    samples = []
    qam_symbols = generate_symbols_16qam()

    for _ in range(num_samples):
        data = generate_random_data(k_bch)
        encoded_data = encode_bch(data, n_bch, k_bch)
        modulated_data = modulate_to_bits_16qam(''.join(map(str, encoded_data)), qam_symbols)
        noisy_data = add_awgn([int(bit) for bit in modulated_data], snr_dB)
        padded_data = pad_signal(noisy_data, pad_length)

        samples.append(padded_data)

    return samples

#Writes CSV file with two columns, one for samples and one for encoding...
def save_to_csv(data, filename, encoding):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["bch16qam_encoded_string", "encoding"])
        for line in data:
            line_str = ''.join(map(str, line))
            writer.writerow([line_str, encoding])

if __name__ == "__main__":
    n_qam = 12
    k_qam = 8
    n_bch = 15
    k_bch = 7

    num_samples = int(input("Enter the Number of Samples:- "))
    snr_dB = float(input("Enter the SNR in dB:- "))
    pad_length = int(input("Enter the Pad Length:- "))
    encoding = int(input("Enter the Encoding Parameter(for CSV):- "))

    samples = generate_samples_16qam(n_qam, k_qam, num_samples, snr_dB, pad_length, n_bch, k_bch)
    save_to_csv(samples, f'BCH{n_qam}{k_qam}_16QAM_SNR{snr_dB}_BL{pad_length}.csv', encoding)

    print(f"BCH{n_qam}{k_qam}_16QAM_SNR{snr_dB}_BL{pad_length}.csv created, download it now!")
    print(f"The current BCH coding used is BCH({n_qam}, {k_qam})")