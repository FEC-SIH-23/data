import pandas as pd

files = ['bch_encoded_noise_snr2.csv', 'convolutional_encoded_noise_snr2.csv', 'ldpc_encoded_noise_2048_snr2.csv', 'turbo_encoded_noise_1784_to_2048_snr2.csv',
         'turbo_encoded_noise_678_to_2048_snr2.csv']

merged = pd.DataFrame()

print("[*] File merging started")
for f in files:
    fname = './all_data/' + f
    df = pd.read_csv(fname)
    merged = pd.concat([merged, df], ignore_index=True)
print("[+] File merging completed")

assert merged.shape[0] == 507400, print(f"[X] Total rows are {merged.shape[0]}")

print("[*] Shuffling Dataframe")

merged = merged.sample(frac=1)

print("[+] Shuffling completed. Dataset is now randomized")

print("[*] Attempting to write to CSV")

merged.to_csv('sih_fec_training_data_merged.csv', index=False)

print("[+] Dataframe written to CSV file. Success!")

