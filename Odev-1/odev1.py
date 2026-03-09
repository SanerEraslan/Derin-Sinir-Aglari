import numpy as np
import pickle
import os

# Dosyalari arama kismi
def dosya_bul(dosya_adi, arama_dizini):
    for root, dirs, files in os.walk(arama_dizini):
        if dosya_adi in files:
            return os.path.join(root, dosya_adi)
    return None

data_klasoru = os.path.join("Odev-1", "data")
egitim_yolu = dosya_bul("data_batch_1", data_klasoru)
test_yolu = dosya_bul("test_batch", data_klasoru)

if not egitim_yolu or not test_yolu:
    print("\nHATA: Dosyalar bulunamadi!")
    print("Lutfen Odev-1/data klasorunu kontrol edin.")
    exit()

print(f"--- Dosyalar Bulundu: {egitim_yolu} ---")

with open(egitim_yolu, 'rb') as f:
    d = pickle.load(f, encoding='bytes')
    X_train, y_train = d[b'data'].astype("float"), np.array(d[b'labels'])

with open(test_yolu, 'rb') as f:
    d = pickle.load(f, encoding='bytes')
    X_test, y_test = d[b'data'].astype("float"), np.array(d[b'labels'])

print("\n" + "="*30 + "\nCIFAR-10 k-NN TEST\n" + "="*30)
secim = input("Mesafe Olcutu (L1/L2): ").upper()
k_val = int(input("k degeri: "))

for i in range(5):
    # Uzaklik hesaplama
    if secim == "L1":
        dist = np.sum(np.abs(X_train - X_test[i]), axis=1)
    else:
        dist = np.sqrt(np.sum(np.square(X_train - X_test[i]), axis=1))
    
    # En yakin k komsuyu bulma
    yakinlar = np.argsort(dist)[:k_val]
    etiketler = y_train[yakinlar]
    tahmin = np.bincount(etiketler).argmax()
    
    print(f"Ornek {i+1}: Tahmin: {tahmin}, Gercek: {y_test[i]}")