import numpy as np
import pickle
import os

# --- ADIM 1: VERİ SETİNİ YEREL DİZİNDEN YÜKLEME ---
# Ödev kuralı: Veri internetten çekilmeyecek, klasörden okunacak.
# Dosyaların 'Odev-1/data/' klasörü altında olduğunu varsayıyoruz.
egitim_batch_yolu = os.path.join("Odev-1", "data", "data_batch_1")
test_batch_yolu = os.path.join("Odev-1", "data", "test_batch")

# Eğitim verilerini yükle
with open(egitim_batch_yolu, 'rb') as f:
    egitim_dict = pickle.load(f, encoding='bytes')
    X_train = egitim_dict[b'data'].astype("float")
    y_train = np.array(egitim_dict[b'labels'])

# Test verilerini yükle
with open(test_batch_yolu, 'rb') as f:
    test_dict = pickle.load(f, encoding='bytes')
    X_test = test_dict[b'data'].astype("float")
    y_test = np.array(test_dict[b'labels'])

# --- ADIM 2: KULLANICI ETKİLEŞİMİ ---
print("-" * 40)
print("CIFAR-10 k-NN SINIFLANDIRICI (Ödev-1)")
print("-" * 40)

# Mesafe seçimi
mesafe_secimi = input("Mesafe ölçütü seçiniz (L1 veya L2): ").strip().upper()

# k değeri seçimi
k = int(input("k komşu sayısını giriniz (Örn: 1, 3, 5): "))

# Performans için ilk 10 test örneği üzerinden işlem yapalım
num_test = 10
tahminler = []

print(
    f"\n{num_test} adet test örneği üzerinde {mesafe_secimi} mesafesi kullanılarak k={k} ile sınıflandırma yapılıyor...\n")

# --- ADIM 3: k-NN ALGORİTMASI (DÜZ AKIŞ) ---
# Düz akış kuralı gereği fonksiyon (def) kullanılmadan işlemler döngü içinde yapılır.
for i in range(num_test):
    test_gorseli = X_test[i]

    # 1. Uzaklık Hesaplama
    if mesafe_secimi == "L1":
        # L1 (Manhattan) Mesafesi: Toplam |x1 - x2|
        uzakliklar = np.sum(np.abs(X_train - test_gorseli), axis=1)
    else:
        # L2 (Öklid) Mesafesi: Kök(Toplam (x1 - x2)^2)
        uzakliklar = np.sqrt(np.sum(np.square(X_train - test_gorseli), axis=1))

    # 2. En yakın k komşuyu bulma
    # Uzaklıkları küçükten büyüğe sıralayıp ilk k indeksi alıyoruz
    en_yakin_indeksler = np.argsort(uzakliklar)[:k]
    en_yakin_etiketler = y_train[en_yakin_indeksler]

    # 3. Oylama Yapma (En çok tekrar eden sınıfı bulma)
    tahmin = np.bincount(en_yakin_etiketler).argmax()
    tahminler.append(tahmin)

    # Sonucu ekrana yazdır
    gercek_sinif = y_test[i]
    durum = "BAŞARILI" if tahmin == gercek_sinif else "BAŞARISIZ"
    print(f"Test Örneği {i + 1}: Tahmin={tahmin}, Gerçek={gercek_sinif} -> {durum}")

# --- ADIM 4: SONUÇLARI HESAPLA ---
dogruluk = (np.sum(np.array(tahminler) == y_test[:num_test]) / num_test) * 100
print("-" * 40)
print(f"Test Sonucu Doğruluk Oranı: %{dogruluk}")
print("-" * 40)