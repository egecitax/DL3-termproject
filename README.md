📘 YZM304 III. Ödev – Duygu Analizi için RNN ile Sınıflandırma
👤 Öğrenci: Ege Çıtak
🎓 Bölüm: Ankara Üniversitesi – Yapay Zeka ve Veri Mühendisliği
📅 Dönem: 2024 – 2025 Bahar
🟦 Giriş
Bu projede, tekrarlayan sinir ağları (RNN) kullanılarak kısa cümlelerdeki duyguların pozitif veya negatif olarak sınıflandırılması amaçlanmıştır. Duygu analizi, doğal dil işleme alanında önemli bir problem olup, özellikle kullanıcı geri bildirimleri ve sosyal medya yorumlarının otomatik analizi gibi alanlarda yaygın olarak kullanılmaktadır.

🟨 Yöntem
İki farklı RNN modeli kullanılmıştır:

1. Elle Yazılmış RNN (Numpy)
Girdi vektörleri one-hot olarak encode edilmiştir.

RNN modeli forward, backward ve train fonksiyonları ile sıfırdan manuel olarak oluşturulmuştur.

Aktivasyon fonksiyonları olarak tanh ve sigmoid kullanılmıştır.

Kayıp fonksiyonu olarak Binary Cross-Entropy uygulanmıştır.

2. PyTorch ile RNN
nn.RNN modülü kullanılarak TorchRNN sınıfı tanımlanmıştır.

Eğitim fonksiyonu train_torch_rnn ile yapılmıştır.

Girdi vektörleri PyTorch tensörlerine dönüştürülerek modele verilmiştir.

Çıkış sigmoid aktivasyonu sonrası 0/1 sınıflandırmasına dönüştürülmüştür.

Her iki model de aynı eğitim ve test veri setleri ile eğitilip test edilmiştir.

🟩 Sonuçlar
🔹 Elle Yazılmış RNN:
Eğitim süreci boyunca loss ve accuracy değerleri gösterilmiş, matplotlib ile loss grafiği çizdirilmiştir.

Test verisi üzerinde başarı oranı konsola yazdırılmıştır.

🔹 PyTorch RNN:
Eğitimde her epoch için loss ve doğruluk konsola yazdırılmıştır.

Model başarılı şekilde çalışmış ve örnek test tahminleri alınmıştır.

📊 Eğitim Kaybı Grafiği:
Matplotlib kullanılarak Numpy tabanlı modelin loss eğrisi çizdirilmiş, modelin zamanla nasıl öğrendiği görselleştirilmiştir.

🟪 Tartışma
Elle yazılmış model, RNN’in temel çalışma prensibini anlamak için faydalı olmuştur. Ancak PyTorch modeli, daha ölçeklenebilir ve pratik bir çözüm sunmaktadır.

PyTorch ile geliştirilen modelin eğitim süreci daha hızlı ve daha az hata eğilimlidir.

İki modelin doğruluk oranları karşılaştırıldığında benzer performans elde edilmiştir.

Girdi cümlelerinin uzunluğu değiştiği için tensor yapılarında dikkatli dönüşüm gerekmiştir.

📚 Referanslar
https://github.com/vzhou842/rnn-from-scratch

PyTorch Dokümantasyonu: https://pytorch.org/docs/stable/index.html

Derin Öğrenme Ders Notları – YZM304

