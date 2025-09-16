## Paper buat proposal nanti

[1] [A Benchmark of Facial Recognition Pipelines and Co-Usability Performances of Modules](https://dergipark.org.tr/tr/download/article-file/3573195)\
Ini menjelaskan perbandingan facedetector seperti MTCNN, RetinaFace, OpenCV DNN, dan sebagainya. RetinaFace dan MTCNN punya performa bagus, jadi kita pakai antara keduanya (paper kebanyakan pakai MTCNN, tapi kalau dilihat dari paper itu, MTCNN sama RetinaFace performanya beda tipis).

[2] [Quick Classification of Xception And Resnet-50 Models on Deepfake Video Using Local Binary Pattern](https://ieeexplore.ieee.org/document/9742852)\
Di sini aku nyoba 2 implementasi face detektor ([MTCNN](https://github.com/PutriSantikaMayangsari/Deep_Fake_Detection/blob/main/MTCNN_LBP.ipynb) dan RetinaFace), ternyata hasilnya sama bagusnya. Jadi sepertinya aku pikir mending pake MTCNN aja, karena kebanyakan paper pake itu.

[3] [Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics](https://arxiv.org/pdf/1909.12962)\
Paper ini menjelaskan dataset lain yang begitu mudah dibedakan antara fake dengan asli. Paper ini menawarkan dataset Celeb-DF yang lebih sulit untuk dibedakan. Nanti seterusnya kita pake ini saja.

## Rangkuman mengenai performa Model Klasifikasi DeepFake dan Dataset

Ini rangkuman yang coba aku kumpulkan dari paper [[1]](#ref1). Hanya untuk catatan sementara saja, nanti dihapus. Intinya ini mencoba melakukan perbandingan berbagai macam teknik untuk mendeteksi DeepFake. Ada beberapa dataset yang di-highlight oleh paper tersebut, aku akan mencoba kasih link jika datasetnya publik.\
Dataset yang dipakai dalam riset DeepFake:

1. DeepFaceLab Dataset (Dalam Paper [[2]](#ref2), DeepFaceLab pakai berbagai macam dataset. Ya dataset yang ada di bawah ini)
2. CelebA Dataset [Link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
3. FaceForensics++ [Original perlu izin dulu](https://github.com/ondyari/FaceForensics), [Versi Kaggle](https://www.kaggle.com/datasets/xdxd003/ff-c23)
4. DFDC Dataset [Original](https://ai.meta.com/datasets/dfdc/), [Varsi Kaggle](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)
5. UADFV [Kaggle](https://www.kaggle.com/datasets/adityakeshri9234/uadfv-dataset)
6. Celeb DF (v2) [Kaggle](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2)
7. Wild Deep Fake [Kaggle](https://www.kaggle.com/datasets/maysuni/wild-deepfake)

### Model
Kalian bisa load model hasil dari training di atas, [Link](https://drive.google.com/file/d/1OdpPbkIf0EJCV3_EOHbH6V3Y0WtdagIW/view?usp=sharing), kemudian lanjut training pake data baru yang lebih susah. Aku menyarankan pakai Celeb DF (v2) atau DFDC. Sepertinya FaceForensics++ terlalu gampang.

Perbandingan Metode:
| Study | Features | Classification Methods | Databases | Best Performance |
|---------------------------------|----------------------------|-------------------------|--------------------------------------------------------------------------------|--------------------------|
| Convolutional LSTM | Image Temporal Information | CNN + RNN | own | Acc. ≈ 97.1 |
| XceptionNet | Image-related Steganalysis | CNN | FF++ (DeepFake, LQ)<br>FF++ (DeepFake, HQ)<br>FF++ (DeepFake, RAW)<br>Celeb-DF | Acc. ≈ 94.0%<br>Acc. ≈ 98.0%<br>Acc. ≈ 100.0%<br>Acc. ≈ 65.5% |
| MesoNet | Mesoscopic Level | CNN | FF++ (DeepFake, LQ)<br>FF++ (DeepFake, HQ)<br>FF++ (DeepFake, RAW)<br>Celeb-DF | Acc. ≈ 90.0%<br>Acc. ≈ 94.4%<br>Acc. ≈ 98.06%<br>Acc. ≈ 54.8% |
| Capsule | Image-related | Capsule Network | DeepFakeTIMIT (LQ)<br>DeepFakeTIMIT (HQ)<br>FF++ (DeepFake)<br>Celeb-DF | AUC ≈ 78.4%<br>AUC ≈ 74.4%<br>AUC ≈ 92.17%<br>AUC ≈ 57.5% |
| Face Warping Artifacts | Image Warping Artifacts | CNN | UADFV<br>DeepFakeTIMIT (LQ)<br>DeepFakeTIMIT (HQ) | AUC ≈ 97.4%<br>AUC ≈ 99.9%<br>AUC ≈ 93.2% |
| Two Stream | Image related Stageanalysis| CNN, SVM | UADFV<br>DeepFakeTIMIT (LQ)<br>DeepFakeTIMIT (HQ) | AUC ≈ 85.1%<br>AUC ≈ 83.5%<br>AUC ≈ 73.5% |
| Recurrent Convolutional Network | Image+Temporal Information | CNN + RNN | FF++ (DeepFake, LQ) | Acc. ≈ 96.9% |
| Visual Artifacts | Visual Artifacts | Logistic Regression, MLP| own | AUC ≈ 85.1% |
| Maachine Learning | Spatial features + Temporal Inconsistency | ResNeXt50 + LSTM| Youtube, FaceForensic++, Kaggle | Acc 93.4% Precision 91.8% Recall 94.2% F1-Score 93.0% AUC-ROC 0.96 Average Confidence 92% |

Coba cek 3 akurasi tertinggi:

1. XceptionNet: FF++(DeepFake,RAW), 100%, [Paper XceptionNet](https://arxiv.org/pdf/1610.02357), [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf), [GitHub](https://github.com/ondyari/FaceForensics)
2. Face Warping Artifacts: DeepFakeTIMIT(LQ), 99.9% [Paper](https://arxiv.org/pdf/1811.00656), [GitHub](https://github.com/yuezunli/CVPRW2019_Face_Artifacts)
3. MesoNet: FF++(DeepFake,RAW), 98%, [Paper](https://hal.science/hal-01867298/file/afchar_WIFS_2018.pdf), [GitHub](https://github.com/DariusAf/MesoNet)

Mungkin Benchmark ini juga bisa membantu [[3]](#ref3).

### Reference Paper

<a id="ref1"/>

[1] [Comparative study of deep learning techniques for DeepFake video detection](https://www.sciencedirect.com/science/article/pii/S2405959524001218?via%3Dihub)

<a id="ref2"/>

[2] [DeepFaceLab: Integrated, flexible and extensible face-swapping framework](https://arxiv.org/pdf/2005.05535)

<a if="ref3"/>

[3] [FaceForensics Benchmark](https://kaldir.vc.in.tum.de/faceforensics_benchmark/)

---

## Insight dari paper " DeepFake Video Detection Using Machine Learning" by Dumbre et al. (link paper di googlesheet grup)

Konsep yang dipakai disini menarik, (mungkin bisa kita coba adopsi untuk purpose metode baru)

Ide dasarnya adalah bagaimana mengekstrak fitur-fitur spasial penting (termasuk inkonsistensi dalam frame video --> indikasi deepfake) kemudian mengevaluasi sequence atau frame yang tidak konsisten.

- ide: fitur spasial + temporal inconsistency
- Metode yang dipakai adalah gabungan CNN dan LSTM
- Ekstraksi fitur dengan ResNeXt50 kemudian evaluasi inkonsistensi dengan LSTM
- Preprocessing Multi-task Cascaded Convolutional Network(MTCNN)

Dataset yang dipakai:

- YouTube (for authentic content)
- FaceForensics++
- Kaggle's DeepFake Detection Chalenge dataset

Metric Score:

- Accuracy 93.4%
- Precision 91.8%
- Recall 94.2%
- F1-Score 93.0%
- AUC-ROC 0.96
- Average Confidence 92%
