# Pendahuluan

Berbagai macam metode klasifikasi video *DeepFake* dengan video asli telah di-implementasikan ke berbagai macam dataset. Kami mencoba merumuskan metode klasifikasi video *DeepFake* dan asli yang lebih efisien tanpa mengorbankan performa.

# Landasan Teori

Kami menggunakan *Local Binary Pattern* (LBP) untuk meringkankan kinerja CNN karena LBP sudah melakukan ekstraksi fitur di langkah *prepocessing*, sehingga CNN hanya tinggal fokus pada fitur yang *high level*. Spatiotemporal Attention kami pakai untuk meningkatkan pengamatan model pada fitur-fitur yang telah terekstrak. Conv-LSTM meningkatkan performa model dalam memahami pola dari frame 1 ke frame selanjutnya.

# Metodologi Riset

## Dataset

Berikut beberapa data yang dipakai:
1. FaceForensics++ [Original perlu izin dulu](https://github.com/ondyari/FaceForensics), [Versi Kaggle](https://www.kaggle.com/datasets/xdxd003/ff-c23)
2. DFDC Dataset [Original](https://ai.meta.com/datasets/dfdc/), [Varsi Kaggle](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)
3. UADFV [Kaggle](https://www.kaggle.com/datasets/adityakeshri9234/uadfv-dataset)
4. Celeb DF (v2) [Kaggle](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2)

Download FF++:
curl -L -o ./datasets/ff-c23.zip\
  https://www.kaggle.com/api/v1/datasets/download/xdxd003/ff-c23

## Algoritma

### Preprocessing

Beberapa metode yang digunakan untuk *preprocessing*:
1. MTCNN = Deteksi Wajah
2. LBP = Ekstrak Tekstur
3. Gaussian Filter = Mengurangi Noise

### Feature Extraction

1. Xception
2. Spatiotemporal Attention
3. Conv-LSTM

# Daftar Pustaka

<a id="ref1"/>

[1] [A. Arini, Rizal Broer Bahaweres, and Javid Al Haq, “Quick Classification of Xception And Resnet-50 Models on Deepfake Video Using Local Binary Pattern,” Jan. 2022, doi: https://doi.org/10.1109/ismode53584.2022.9742852](https://ieeexplore.ieee.org/document/9742852)

<a id="ref2"/>

[2] [A. Das, K.S Angel Viji, and L. Sebastian, “A Survey on Deepfake Video Detection Techniques Using Deep Learning,” Jul. 2022, doi: https://doi.org/10.1109/icngis54955.2022.10079802.](https://ieeexplore.ieee.org/document/10079802)

<a id="ref3"/>

[3] [A. K. Jha, A. K. Yadav, A. K. Dubey, A. Kumar, and A. Sharma, “Deep Learning Based Deepfake Video Detection System,” 2025 3rd International Conference on Disruptive Technologies (ICDT), pp. 408–412, Mar. 2025, doi: https://doi.org/10.1109/icdt63985.2025.10986738.](https://ieeexplore.ieee.org/document/10986738)

<a id="ref4"/>

[4] [I. Petrov Freelancer et al., “DeepFaceLab: Integrated, flexible and extensible face-swapping framework.” Available: https://arxiv.org/pdf/2005.05535](https://arxiv.org/pdf/2005.05535)

<a id="ref5"/>

[5] [N. M. Emara and Mazen Nabil Elagamy, “DeepStream-X: A Two-Stream Deepfake Detection Framework using Spatiotemporal and Frequency Features,” pp. 290–296, Dec. 2024, doi: https://doi.org/10.1109/iccta64612.2024.10974904.](https://ieeexplore.ieee.org/document/10974904)

<a id="ref6">

[6] [R. Khan et al., “Comparative study of deep learning techniques for DeepFake video detection,” ICT Express, vol. 10, no. 6, Oct. 2024, doi: https://doi.org/10.1016/j.icte.2024.09.018.](https://www.sciencedirect.com/science/article/pii/S2405959524001218?via%3Dihub)

<a id="ref7">
  
[7] [S. Graphics, “Benchmark Results - FaceForensics Benchmark,” kaldir.vc.in.tum.de. https://kaldir.vc.in.tum.de/faceforensics_benchmark/](https://kaldir.vc.in.tum.de/faceforensics_benchmark/)

<a id="ref8">
  
[8] [S. Serengil and A. Özpınar, “A Benchmark of Facial Recognition Pipelines and Co-Usability Performances of Modules,” Bilişim Teknolojileri Dergisi, vol. 17, no. 2, pp. 95–107, Apr. 2024, doi: https://doi.org/10.17671/gazibtd.1399077.](https://dergipark.org.tr/tr/download/article-file/3573195)

<a id="ref9">
  
[9] [W. M. Wubet*, “The deepfake challenges and deepfake video detection,” International Journal of Innovative Technology and Exploring Engineering, vol. 9, no. 6, pp. 789–796, Apr. 2020. doi:10.35940/ijitee.e2779.049620 ](https://www.ijitee.org/portfolio-item/E2779039520/)

<a id="ref10">
  
[10] [Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, “Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics,” arxiv.org, Sep. 2019, Available: https://arxiv.org/abs/1909.12962](https://arxiv.org/pdf/1909.12962)
