# ta-clustering
Repository ini berisi program Tugas Akhir yang berjudul **Pengelompokan Kebutuhan Upgrade Bandwidth Pelanggan Broadband Internet Menggunakan Algoritma K-Means**.

Program ini bertujuan untuk mengelompokan data pelanggan internet wireless broadband dengan lima atribut utama menggunakan algoritma K-Means yang kemudian dioptimasi menggunakan metrik evaluasi Davies-Bouldin Index. Dataset yang digunakan pada project ini dinormalisasi menggunakan metode MinMax Normalization dan sistem akan mengelompokkan data yang belum dan telah dinormalisasi untuk melihat perbedaan kualitas pengelompokan yang dilakukan.

Berikut ini beberapa Library Python yang digunakan dalam program.
- streamlit
- numpy
- pandas
- matplotlib
- seaborn
- sklearn

Alur pengerjaan sistem dalam project ini adalah sebagai berikut
- Data inputan user akan dinormalisasi menggunakan metode MinMax Normalization.
- Sistem akan memberikan preview data yang belum dan telah dinormalisasi.
- Kedua data akan dikelompokkan menggunakan algoritmna K-Means, dan dioptimasi menggunakan metrik evaluasi DBI.
- Sistem menampilkan hasil pengelompokan berupa tabel yang disesuaikan dengan jumlah cluster optimal dari perhitungan DBI.
- Sistem menampilkan visualisasi hasil pengelompokan dalam bentuk pairplot dan scatterplot.

Tahapan yang perlu dilakukan dalam menggunakan aplikasi ini adalah sebagai berikut.
- Upload dataset yang telah disediakan pada repository kedalam field upload.
- Hasil pengelompokan berupa tabel dan visualisasi dapat diakses melalui tab pengelompokan dan visualisasi hasil pengelompokan.

**PENTING**
Pada line 59 di main.py terdapat ResearchSettings dimana user dapat mengatur range jumlah cluster yang akan dihitung nilai DBI-nya. User juga dapat menggunakan seed penetapan centroid secara random agar hasil pengelompokan yang didapatkan dapat berubah setiap kali sistem digunakan.
