import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from preprocessing import LoadData, DataPreprocessing
from model import Model


class Main:
    @staticmethod
    def main():
        # # Membuat dua kolom, satu untuk gambar dan yang lainnya untuk judul
        # col1, col2 = st.columns([1, 4])
        #
        # with col1:
        #     st.image("logo.png", width=100)
        #
        # with col2:
        st.markdown("""
        # Pengelompokan Kebutuhan Upgrade Bandwidth Pelanggan Broadband Internet Menggunakan Algoritma K-Means
        """)

        st.write(
            """<style>
            [data-testid="stHorizontalBlock"] {
                align-items: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # File Upload
        fileUpload = st.file_uploader("Unggah File Excel", type=["xlsx"])

        if fileUpload is not None:
            tab1, tab2, tab3 = st.tabs(
                ["Data Preview", "Pengelompokan", "Visualisasi Hasil Pengelompokan"])
            data = LoadData.read_data(fileUpload)

            with tab1:
                st.write("Data yang belum dinormalisasi")
                data_preview = pd.DataFrame(data, columns=['Jumlah Keluhan', 'Paket Berlangganan', 'Lama Berlangganan',
                                                           'Instalasi', 'PoP'])
                st.write(data_preview)

            # Normalization
            normalizedData = DataPreprocessing.min_max_norm(data)
            df_norm = pd.DataFrame(normalizedData,
                                   columns=['Jumlah Keluhan', 'Paket Berlangganan', 'Lama Berlangganan', 'Instalasi',
                                            'PoP'])
            with tab1:
                st.write("Data yang telah dinormalisasi")
                st.write(df_norm)

            # Research Settings
            k_range = range(2, 8)
            dbiVal = Model.kmeans(data, k_range)
            optimalClusterIndex = np.argmin(dbiVal)
            optimalClusterNum = k_range[optimalClusterIndex]

            norm_dbiVal = Model.kmeans(normalizedData, k_range)
            norm_optimalClusterInd = np.argmin(norm_dbiVal)
            norm_optimalClusterNum = k_range[norm_optimalClusterInd]

            # Optimized Clustering
            # random_state = None  # seed
            random_state = 777  # seed
            centroids = Model.initializeCentroids(data, optimalClusterNum, random_state=random_state)

            for _ in range(100):  # jumlah iterasi
                clusters = Model.assignClusters(data, centroids)
                new_centroids = Model.updateCentroids(data, clusters, optimalClusterNum)
                if np.all(centroids == new_centroids):
                    break
                centroids = new_centroids

            norm_centroids = Model.initializeCentroids(normalizedData, norm_optimalClusterNum,
                                                       random_state=random_state)
            for _ in range(100):  # jumlah iterasi
                norm_clusters = Model.assignClusters(normalizedData, norm_centroids)
                norm_new_centroids = Model.updateCentroids(normalizedData, norm_clusters, norm_optimalClusterNum)
                if np.all(norm_centroids == norm_new_centroids):
                    break
                norm_centroids = norm_new_centroids

            # Print Results
            with tab2:
                with st.expander("Hasil Clustering data yang belum dinormalisasi"):
                    tabelClustering = []
                    for i in range(optimalClusterNum):
                        cluster_data = data[clusters == i]
                        dataNum = len(cluster_data)
                        tabelClustering.append(
                            {'Cluster': i + 1, 'Number of Data': dataNum, 'Centroid': np.round(centroids[i], 2)})

                        # Convert cluster data to DataFrame
                        dfCluster = pd.DataFrame(cluster_data,
                                                 columns=['Jumlah Keluhan', 'Paket Berlangganan', 'Lama Berlangganan',
                                                          'Instalasi', 'PoP'])
                        st.write(f"Cluster {i + 1}:")
                        st.write(dfCluster)

                    st.write("Jumlah Data pada tiap Cluster")
                    df_numData = pd.DataFrame(
                        [{'Cluster': item['Cluster'], 'Jumlah Data': item['Number of Data'],
                          'Centroid': item['Centroid']} for item in tabelClustering])
                    st.write(df_numData)

                    st.write("Nilai DBI pada Setiap Cluster")
                    for k, dbi in zip(k_range, dbiVal):
                        st.write(f"Jumlah Cluster: {k}, DBI: {dbi}")
                    st.write('Jumlah Cluster Optimal: ', dbiVal[optimalClusterIndex])

                with st.expander("Hasil Clustering Data yang sudah dinormalisasi"):
                    norm_tabelClustering = []
                    for i in range(norm_optimalClusterNum):
                        normClusterData = normalizedData[norm_clusters == i]
                        norm_dataNum = len(normClusterData)
                        norm_tabelClustering.append(
                            {'Cluster': i + 1, 'Number of Data': norm_dataNum, 'Centroid': np.round(norm_centroids[i], 2)})

                        # Convert cluster data to DataFrame
                        norm_dfCluster = pd.DataFrame(normClusterData,
                                                      columns=['Jumlah Keluhan', 'Paket Berlangganan',
                                                               'Lama Berlangganan', 'Instalasi', 'PoP'])
                        st.write(f"Cluster {i + 1}:")
                        st.write(norm_dfCluster)

                    st.write("Jumlah Data pada tiap Cluster")
                    df_numData = pd.DataFrame(
                        [{'Cluster': item['Cluster'], 'Jumlah Data': item['Number of Data'],
                          'Centroid': item['Centroid']} for item in norm_tabelClustering])
                    st.write(df_numData)

                    st.write("Nilai DBI pada Setiap Cluster")
                    for k, dbi in zip(k_range, norm_dbiVal):
                        st.write(f"Jumlah Cluster: {k}, DBI: {dbi}")
                    st.write('Jumlah Cluster Optimal: ', norm_dbiVal[norm_optimalClusterInd])

            # Visualisasi (PCA)
            pca = PCA(n_components=3)

            pcaData = pca.fit_transform(data)
            norm_pcaData = pca.fit_transform(normalizedData)

            # # Calculate centroids in PCA space
            # pca_centroids = pca.transform(centroids)
            # norm_pca_centroids = pca.transform(norm_centroids)

            fig = plt.figure()
            ax1 = fig.add_subplot()
            scatter1 = ax1.scatter(pcaData[:, 0], pcaData[:, 1], c=clusters, cmap='viridis')
            # ax1.scatter(pca_centroids[:, 0], pca_centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
            ax1.set_xlabel('PCA 1')
            ax1.set_ylabel('PCA 2')
            ax1.set_title('PCA Visualization (Non-Normalized Data)')
            legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters")
            ax1.add_artist(legend1)

            norm_fig = plt.figure()
            ax2 = norm_fig.add_subplot()
            scatter2 = ax2.scatter(norm_pcaData[:, 0], norm_pcaData[:, 1], c=norm_clusters, cmap='viridis')
            # ax2.scatter(norm_pca_centroids[:, 0], norm_pca_centroids[:, 1], marker='*', s=200, c='black',
            #             label='Centroids')
            ax2.set_xlabel('PCA 1')
            ax2.set_ylabel('PCA 2')
            ax2.set_title('PCA Visualization (Normalized Data)')
            legend2 = ax2.legend(*scatter2.legend_elements(), title="Clusters")
            ax2.add_artist(legend2)

            # Plot Clusters in 3D
            fig3d = plt.figure(figsize=(8, 6))
            ax = fig3d.add_subplot(111, projection='3d')
            colors = plt.cm.viridis(np.linspace(0, 1, len(centroids)))
            attributes = ['Jumlah Keluhan', 'Paket Berlangganan', 'Lama Berlangganan', 'Instalasi', 'PoP']

            for i in range(len(centroids)):
                points = data[clusters == i]
                ax.scatter(points[:, 2], points[:, 1], points[:, 0], c=colors[i], s=50,
                           label=f'Cluster {i + 1}')
            ax.scatter(centroids[:, 2], centroids[:, 1], centroids[:, 0], marker='*', s=200, c='black',
                       label='Centroids')
            ax.set_xlabel(attributes[2])
            ax.set_ylabel(attributes[1])
            ax.set_zlabel(attributes[0])
            ax.set_title('Hasil Clustering (Data yang belum dinormalisasi)')
            ax.legend()
            ax.grid(True)

            # Plot Clusters in 3D (Normalized Data)
            norm_fig3d = plt.figure(figsize=(8, 6))
            norm_ax = norm_fig3d.add_subplot(111, projection='3d')
            norm_colors = plt.cm.viridis(np.linspace(0, 1, len(norm_centroids)))
            attributes = ['Jumlah Keluhan', 'Paket Berlangganan', 'Lama Berlangganan', 'Instalasi', 'PoP']

            for i in range(len(norm_centroids)):
                norm_points = normalizedData[norm_clusters == i]
                norm_ax.scatter(norm_points[:, 2], norm_points[:, 1], norm_points[:, 0], c=norm_colors[i], s=50,
                                label=f'Cluster {i + 1}')
            norm_ax.scatter(norm_centroids[:, 2], norm_centroids[:, 1], norm_centroids[:, 0], marker='*', s=200,
                            c='black', label='Centroids')
            norm_ax.set_xlabel(attributes[2])
            norm_ax.set_ylabel(attributes[1])
            norm_ax.set_zlabel(attributes[0])
            norm_ax.set_title('Hasil Clustering (Data yang telah dinormalisasi)')
            norm_ax.legend()
            norm_ax.grid(True)

            with tab3:
                with st.expander("Visualisasi Pairplot"):
                    st.write("Visualisasi Pairplot Hasil Clustering untuk data yang belum dinormalisasi")
                    data_preview['Cluster'] = clusters
                    pairplot_fig = sns.pairplot(data_preview, hue='Cluster')
                    st.pyplot(pairplot_fig)

                    st.write("Visualisasi Pairplot Hasil Clustering untuk data yang telah dinormalisasi")
                    df_norm['Cluster'] = norm_clusters
                    norm_pairplot_fig = sns.pairplot(df_norm, hue='Cluster')
                    st.pyplot(norm_pairplot_fig)

                with st.expander("Visualisasi Scatterplot"):
                    st.write("Visualisasi Hasil Clustering untuk data yang belum dinormalisasi")
                    st.pyplot(fig3d)

                    st.write("Visualisasi PCA Hasil Clustering data yang belum dinormalisasi")
                    st.pyplot(fig)

                    st.write("Visualisasi hasil clustering untuk data yang telah dinormalisasi")
                    st.pyplot(norm_fig3d)

                    st.write("Visualisasi PCA Hasil Clustering data yang belum dinormalisasi")
                    st.pyplot(norm_fig)

if __name__ == "__main__":
    Main.main()
