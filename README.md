# Comparison of methods for Anomaly Detection in Intraday Individual Stock Prices

This project is an attempt to compare the effectiveness of various methods of Anomaly Detection for intraday individual stock prices. Methods contrasted include statistical such as ARIMA and Histogram; Proximity, Clustering, and angle-based methods such as k-NN Distance, Local Outlier Factor, and Clustering-Based Local Outlier Factor, and Angle-based Outlier Detection; classification based methods such as Isolation Forest and One-class SVM Detector;  the dimensionality reduction method of Principal Component Analysis; and finally neural network related methods such as Autoencoders and Deep Support Vector Data Description.
&nbsp;

In stock price time series, “normal” or “outlier” labeled results for training and validation are generally unavailable, necessitating an unsupervised learning method. Unsupervised anomaly detection methods implicitly assume that the normal objects are somewhat “clustered.”  Given the unsupervised nature of stock price anomalies, we try to detect all three types of (global, conditional, and collective) outliers in entirely artificially constructed situations. We then inject outliers into the actual stock prices of Nikkei 225 constituents using minute intervals over two years. The PDF file [AnomalyDetectionIntraDayStockPrices.pdf](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/blob/main/AnomalyDetectionIntraDayStockPrices.pdf) contains a comparison of the effectiveness of the different anomaly detection methods.

&nbsp;

## folder structure

### [data](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/tree/main/data)

`data/prices/` - raw minute price data in csv format (not included due to licensing)
| Date                | HIGH | LOW  | OPEN | CLOSE | COUNT | VOLUME |
|---------------------|------|------|------|-------|-------|--------|
| 2020-11-24 00:04:00 | 9430 | 9392 | 9415 | 9410  | 397   | 729900 |
| 2020-11-24 00:05:00 | 9453 | 9409 | 9409 | 9446  | 320   | 169200 |
|                     |      |      |      |       |       |        |

[data/results/](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/tree/main/data/results) - anomaly detection results in csv format
|    Anomaly |     Model |   Stock | Accuracy | Precision | Recall | F1 Score |
|-----------:|----------:|--------:|---------:|----------:|-------:|---------:|
| Collective |  cluster  |  9432.T | 0.9993   | 0.8477    | 1      | 0.9176   |
| Collective |  DeepSVDD |  6098.T | 0.9992   | 0.8249    | 1      | 0.9040   |
|            |           |         |          |           |        |          |

[data/Nikkei225.csv](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/blob/main/data/Nikkei225.csv) - [Nikkei 225 Constituents & TSI](https://www.nikkei.com/markets/kabu/nidxprice/)
|   | Instrument | Organization Name | TSE33 Subsector name |
|---|------------|-------------------|----------------------|
| 0 | 2802.T     | Ajinomoto Co Inc  | Foods                |
| 1 | 9202.T     | ANA Holdings Inc  | Air Transportation   |
|   |            |                   |                      |

[data/TSE33.csv](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/blob/main/data/TSE33.csv) - [TOPIX Sector Indices (TSI) and Market Data Codes](https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_13_sector.pdf)
| TSI                             | QUICK_PR | QUICK_TR | BBG_TR         | BBG_PR           | REFINITIV_PR | REFINITIV_TR |
|---------------------------------|----------|----------|----------------|------------------|--------------|--------------|
| Fishery, Agriculture & Forestry | 321      | S321/TSX | TPFISH <INDEX> | TPXDFISH <INDEX> | .IFISH.T     | .IFISHDV.T   |
| Foods                           | 322      | S322/TSX | TPFOOD <INDEX> | TPXDFOOD <INDEX> | .IFOOD.T     | .IFOODDV.T   |
|                                 |          |          |                |                  |              |              |

[data/eikon_api.py](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/blob/main/data/eikon_api.py) - Class for downloading and storing data from Refinitiv Eikon api (requires license) in `data/prices/` folder

[data/jpx.py](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/blob/main/data/jpx.py) - class for preprocessing and exposing Japan Exchange Goup price data (data must already be in data folder see `data/eikon_api.py`)

&nbsp;
&nbsp;
### [utils](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/tree/main/utils)

[utils/outlier.py](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/blob/main/utils/outlier.py) - Code for injecting outliers into lists, series, and dataframes

&nbsp;
&nbsp;

### [data_examination.ipynb](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/blob/main/data_examination.ipynb) - investigating stock and TSI price data

### [main_test.ipynb](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/blob/main/main_test.ipynb) - main notebook for generating results from 10 anomaly detection methods

[arima.ipynb](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/blob/main/arima.ipynb) - investigating and running ARIMA anomaly detection method

[results.txt](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/blob/main/results.txt) - results generated from main_test.ipynb and arima.ipynb. Should be appended manually to results in `data/results/results.txt`.

[visualization_2d_and_3dTSNE.ipynb](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/blob/main/visualization_2d_and_3dTSNE.ipynb) - 2d visualizations of model results as well as t-SNE plot to view high dimensional data

[visualization_3d_interactive.ipynb](https://github.com/JamesSullivan/stock_intraday_anomaly_detection_comparison/blob/main/visualization_3d_interactive.ipynb) - interactive 3-D Visualization of Anomaly Detection of Global, Contextual, and Collective Outliers

