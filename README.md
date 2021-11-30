# 📈 Time series resources 📉

A collection of resources for working with sequential and time series data

- [📈 Time series resources 📉](#-time-series-resources-)
  - [📦 Packages](#-packages)
    - [Python](#python)
      - [Date and Time](#date-and-time)
      - [Feature Engineering](#feature-engineering)
      - [Time Series Segmentation & Change Point Detection](#time-series-segmentation--change-point-detection)
      - [Time Series Augmentation](#time-series-augmentation)
      - [Visualization](#visualization)
      - [Benchmarking & Contests](#benchmarking--contests)
    - [R](#r)
    - [Java](#java)
    - [JavaScript](#javascript)
      - [Visualization](#visualization-1)
    - [Spark](#spark)
  - [🗄️ Databases](#️-databases)
    - [Managed database services](#managed-database-services)
  - [✏️ Annotation and Labeling](#️-annotation-and-labeling)
  - [📝 Papers with code](#-papers-with-code)
  - [💻 Repos with Models](#-repos-with-models)
  - [📚 Books](#-books)
  - [🎓 Courses](#-courses)
  - [💬 Communities](#-communities)
  - [🗃 Organizations](#-organizations)
  - [💼 Commercial Platforms](#-commercial-platforms)
  - [🕶️ More Awesomeness](#️-more-awesomeness)

## 📦 Packages

### Python

- [adtk](https://github.com/arundo/adtk) A Python toolkit for rule-based/unsupervised anomaly detection in time series.
- [alibi-detect](https://github.com/SeldonIO/alibi-detect) Algorithms for outlier, adversarial and drift detection.
- [Auto_TS](https://github.com/AutoViML/Auto_TS) Automatically build ARIMA, SARIMAX, VAR, FB Prophet and XGBoost Models on Time Series data sets with a Single Line of Code. Now updated with Dask to handle millions of rows.
- [cesium](https://github.com/cesium-ml/cesium) Open-Source Platform for Time Series Inference.
- [darts](https://github.com/unit8co/darts) Time Series Made Easy in Python. A python library for easy manipulation and forecasting of time series.
- [deeptime](https://github.com/deeptime-ml/deeptime) Python library for analysis of time series data including dimensionality reduction, clustering, and Markov model estimation.
- [dtw-python](https://github.com/DynamicTimeWarping/dtw-python) Python port of R's Comprehensive Dynamic Time Warp algorithm package.
- [gluon-ts](https://github.com/awslabs/gluon-ts) Probabilistic time series modeling in Python from AWS.
- [gordo](https://github.com/equinor/gordo) Building thousands of models with time series data to monitor systems.
- [greykite](https://github.com/linkedin/greykite) A flexible, intuitive and fast forecasting library from LinkedIn.
- [hmmlearn](https://github.com/hmmlearn/hmmlearn) Hidden Markov Models in Python, with `scikit-learn` like API.
- [kats](https://github.com/facebookresearch/kats) A kit to analyze time series data, a lightweight, easy-to-use, generalizable, and extendable framework to perform time series analysis, from understanding the key statistics and characteristics, detecting change points and anomalies, to forecasting future trends.
- [lifelines](https://github.com/CamDavidsonPilon/lifelines) Survival analysis in Python.
- [luminaire](https://github.com/zillow/luminaire) A python package that provides ML driven solutions for monitoring time series data. Luminaire provides several anomaly detection and forecasting capabilities that incorporate correlational and seasonal patterns in the data over time as well as uncontrollable variations.
- [mass-ts](https://github.com/matrix-profile-foundation/mass-ts) Mueen's Algorithm for Similarity Search, a library used for searching time series sub- sequences under z-normalized Euclidean distance for similarity.
- [matrixprofile](https://github.com/matrix-profile-foundation/matrixprofile) A Python library making time series data mining tasks, utilizing matrix profile algorithms, accessible to everyone.
- [Merlion](https://github.com/salesforce/Merlion) A Python library for time series intelligence. It provides an end-to-end machine learning framework that includes loading and transforming data, building and training models, post-processing model outputs, and evaluating model performance.
- [orbit](https://github.com/uber/orbit) A package for Bayesian forecasting with object-oriented design and probabilistic models under the hood from Uber.
- [pastas](https://github.com/pastas/pastas) An open-source Python framework for the analysis of hydrological time series.
- [pmdarima](https://github.com/alkaline-ml/pmdarima) A statistical library designed to fill the void in Python's time series analysis capabilities, including the equivalent of R's `auto.arima` function.
- [prophet](https://github.com/facebook/prophet) Tool for producing high quality forecasts for time series data that has multiple seasonality with linear or non-linear growth.
- [PyDLM](https://github.com/wwrechard/pydlm) Bayesian time series modeling package. Based on the Bayesian dynamic linear model (Harrison and West, 1999) and optimized for fast model fitting and inference.
- [PyFlux](https://github.com/RJT1990/pyflux) Open source time series library for Python.
- [Pyod](https://github.com/yzhao062/Pyod) A Python toolbox for scalable outlier detection (Anomaly Detection).
- [scikit-hts](https://github.com/carlomazzaferro/scikit-hts) Hierarchical Time Series Forecasting with a familiar API.
- [seglearn](https://github.com/dmbee/seglearn) A python package for machine learning time series or sequences.
- [shyft](https://gitlab.com/shyft-os/shyft) Time-series for python and c++, including distributed storage and calculations Hydrologic Forecasting Toolbox, high-performance flexible stacks, including calibration Energy-market models and micro services.
- [similarity_measures](https://github.com/cjekel/similarity_measures) Quantify the difference between two arbitrary curves.
- [sktime](https://github.com/alan-turing-institute/sktime) A `scikit-learn` compatible Python toolbox for learning with time series.
- [statsmodels.tsa](https://www.statsmodels.org/stable/tsa.html) Time Series Analysis (tsa) `statsmodels.tsa` contains model classes and functions that are useful for time series analysis.
- [stumpy](https://github.com/TDAmeritrade/stumpy) A powerful and scalable Python library that can be used for a variety of time series data mining tasks.
- [TICC](https://github.com/davidhallac/TICC) A python solver for efficiently segmenting and clustering a multivariate time series.
- [timemachines](https://github.com/microprediction/timemachines) Continuously evaluated, functional, incremental, time-series forecasting.
- [TimeSynth](https://github.com/TimeSynth/TimeSynth) A multipurpose library for synthetic time series in Python.
- [tsflex](https://github.com/predict-idlab/tsflex) Flexible time series feature extraction & processing.
- [tslearn](https://github.com/tslearn-team/tslearn) The machine learning toolkit for time series analysis in Python.
- [tsmoothie](https://github.com/cerlymarco/tsmoothie) A python library for time-series smoothing and outlier detection in a vectorized way.

#### Date and Time
*Libraries for working with dates and times.*

- [astral](https://github.com/sffjunkie/astral) Python calculations for the position of the sun and moon.
- [Arrow](https://arrow.readthedocs.io/en/latest/) - A Python library that offers a sensible and human-friendly approach to creating, manipulating, formatting and converting dates, times and timestamps.
- [Chronyk](https://github.com/KoffeinFlummi/Chronyk) - A Python 3 library for parsing human-written times and dates.
- [dateutil](https://github.com/dateutil/dateutil) - Extensions to the standard Python [datetime](https://docs.python.org/3/library/datetime.html) module.
- [delorean](https://github.com/myusuf3/delorean/) - A library for clearing up the inconvenient truths that arise dealing with datetimes.
- [maya](https://github.com/timofurrer/maya) - Datetimes for Humans.
- [moment](https://github.com/zachwill/moment) - A Python library for dealing with dates/times. Inspired by [Moment.js](http://momentjs.com/).
- [Pendulum](https://github.com/sdispater/pendulum) - Python datetimes made easy.
- [PyTime](https://github.com/shinux/PyTime) - An easy-to-use Python module which aims to operate date/time/datetime by string.
- [pytz](https://launchpad.net/pytz) - World timezone definitions, modern and historical. Brings the [tz database](https://en.wikipedia.org/wiki/Tz_database) into Python.
- [when.py](https://github.com/dirn/When.py) - Providing user-friendly functions to help perform common date and time actions.

#### Feature Engineering

- [featuretools](https://github.com/alteryx/featuretools) An open source python library for automated feature engineering.
- [tsfel](https://github.com/fraunhoferportugal/tsfel) An intuitive library to extract features from time series.
- [tsfresh](https://github.com/blue-yonder/tsfresh) The package contains many feature extraction methods and a robust feature selection algorithm.

#### Time Series Segmentation & Change Point Detection

- [bayesian_changepoint_detection](https://github.com/hildensia/bayesian_changepoint_detection) Methods to get the probability of a change point in a time series. Both online and offline methods are available.
- [changepy](https://github.com/ruipgil/changepy) Change point detection in time series in pure python.
- [ruptures](https://github.com/deepcharles/ruptures) A Python library for off-line change point detection. This package provides methods for the analysis and segmentation of non-stationary signals.
- [TCPDBench](https://github.com/alan-turing-institute/TCPDBench) Turing Change Point Detection Benchmark, a benchmark evaluation of change point detection algorithms.

#### Time Series Augmentation

- [deltapy](https://github.com/firmai/deltapy) Tabular Data Augmentation & Feature Engineering.
- [tsaug](https://github.com/arundo/tsaug) A Python package for time series augmentation.
- [time_series_augmentation](https://github.com/uchidalab/time_series_augmentation) An example of time series augmentation methods with Keras.

#### Visualization

- [atlair](https://github.com/altair-viz/altair) Declarative statistical visualization library for Python.
- [matplotlib](https://matplotlib.org/) A comprehensive library for creating static, animated, and interactive visualizations in Python.
- [plotly](https://plotly.com/) A graphing library makes interactive, publication-quality graphs.
- [seaborn](https://seaborn.pydata.org/) A data visualization library based on [matplotlib](https://matplotlib.org) that provides a high-level interface for drawing attractive and informative statistical graphics.

#### Benchmarking & Contests

- [timeseries-elo-ratings](https://github.com/microprediction/timeseries-elo-ratings) computes [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/overall.html).
- [m6](https://github.com/microprediction/m6) Provides utilities for the [M6-Competition](https://mofc.unic.ac.cy/the-m6-competition/)
- Microprediction time-series [competitions](https://www.microprediction.com/competitions) 

### R

- [fable](https://github.com/tidyverts/fable) A [tidyverts](https://github.com/tidyverts) package for tidy time series forecasting.
- [feasts](https://github.com/tidyverts/feasts) A [tidyverts](https://github.com/tidyverts) package for feature extraction and statistics for time series.
- [modeltime](https://github.com/business-science/modeltime) Modeltime unlocks time series forecast models and machine learning in one framework.
- [timetk](https://github.com/business-science/timetk) A `tidyverse` toolkit to visualize, wrangle, and transform time series data.
- [tsibble](https://github.com/tidyverts/tsibble) A [tidyverts](https://github.com/tidyverts) package with tidy temporal data frames and tools.
- [fasster](https://github.com/tidyverts/fasster) A [tidyverts](https://github.com/tidyverts) package for forecasting with additive switching of seasonality, trend and exogenous regressors.

### Java

- [SFA](https://github.com/patrickzib/SFA) Scalable Time Series Data Analytics.
- [tsml](https://github.com/uea-machine-learning/tsml) Java time series machine learning tools in a Weka compatible toolkit.

### JavaScript

#### Visualization

- [cubism](https://github.com/square/cubism) A [D3](http://d3js.org) plugin for visualizing time series. Use Cubism to construct better realtime dashboards, pulling data from [Graphite](https://github.com/square/cubism/wiki/Graphite), [Cube](https://github.com/square/cubism/wiki/Cube) and other sources.
- [echarts](https://github.com/apache/echarts) A free, powerful charting and visualization library offering an easy way of adding intuitive, interactive, and highly customizable charts to your commercial products.
- [fusiontime](https://www.fusioncharts.com/fusiontime) Helps you visualize time-series and stock data in JavaScript, with just a few lines of code.
- [highcharts](https://github.com/highcharts/highcharts) A JavaScript charting library based on SVG, with fallbacks to VML and canvas for old browsers.
- [synchro-charts](https://github.com/awslabs/synchro-charts) A front-end component library that provides a collection of components to visualize time-series data.

### Spark

- [flint](https://github.com/twosigma/flint) A Time Series Library for Apache Spark.

## 🗄️ Databases

- [atlas](https://github.com/Netflix/atlas) In-memory dimensional time series database from Netflix.
- [cassandra](https://cassandra.apache.org/_/index.html) Apache Cassandra is an open source NoSQL distributed database trusted by thousands of companies for scalability and high availability without compromising performance.
- [ClickHouse](https://clickhouse.com/) An open-source, high performance columnar OLAP database management system for real-time analytics using SQL.
- [cratedb](https://crate.io/use-cases/time-series/) The SQL database for complex, large scale time series workloads in industrial IoT.
- [druid](https://druid.apache.org/) A high performance real-time analytics database.
- [fauna](https://fauna.com) Fauna is a flexible, developer-friendly, transactional database delivered as a secure and scalable cloud API with native GraphQL.
- [InfluxDB](https://www.influxdata.com/products/influxdb/) Is the essential time series toolkit - dashboards, queries, tasks and agents all in one place.
- [KairosDB](https://kairosdb.github.io/) Fast Time Series Database on Cassandra.
- [opendTSDB](http://opentsdb.net/) The Scalable Time Series Database.
- [prometheus](https://prometheus.io/) An open-source systems monitoring and alerting toolkit originally built at [SoundCloud](https://soundcloud.com).
- [QuestDB](https://github.com/questdb/questdb) An open source SQL database designed to process time series data, faster.
- [TimeScaleDB](https://www.timescale.com/) TimescaleDB is the leading open-source relational database with support for time-series data.

### Managed database services

- [Amazon Timestream](https://aws.amazon.com/timestream/) A fast, scalable, and serverless time series database service for IoT and operational applications.
- [Azure Database for PostgreSQL](https://azure.microsoft.com/en-us/services/postgresql/#overview) Fully managed, intelligent, and scalable PostgreSQL with support for TimeScaleDB.
- [Azure time series insights](https://azure.microsoft.com/en-us/services/time-series-insights/) Visualize IoT data in real time.
- [InfluxDB Cloud](https://www.influxdata.com/products/influxdb-cloud/) It’s a fast, elastic, serverless real-time monitoring platform, dashboarding engine, analytics service and event and metrics processor.
- [TimeScaleDB Cloud](https://www.timescale.com/products/#Timescale-Cloud) Managed TimeScaleDB service.

## ✏️ Annotation and Labeling

- [AnnotateChange](https://github.com/alan-turing-institute/AnnotateChange) - A simple flask application to collect annotations for the Turing Change Point Dataset, a benchmark dataset for change point detection algorithms.
- [Curve](https://github.com/baidu/Curve) - An open-source tool to help label anomalies on time-series data
- [TagAnomaly](https://github.com/Microsoft/TagAnomaly) - Anomaly detection analysis and labeling tool, specifically for multiple time series (one time series per category)
- [time-series-annotator](https://github.com/CrowdCurio/time-series-annotator) - Time Series Annotation Library implements classification tasks for time series.
- [WDK](https://github.com/avenix/WDK) - The Wearables Development Toolkit (WDK) is a set of tools to facilitate the development of activity recognition applications with wearable devices.

## 📝 Papers with code

- **Deep learning for time series classification: a review**, _H. I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, P-A. Muller_, Data Mining and Knowledge Discovery 2019
  - [code](https://github.com/hfawaz/dl-4-tsc)
  - [paper](https://arxiv.org/abs/1809.04356v3)

- **Greedy Gaussian Segmentation of Multivariate Time Series**, _D. Hallac, P. Nystrup, and S. Boyd_, Advances in Data Analysis and Classification, 13(3), 727–751, 2019.
	- [code](https://github.com/cvxgrp/GGS)
	- [paper](http://stanford.edu/~boyd/papers/ggs.html)

- **U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging**, _Mathias Perslev, Michael Jensen, Sune Darkner, Poul Jørgen Jennum, Christian Igel_, _NeurIPS_, 2019.
	- [paper](https://arxiv.org/abs/1910.11162)
	- [code](https://github.com/perslev/U-Time)

- **A Better Alternative to Piecewise Linear Time Series Segmentation**, _Daniel Lemire_, SIAM Data Mining, 2007.
	- [paper](https://arxiv.org/abs/cs/0605103)
	- [code](https://github.com/lemire/PiecewiseLinearTimeSeriesApproximation)

- **Time-series Generative Adversarial Networks**, _Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar_, NeurIPS, 2019.
	- [paper](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)
	- [code](https://github.com/jsyoon0823/TimeGAN)

- **Learning to Diagnose with LSTM Recurrent Neural Networks**, _Zachary C. Lipton, David C. Kale, Charles Elkan, Randall Wetzel_, arXiv:1511.03677, 2015.
	- [paper](https://arxiv.org/abs/1511.03677)
	- [code](https://github.com/aqibsaeed/Multilabel-timeseries-classification-with-LSTM)

## 💻 Repos with Models

- [ESRNN-GPU](https://github.com/damitkwr/ESRNN-GPU) PyTorch GPU implementation of the ES-RNN model for time series forecasting.
- [LSTM-Neural-Network-for-Time-Series-Prediction](https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction) LSTM built using Keras Python package to predict time series steps and sequences.
- [ROCKET](https://github.com/angus924/rocket) Exceptionally fast and accurate time series classification using random convolutional kernels.
- [TensorFlow-Time-Series-Examples](https://github.com/hzy46/TensorFlow-Time-Series-Examples) Time Series Prediction with `tf.contrib.timeseries`
- [TimeSeries](https://github.com/zhykoties/TimeSeries) Implementation of deep learning models for time series in PyTorch.
- [UCR_Time_Series_Classification_Deep_Learning_Baseline](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline) Fully Convlutional Neural Networks for state-of-the-art time series classification.

## 📚 Books

- [Bayesian Time Series Models ](https://www.cambridge.org/pl/academic/subjects/computer-science/pattern-recognition-and-machine-learning/bayesian-time-series-models?format=HB) 💲 _David Barber, A. Taylan Cemgil, Silvia Chiappa_, Cambridge Academic Press 2011
- [Forecasting principles and practice (3rd ed)](https://otexts.com/fpp3/) 🆓 _Rob J Hyndman and George Athanasopoulos_ 2021
- [Practical Time Series Analysis](https://www.packtpub.com/product/practical-time-series-analysis/9781788290227) 💲 _Avishek Pal, PKS Prakash_, Packt 2017
  - [repo with code](https://github.com/PacktPublishing/Practical-Time-Series-Analysis)
- [Practical Time Series Analysis: Prediction with Statistics and Machine Learning](https://www.oreilly.com/library/view/practical-time-series/9781492041641/) 💲  _Aileen Nielsen_, O’Reilly 2019
	- [repo with code](https://github.com/AileenNielsen/TimeSeriesAnalysisWithPython)
- [Machine Learning for Time-Series with Python](https://www.packtpub.com/product/machine-learning-for-time-series-with-python/9781801819626) 💲 _Ben Auffarth_, Packt Publishing 2021
  - [repo with code](https://github.com/PacktPublishing/Machine-Learning-for-Time-Series-with-Python)
- [Visualization of Time-Oriented Data](https://amzn.to/3GITSpD) 💲 _Wolfgang Aigner, Silvia Miksch, Heidrun Schumann, Christian Tominski_, Springer-Verlag 2011

## 🎓 Courses

- [Analytics Vidhya - Time Series Forecasting using Python](https://courses.analyticsvidhya.com/courses/creating-time-series-forecast-using-python)
- [Coursera - Intro to Time Series Analysis in R](https://www.coursera.org/projects/intro-time-series-analysis-in-r)
- [Coursera - Sequences, Time Series and Prediction](https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction)
- [Coursera - Practical Time Series Analysis](https://www.coursera.org/learn/practical-time-series-analysis)
- [Data For Science - Time Series for Everyone](https://data4sci.com/timeseries)
- [Kaggle - Time Series](https://www.kaggle.com/learn/time-series)
- [Udacity - Time Series Forecasting](https://www.udacity.com/course/time-series-forecasting--ud980)

## 💬 Communities

- [r/TimeSeries](https://www.reddit.com/r/TimeSeries) Subreddit for graphs of time series and predictions based on time series.

## 🗃 Organizations

- [Matrix Profile Foundation](https://www.matrixprofile.org/)

## 💼 Commercial Platforms

- [HAKOM TSM Platform](https://www.hakom.at/en/portfolio/technology/feasyuse/) Comprehensive functionality for pre-processing and storing time series from different sources in different formats for all analytical and operational purposes.

## 🕶️ More Awesomeness

- [cuge1995/awesome-time-series](https://github.com/cuge1995/awesome-time-series)
- [MaxBenChrist/awesome_time_series_in_python](https://github.com/MaxBenChrist/awesome_time_series_in_python)
- [vinta/awesome-python](https://github.com/vinta/awesome-python)
- [xephonhq/awesome-time-series-database](https://github.com/xephonhq/awesome-time-series-database) A curated list of awesome time series databases, benchmarks and papers.
