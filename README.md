🔎 DataDetective AI

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=24&duration=2800&pause=800&color=2A9DB8&center=true&vCenter=true&width=650&lines=Explore+your+data.;Detect+unusual+records.;Clean+and+export+CSV+files." alt="DataDetective AI animated tagline" />
</p><p align="center">
  <strong>Explore datasets. Detect anomalies. Clean data. Export results.</strong>
</p><p align="center">
  <a href="https://data-detective-e1l1.onrender.com/">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-Open_App-2A9DB8?style=for-the-badge" alt="Launch live application" />
  </a>
  <a href="https://github.com/Sumaya04122006/Data-Detective">
    <img src="https://img.shields.io/badge/GitHub-Source_Code-181717?style=for-the-badge&logo=github" alt="View source code" />
  </a>
</p><p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Gradio-Interactive_UI-FF7C00?style=flat-square" alt="Gradio" />
  <img src="https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat-square&logo=pandas" alt="Pandas" />
  <img src="https://img.shields.io/badge/Scikit--learn-Machine_Learning-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=flat-square&logo=plotly" alt="Plotly" />
</p>---

📌 Overview

DataDetective AI is an interactive CSV analysis application built with Python and machine learning. It helps users inspect datasets, evaluate data quality, explore patterns through interactive visualizations, identify potentially unusual records, and export cleaned data.

The application brings essential data exploration and preprocessing tools into one interface, making it easier to understand a dataset before using it for further analysis.

✨ Features

Feature| Description
📂 CSV Upload| Import CSV files for analysis.
🧪 Sample Dataset| Explore the application without preparing a dataset.
👀 Dataset Preview| Inspect the initial rows of a dataset.
📊 Data Profiling| Examine data types, descriptive statistics, missing values, duplicate rows, and unique values.
🚨 Anomaly Detection| Apply Isolation Forest to identify potentially unusual records.
🎛️ Configurable Analysis| Select numeric features and adjust the model's contamination parameter.
📈 Interactive Visualizations| Explore data distributions, feature relationships, and correlations using Plotly.
🧹 Data Cleaning| Fill missing values, remove incomplete rows, or eliminate duplicate records.
📥 CSV Export| Download processed data for subsequent analysis.

🚀 Live Demo

"Launch DataDetective AI →" (https://data-detective-e1l1.onrender.com/)

Upload your own CSV file or load the built-in sample dataset to explore the application's workflow.

«Note: The application is hosted on Render. If the hosting instance has been idle, the initial page load may take longer.»

🔄 Application Workflow

flowchart TD
    A["Upload CSV or Load Sample"] --> B["Read and Validate Dataset"]
    B --> C["Preview and Profile Data"]
    C --> D["Select Numeric Features"]
    D --> E["Run Isolation Forest"]
    E --> F["Review Flagged Records"]
    F --> G["Explore Interactive Charts"]
    C --> H["Select Cleaning Strategy"]
    H --> I["Export Processed CSV"]

    style A fill:#123C4A,stroke:#2A9DB8,color:#FFFFFF
    style E fill:#123C4A,stroke:#2A9DB8,color:#FFFFFF
    style I fill:#123C4A,stroke:#2A9DB8,color:#FFFFFF

🧠 How Anomaly Detection Works

DataDetective AI uses Isolation Forest, an unsupervised machine-learning algorithm that identifies observations that are relatively easy to isolate from the rest of a dataset.

The analysis follows these steps:

1. Select the numeric features to analyze.
2. Prepare the selected data for model input.
3. Fit the Isolation Forest model.
4. Identify records classified as potential anomalies.
5. Review the flagged records and supporting visualizations.

The contamination parameter influences the model's expected proportion of anomalies. It does not measure the dataset's actual error rate.

«Important: An anomaly is not automatically an error, a fraudulent record, or a meaningful problem. Flagged observations require further investigation and domain context.»

🧹 Data Cleaning

The application provides several cleaning strategies:

Strategy| Behavior
Keep missing values| Preserves existing missing entries.
Fill missing values| Uses the numeric median and categorical mode where applicable.
Drop incomplete rows| Removes rows containing missing values.
Remove duplicates| Eliminates duplicate rows.

Cleaning decisions can affect dataset size, statistical properties, and downstream analysis. Review the selected strategy before exporting the processed dataset.

🛠️ Technology Stack

Technology| Purpose
Python| Application logic and data processing
Gradio| Interactive web interface
Pandas| Data manipulation, profiling, and cleaning
NumPy| Numerical operations
Scikit-learn| Isolation Forest anomaly detection
Plotly Express| Interactive data visualizations

⚙️ Run Locally

Follow these steps to run the application on your machine.

1. Clone the repository

git clone https://github.com/Sumaya04122006/Data-Detective.git
cd Data-Detective

2. Create a virtual environment

python -m venv .venv

Activate the environment:

Windows

.venv\Scripts\activate

macOS / Linux

source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Start the application

python app.py

Open the local URL displayed in your terminal.

📁 Project Structure

Data-Detective/
├── app.py
├── requirements.txt
├── README.md
└── docs/
    └── screenshots/

The "docs/screenshots/" directory is optional. Add genuine screenshots of the running application to showcase its interface and functionality.

⚠️ Limitations and Responsible Use

- Anomaly detection: Identifies unusual patterns but does not establish their cause.
- Model sensitivity: Results depend on feature selection, data quality, and model settings.
- Data preprocessing: Filling or removing missing values can influence downstream results.
- Independent validation: Findings should be reviewed before being used for consequential decisions.
- Data privacy: Avoid uploading confidential, personally identifiable, or sensitive data to a public deployment without appropriate privacy and security protections.
- Deployment scope: Do not assume persistent database storage, user authentication, or enterprise-grade data governance unless these capabilities have been implemented and verified.

🔭 Future Improvements

- Generate downloadable anomaly reports with additional context.
- Support additional file formats.
- Expand configurable preprocessing and data-quality checks.
- Improve automated testing and error handling.
- Introduce stronger privacy controls and deployment monitoring.

👩‍💻 Author

Sumaya Fathima
Bachelor of Information Technology student specializing in Artificial Intelligence at Liwa University.

<p>
  <a href="https://github.com/Sumaya04122006">GitHub</a> ·
  <a href="https://www.linkedin.com/in/sumaya-fathima-7148063a9">LinkedIn</a> ·
  <a href="https://data-detective-e1l1.onrender.com/">Live Application</a>
</p>---

<p align="center">
  <sub>Built with Python for interactive dataset exploration, anomaly detection, and CSV cleaning.</sub>
</p>
