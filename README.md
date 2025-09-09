# Laptop Price Prediction Application

A machine learning application that predicts laptop prices based on their specifications. Built with Python, Scikit-learn, and Streamlit.

## Project Overview

This application uses machine learning to predict laptop prices based on various specifications like CPU speed, memory size, GPU, and display features. The model was trained on historical laptop price data and provides price estimates in Euros.

## 📊 Features

- Real-time price prediction
- User-friendly web interface
- Support for multiple laptop specifications
- Instant feedback and results
- Comprehensive error handling

## 🛠️ Technical Stack

- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

## 📋 System Requirements

- Python 3.8 or higher
- Minimum 4GB RAM (8GB recommended)
- 1GB free disk space
- Windows 10/11, macOS, or Linux

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/scientistEgong/Laptop-Price-Prediction-Model.git
cd laptop-price-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

1. Navigate to the app directory:
```bash
cd app
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and go to:
```
http://localhost:8501
```

## 📝 How to Use

1. Enter Laptop Specifications:
   - CPU Speed (GHz)
   - Memory Size (GB)
   - Storage Type (SSD/HDD)
   - GPU Company
   - Screen Resolution

2. Click the "Predict Price" button

3. View the predicted price and confidence information

## 📁 Project Structure

```
Car-Price-Prediction-model/
├── app/
│   ├── models/
│   │   └── best_model.joblib
│   └── app.py
├── Project_Files/
│   ├── Notebook_Files/
│   │   └── Notebooks/
│   │       ├── Feature_Engineering.ipynb
│   │       └── Model_building_and_training.ipynb
│   └── Dependencies_Check/
│       └── Dependencies_Check.ipynb
├── requirements.txt
└── README.md
```

## 📊 Model Performance

- Algorithm: Linear Regression
- R² Score: [Your R² score]
- RMSE: [Your RMSE value]
- MAE: [Your MAE value]

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](link-to-issues).

## 📝 License

This project is [MIT](LICENSE) licensed.


## 🙏 Acknowledgments

- Dataset source: [Source name/link]
- Contributors and reviewers
- Open source community

## 📞 Support

For support, email [your-email@example.com] or create an issue in the repository.