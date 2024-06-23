# cloud-burst-prediction
This repository contains a project for predicting cloud bursts using machine learning techniques. The project is developed in a Jupyter Notebook and includes data loading, preprocessing, model training, and model evaluation.

## Project Structure

The project structure is as follows:

- `cloud_burst_prediction.ipynb`: The main Jupyter Notebook file containing the code for the project.
- `dataset/cloud_burst_dataset.csv`: The dataset used for training and testing the model.
- `random_forest_model.joblib`: The trained Random Forest model saved using joblib.

## Dependencies

The project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- joblib

## Setup and Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/cloud_burst_prediction.git
   cd cloud_burst_prediction
   ```

2. **Install the required libraries:**

   If you are using a virtual environment, activate it first. Then install the required libraries:

   ```bash
   pip install pandas numpy scikit-learn joblib
   ```

3. **Load and run the Jupyter Notebook:**

   Open the `cloud_burst_prediction.ipynb` notebook using Jupyter Notebook or Jupyter Lab:

   ```bash
   jupyter notebook cloud_burst_prediction.ipynb
   ```

4. **Run the Notebook:**

   Execute the cells in the notebook to load the data, preprocess it, train the model, and evaluate its performance.

## Data Preprocessing

The dataset used in this project includes the following features:

- `MaximumTemperature`
- `WindSpeed9am`
- `Humidity9am`
- `Pressure9am`

The target variable is `CloudBurst Today`, which indicates whether a cloud burst occurred on that day.

## Model Training

The project uses a Random Forest classifier for predicting cloud bursts. The data is split into training and testing sets, and the model is trained on the training set. The performance of the model is evaluated using the test set, and the accuracy is printed.

## Model Saving

The trained model is saved using `joblib` for future use. The saved model can be loaded and used for predictions on new data.

## Additional Notes

- The notebook includes commented-out code for imputing missing values and transforming the features using logarithmic transformation. You can uncomment and modify this code as needed based on your data preprocessing requirements.
- The notebook also includes commented-out code for installing and using the SHAP library for model interpretability. You can uncomment and use this code to generate SHAP values and plots.
