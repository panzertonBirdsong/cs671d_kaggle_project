# Airbnb Price Prediction (CS671D Kaggle Competition)

This project is for Duke University's course COMPSCI-671d's kaggle competition. The competition aims to predict the prices of AirBNBs in New York City based off of information about its location, amenities, host, availability, etc.. The XGBoost model is the final submitted one and achieved 10th position out of 137 students intotal.

## Design

Here presents the basic ideas of the project. For the details please read the ProjectReport.pdf.

### 1. Exploratory Analysis & Feature Selection
The project began by analyzing the distribution of the dataset, which contained numerical, boolean, datetime, and object features.
* **Distribution Analysis**: Kernel Density Estimation (KDE) plots revealed that the target variable (`price`) was balanced, but many feature distributions were heavily skewed.
* **Correlation Filtering**: We calculated the correlation between each feature and the target variable. Features with a correlation absolute value below 0.005 were dropped to reduce noise.
* **Irrelevant Data**: Features like `name` and `description` were dropped as they were deemed irrelevant for the regression task.

### 2. Feature Engineering
We implemented several strategies to maximize the information extracted from the data:
* **Datetime Processing**: Date fields were split into three separate numerical features: Year, Month, and Day.
* **Categorical Encoding**: Object features were One-Hot Encoded, while the `amenities` list was processed using a `MultiLabelBinarizer`.
* **Custom Interactions**: We engineered interaction features to capture specific market dynamics. For example, the ratio of bedrooms to bathrooms was calculated to distinguish between "luxury" (more bathrooms per room) and "economic" (shared bathrooms) listings.

### 3. Model Comparison
Two distinct model architectures were implemented and tested:

* **XGBRegressor (Selected Model)**:
    * **Reasoning**: XGBoost was chosen for its ability to handle skewed distributions and missing values without requiring extensive normalization.
    * **Training**: We employed a two-stage training process. First, we trained on all features to determine feature importance, then refined the model by selecting only the features contributing to the top 95% of importance.
    * **Optimization**: Hyperparameters (estimators, learning rate, regularization) were tuned using `GridSearchCV` with 5-fold cross-validation.

* **MLP (Multi-Layer Perceptron)**:
    * **Reasoning**: A neural network was implemented to capture nuanced, non-linear relationships.
    * **Architecture**: The model used the Adam optimizer (combining Momentum and RMSProp) and a 2-layer architecture, which experiments showed performed better than deeper networks.
    * **Normalization**: Unlike XGBoost, the MLP required all numerical data to undergo Min-Max normalization.
    * **Result**: The MLP consistently showed higher validation loss compared to XGBoost, so XGBoost was selected for the final submission.

## Repository Structure

* **`XGBRegressor/`**: Contains the Jupyter notebook for the XGBoost model implementation and experiments.
* **`MLP/`**: Contains the Python scripts for the Neural Network implementation (`mlp_model.py`, `price_model.py`, `main.py`).
* **`ProjectReport.pdf`**: The detailed final report explaining the methodology, feature selection, and results.
* **`requirements.txt`**: List of Python dependencies required to run the code.
* **`data.zip`**: Compressed dataset used for training.

## Setup and Installation

1.  **Environment Setup**
    Ensure you have Python installed. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Preparation**
    * Unzip `data.zip` into the project directory.
    * **Before Running**: Before running any scripts or notebooks, please open `project.ipynb` or `main.py` and **check the datapath**. Update the path variable to point to the location where you extracted the data on your local machine.

## Usage

### Running the XGBoost Model
Open the `project.ipynb` notebook found in the `XGBRegressor` folder.

**Important Note on Execution Sequence:**
When running the training section of the notebook, the `train_test_split` block and the `GridSearchCV` block **cannot be run in sequence**. If you wish to switch between these blocks, please **rerun the data reading block** first to reset and process the dataframe correctly.

### Running the MLP Model
To train the neural network, run the main script from the root directory:
```bash
python MLP/main.py
