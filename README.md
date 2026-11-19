# Experiment 4: Multiple Regression Analysis (Fish Dataset)

## Aim
To perform Multiple Linear Regression on a dataset imported from a CSV file to analyze how multiple features together predict a dependent variable.

## Dataset
The dataset used for this experiment is `fish.csv` which contains real-world physical measurements of various fish species:
* **Independent Variables ($X$):**
  * `L`: Mean length derived from `Length1`, `Length2`, and `Length3`
  * `Height`: Vertical height of the fish
  * `Width`: Diagonal width of the fish
* **Dependent Variable ($Y$):**
  * `Weight`: Weight of the fish (in grams)

## Algorithm / Steps Followed
1. Import the required Python libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. Load the fish dataset into a Pandas DataFrame using `pd.read_csv()`.
3. Display the dataset and inspect its columns.
4. Define the dependent ($Y$) and independent ($X$) variables.
5. Train the Multiple Linear Regression model using `LinearRegression()`.
6. Display the model coefficients (`model.coef_`), intercept (`model.intercept_`), and regression equation.
7. Predict target variable (`Weight`) values using `model.predict()`.
8. Evaluate model accuracy using the $R^2$ score (`r2_score()`).
9. Plot actual vs. predicted values for visual comparison.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Creation / Handling** | `pd.read_csv()` | Load the CSV dataset into a Pandas DataFrame |
| | `np.array()` / `pd.DataFrame()` | Create and manage numerical arrays and tabular data for regression |
| **Model Building (Scikit-learn)** | `LinearRegression()` | Initialize the Multiple Linear Regression model |
| | `model.fit()` | Train the regression model using multiple independent variables ($X$) and target ($Y$) |
| | `model.predict()` | Predict dependent variable values ($Y$) using the trained model |
| | `r2_score()` | Evaluate the accuracy of the regression model using the $R^2$ score |
| **Model Attributes** | `model.intercept_` | Retrieve the intercept (constant term) of the regression equation |
| | `model.coef_` | Retrieve the coefficients (slopes) for each independent variable |
| **Visualization (Matplotlib)** | `plt.plot()` | Plot actual and predicted weight values for comparison |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, `plt.show()` | Configure titles, axis labels, legend, and display the plotted figure |

## Program & Output

# Step 1: Import Required Libraries  
import pandas as pd    
import numpy as np  
from sklearn.linear_model import LogisticRegression  
import matplotlib.pyplot as plt  

# Step 2: Create Dataset  
# Features: Hours_Studied, Previous_Exam_Score  
# Target: Pass (1 = Pass, 0 = Fail)  

data = {  
    'Hours_Studied': [1, 2.5, 3, 3.5, 5, 6.5, 7.5, 6.5, 8.5, 6],  
    'Previous_Exam_Score': [55, 60, 65, 70, 75, 80, 85, 90, 95, 100],  
    'Pass': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  
}

df = pd.DataFrame(data)

print("----- Dataset -----")  
print(df)  
 
# Step 3: Define Independent (X) and Dependent (Y) Variables  
X = df[['Hours_Studied', 'Previous_Exam_Score']]  
Y = df['Pass']  

# Step 4: Create and Train Logistic Regression Model  
model = LogisticRegression()  
model.fit(X, Y)  

# Step 5: Display Model Coefficients  
print("\nIntercept (a):", model.intercept_)  
print("Coefficients (b1, b2):", model.coef_)  

# Step 6: Predict Class Labels and Probabilities  
Y_pred = model.predict(X)  
Y_prob = model.predict_proba(X)[:, 1]  

# Step 7: Display Predictions  
df['Predicted_Pass'] = Y_pred  
df['Pass_Probability'] = Y_prob  

print("\nPredictions with Probabilities:\n")  
print(df)  

# Step 8: Visualization  
plt.figure(figsize=(8, 6))  
plt.scatter(  
    df['Hours_Studied'],  
    df['Previous_Exam_Score'],  
    c=df['Pass'],   
    cmap='coolwarm',  
    s=100,  
    label='Actual'  
)
plt.title("Logistic Regression - Student Pass/Fail Classification")  
plt.xlabel("Hours_Studied")  
plt.ylabel("Previous Exam Score")  
plt.legend()  
plt.show()  

<img width="730" height="561" alt="image" src="https://github.com/user-attachments/assets/2a53d8e4-e8e2-4594-a33c-2bcc7909e3e8" />

<img width="475" height="297" alt="image" src="https://github.com/user-attachments/assets/084d76ec-732a-4237-abed-01e411450dc2" />


## Result
