# Driving Behavior Prediction using MLOps
![flow](/img/flow.png)
<p align="center">
  Fig. 1 Overview of the proposed scheme
</p>

## Dataset
- [Driving Behavior Dataset](https://www.kaggle.com/datasets/outofskills/driving-behavior)
- Dataset Paper: I. Cojocaru and P. Popescu (2022). [Building a Driving Behaviour Dataset](https://rochi.utcluj.ro/articole/10/RoCHI2022-Cojocaru-I-1.pdf). Proceedings of RoCHI 2022.
- I have used Normal and Aggressive Class for this dataset, so the experiment in this repository is a binary classification task.

## Feature Engineering
- In the original dataset, there are 6 variables -  Acceleration (X, Y, Z axis in meters per second squared (m/s2)) and Rotation (X, Y, Z axis in degrees per second (°/s)).
- Beyond the existing variables, I have added some features that can be calculated by using existing variables. 

### 1. Acceleration Magnitude
- $\text{AccMagnitude} = \sqrt{\text{AccX}^2 + \text{AccY}^2 + \text{AccZ}^2}$
- The overall magnitude of 3-axis acceleration

### 2. Rotation Magnitude
- $\text{RotMagnitude} = \sqrt{\text{RotX}^2 + \text{RotY}^2 + \text{RotZ}^2}$
- The overall magnitude of 3-axis rotational velocity

### 3. Jerk
- $\text{JerkX} = \frac{d(\text{AccX})}{dt}$
- $\text{JerkY} = \frac{d(\text{AccY})}{dt}$
- $\text{JerkZ} = \frac{d(\text{AccZ})}{dt}$
- $\text{JerkMagnitude} = \sqrt{\text{JerkX}^2 + \text{JerkY}^2 + \text{JerkZ}^2}$
- The rate of change of acceleration over time
- Sudden changes in acceleration can indicate aggressive driving.

## Hyperparameter Tuning
- Using [Optuna](https://optuna.org/) to optimize hyperparameters of the predictive model

## MLOps

## Experimental Result
<table>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="4">w/o Feature Engineering (Original Data)</th>
    <th colspan="4">w/ Feature Engineering (Our Scheme)</th>
  </tr>
  <tr>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <th>Logistic Regression</th>
    <td>0.5602</td>
    <td>0.5572</td>
    <td>0.5557</td>
    <td>0.5691</td>
    <td>0.5930</td>
    <td>0.5901</td>
    <td>0.5900</td>
    <td>0.5994</td>
  </tr>
  <tr>
    <th>MLP Classifier</th>
    <td>0.5915</td>
    <td>0.5878</td>
    <td>0.5874</td>
    <td>0.5983</td>
    <td>0.5986</td>
    <td>0.5989</td>
    <td>0.5987</td>
    <td>0.6022</td>
  </tr>
  <tr>
    <th>K-Neighbors Classifier</th>
    <td>0.5549</td>
    <td>0.5546</td>
    <td>0.5547</td>
    <td>0.5602</td>
    <td>0.5700</td>
    <td>0.5680</td>
    <td>0.5677</td>
    <td>0.5773</td>
  </tr>
  <tr>
    <th>SGD Classifier</th>
    <td>0.5503</td>
    <td>0.5483</td>
    <td>0.5472</td>
    <td>0.5591</td>
    <td>0.5926</td>
    <td>0.5813</td>
    <td>0.5761</td>
    <td>0.5989</td>
  </tr>
  <tr>
    <th>Random Forest</th>
    <td>0.5520</td>
    <td>0.5523</td>
    <td>0.5520</td>
    <td>0.5552</td>
    <td>0.5671</td>
    <td>0.5675</td>
    <td>0.5671</td>
    <td>0.5702</td>
  </tr>
  <tr>
    <th>Decision Tree</th>
    <td>0.5376</td>
    <td>0.5380</td>
    <td>0.5361</td>
    <td>0.5370</td>
    <td>0.5398</td>
    <td>0.5400</td>
    <td>0.5396</td>
    <td>0.5425</td>
  </tr>
  <tr>
    <th>Gaussan NB</th>
    <td>0.5893</td>
    <td>0.5767</td>
    <td>0.5699</td>
    <td>0.5956</td>
    <td>0.5949</td>
    <td>0.5845</td>
    <td>0.5882</td>
    <td>0.6011</td>
  </tr
  <tr>
    <th>AdaBoost</th>
    <td>0.5848</td>
    <td>0.5798</td>
    <td>0.5784</td>
    <td>0.5923</td>
    <td>0.5885</td>
    <td>0.5869</td>
    <td>0.5871</td>
    <td>0.5945</td>
  </tr>
  <tr>
    <th>Gradient Boosting</th>
    <td>0.5838</td>
    <td>0.5799</td>
    <td>0.5790</td>
    <td>0.5912</td>
    <td>0.5861</td>
    <td>0.5856</td>
    <td>0.5858</td>
    <td>0.5912</td>
  </tr>
  <tr>
    <th>XGBoost</th>
    <td>0.5421</td>
    <td>0.5423</td>
    <td>0.5420</td>
    <td>0.5453</td>
    <td>0.5787</td>
    <td>0.5794</td>
    <td>0.5785</td>
    <td>0.5807</td>
  </tr>
  <tr>
    <th>CatBoost</th>
    <td>0.5836</td>
    <td>0.5808</td>
    <td>0.5805</td>
    <td>0.5906</td>
    <td>0.5946</td>
    <td>0.5941</td>
    <td>0.5942</td>
    <td>0.5994</td>
  </tr>
  <tr>
    <th>LightGBM</th>
    <td>0.5622</td>
    <td>0.5621</td>
    <td>0.5621</td>
    <td>0.5669</td>
    <td><b>0.5989</b></td>
    <td><b>0.5991</b></td>
    <td><b>0.5990</b></td>
    <td><b>0.6028</b></td>
  </tr>
</table>


## My Certificates of Related MLOps Courses
- [Coursera - Structuring Machine Learning Projects (February 02, 2022)](https://www.coursera.org/account/accomplishments/certificate/VV3K9H8C6TFK)
- [Coursera -  Introduction to Machine Learning in Production (September 21, 2021)](https://www.coursera.org/account/accomplishments/certificate/26DXRJR5KVZR)

