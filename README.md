# Machine Learning Model

So in this project, we developed a machine learning model that is able to classify brain tumor types based on MRI images. This model uses a Convolutional Neural Network (CNN) architecture that is specifically designed for classification tasks. This model has four convolutional layers equipped with BatchNormalization and MaxPooling. This model is trained using a dataset that has been preprocessed through augmentation and normalization, so that it can produce accurate predictions for brain tumor types. We also trained our model with 30 epochs and after that we obtained the following results:

![image](https://github.com/user-attachments/assets/d7511e56-9136-4e25-99fc-1c7f2dc33e27)

![image](https://github.com/user-attachments/assets/ba6a47ed-4d22-49f1-85b9-2e1170d98bbd)

# Model Evaluation on Testing Data
To evaluate the performance of the model in classifying brain tumor types, we used the Confusion Matrix and Classification Report.

## Here are the results:
![image](https://github.com/user-attachments/assets/525991b0-ac51-4ad2-bb65-b4cb1e8cdf00)

Classification Report:
| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Glioma      | 0.98      | 0.92   | 0.96     | 163     |
| Meningioma  | 0.92      | 0.88   | 0.90     | 165     | 
| Notumor     | 0.99      | 0.99   | 0.99     | 201     | 
| Pituitary   | 0.90      | 0.99   | 0.94     | 176     | 
| **Accuracy**|           |        | **0.96** | 705     |
| Macro Avg   | 0.96      | 0.96   | 0.96     | 705     |
| Weighted Avg| 0.96      | 0.96   | 0.96     | 705     |
