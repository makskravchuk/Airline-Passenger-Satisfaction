Business needs: Predicting customer dissatisfaction to reduce revenue loss

The purpose of the simulation:
• Development of a predictive model
• Analysis of factors leading to customer dissatisfaction
• Development of strategies to prevent customer dissatisfaction

Success criteria:
• Accuracy of predictions
• Reduction in customer dissatisfaction
• Increase in revenue

Requirements

    python 3.7

    numpy==1.17.3
    pandas==1.1.5
    sklearn==1.0.0
    
Running:

    To run the demo, execute predict.py

    After running the script in that folder will be generated <prediction_results.csv> 
    The file has 'satisfaction_pred' column with the result value.

    The input is expected csv file in the folder data with a name <new_data.csv>. The file shoud have all features columns. 
    
Training a Model:

    Before you run the training script for the first time, you will need to create dataset. The file <train_data.csv> should contain all features columns and target for prediction satisfaction.
    After running the script the "param_dict.pickle"  and "finalized_model.saw" will be created.
    Run the training script train.py

    The model accuracy is 97%
