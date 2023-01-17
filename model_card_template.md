# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Person: Amandeep Singh
Model Version: 0.0.1
Model Type: Classification

Model is trained on publicly available census data. Training data contains information about education, workclass, demographics which is used to classify the sample into higher or lower salary
bands

## Intended Use

Use of the model is classify the sample into lower or higher salary bands based on provided background information

## Training Data

Training data is based on publicly available census data. Data contains information about the workclass, education, age, sex, martital status and enconomic status

Data is split into 70:30 ratio 70% data is used for model training

## Evaluation Data

Remaining 30% data is used for model Evaluattion

## Metrics
_Please include the metrics used and your model's performance on those metrics._

Model is evaluated based on below three metrices

- Percision: Current model's performance is >= .9
- Recall: Current model's performance is >= .9
- FBeta: Current model's performance is >= .9

## Ethical Considerations

Purpose of the model is only for MLOps evaluation purpose, it is not intended to use for unentical purpose in which a person's status is determined based on his race, sex and enthenticity

## Caveats and Recommendations

Model is trained on limited data and performance can degrade in future depending on change in socio economic variables, so it is recommended to monitor the performance and retrain the model if required.
