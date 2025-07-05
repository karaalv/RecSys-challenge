# RecSys Challenge 2015 Report 

*Alvin Karanja*

## 1. Introduction

This report outlines the approach taken to address the prediction task in the RecSys Challenge 2015. The challenge involves a two stage classification problem: first, identifying whether a user session will result in a purchase and second, predicting which specific items within the session will be purchased.

Model performance is assessed using a specialised evaluation metric designed for the competition. This metric rewards correctly identified purchase sessions and accurate item predictions, while penalising incorrect purchase predictions, making it essential to balance precision and recall effectively.

The report details the full modelling pipeline, including data preprocessing, feature engineering, model training and evaluation, and final model selection. Each stage is approached with a focus on both practical implementation and analytical soundness.

This challenge is particularly relevant to digital marketing analytics, as it reflects a real world scenario in which the objective is to anticipate user behaviour and make data driven recommendations. Solving such a task demands a strong understanding of consumer patterns, the ability to extract and engineer meaningful features, and the application of machine learning methods with rigour and interpretability to ensure reliable outcomes.

## 2. Data Overview

### 2.1 Dataset Description

The dataset used in this project originates from the 2015 ACM Recommender Systems Challenge $^{[1]}$, comprising large scale e-commerce data provided by a major European retailer. It captures six months of user interactions on the retailer’s website, including session level clickstream activity and purchase events.

There are two primary files in the dataset:

- `yoochoose-clicks.dat`: Contains the clickstream data, where each row corresponds to a user click on an item during a session. Each record includes the session ID, item ID, timestamp, and associated metadata.
- `yoochoose-buys.dat`: Contains purchase events, with each record specifying the session ID, item ID, timestamp, and additional purchase-related fields.

The dataset is sizable, featuring over 33 million click records and more than 1 million purchase records. Its scale and granularity make it well suited for building and evaluating predictive models in a realistic setting.

Given its origin, the dataset is both credible and highly relevant for this task, offering real world behavioural data that closely mirrors the challenges faced in digital marketing and recommender systems.

### 2.2 Data Cleaning and Preprocessing

Before conducting any analysis or feature engineering, an initial exploratory data analysis (EDA) was performed to understand the structure and characteristics of the dataset. This included inspecting data types, checking for missing values, and reviewing basic statistics.

The dataset was found to be relatively clean, with no significant missing values or anomalies. However, during the inspection of the purchase file (`yoochoose-buys.dat`), some records that appeared to be duplicate were identified. Further investigation confirmed, based on the dataset documentation, that these duplicates were likely intentional and preserved to reflect the real world nature of the challenge data.

Given the challenge’s focus on predicting purchases at the session level, no extensive data cleaning was conducted, as the objective was to work within the constraints of the provided data. Preprocessing efforts were instead minimal and tailored toward enabling feature engineering and model development.

To facilitate easier manipulation and analysis, the original data files were saved as CSV files with appropriate column headers.

A more detailed EDA, including further discussions is provided in **Appendix A**.

## 3. General Approach

The prediction task consists of two components: first, determining whether a session will result in a purchase; and second, identifying which items within the session will be purchased. Accordingly, the problem is framed as a two stage classification task.

- **Stage 1**: Classify sessions as either purchase or non-purchase.
- **Stage 2**: For sessions predicted to result in a purchase, classify items within the session to determine which are likely to be bought.

This staged approach enables each model to specialise in a distinct aspect of the task, improving both interpretability and predictive performance.

To inform the design of this solution, a review of previous work on similar problems, including published approaches to the RecSys Challenge $^{[2]}$, was conducted. Notable techniques drawn from prior research include threshold optimisation tailored to the challenge’s evaluation metric, and fine tuned item-level classifiers trained exclusively on purchase sessions.

While inspired by past methodologies, the feature engineering, model implementation, and optimisation strategies adopted in this project are independently developed, with a focus on maximising performance under the challenge specific evaluation metric.

## 4. Feature Engineering

This section outlines the feature engineering process for both stages of the classification pipeline. It includes a discussion of the features selected, as well as an analysis of their distributions and relationships with the target variable.

A more detailed examination of the engineered features, accompanied by visualisations and statistical summaries, can be found in **Appendix B**.

### 4.1 Feature Selection Overview

Feature selection was based solely on the information available in the clickstream data. The purchase data was used exclusively for labelling sessions as purchases and was not used to derive any additional input features.

The raw clickstream data included the following attributes:

- `session_id`: Unique identifier for each user session
- `item_id`: Unique identifier for each item clicked
- `timestamp`: Time of the click event
- `category`: Category of the item clicked

Derived features were informed by behavioural intuition and common patterns observed in e-commerce settings. These included metrics such as session duration, item click frequency, and temporal activity patterns, factors typically indicative of user intent and engagement.

The subsequent sections detail the features engineered for each classification stage, along with their definitions and the rationale behind their inclusion.

### 4.2 Item Classification Features

The item level features were derived from aggregating the clickstream data by both `session_id` and `item_id`. This aggregation enabled the computation of item specific metrics within each session, forming the basis for classifying which items are likely to be purchased.

The feature engineering process focused on capturing behavioural signals indicative of purchase intent, as well as temporal and seasonal patterns in user activity. The final set of features used for item level classification includes:

- `item_clicks`: Total number of clicks on the item within the session, serving as a proxy for user interest.
- `item_time`: Estimated time spent on the item, calculated as the cumulative time between successive clicks on the same item.
- `day_of_week`: Binary indicator of whether the session occurred on a weekday or weekend, reflecting possible shifts in user behaviour across the week.
- `hour_sin` / `hour_cos`: Sine and cosine transformations of the hour of the first item click, used to encode cyclical patterns in time-of-day behaviour without introducing artificial discontinuities $^{[3]}$.
- `item_categories`: Number of unique categories associated with the item in the session, representing the diversity of content engagement.
- `item_propensity`: Historical likelihood of an item being purchased, computed as the ratio of purchase sessions to total sessions in which the item was clicked.

To address the highly skewed, long tail distributions observed in features such as `item_clicks`, `item_time`, and `item_categories`, a log(1 + x) transformation was applied. This helps reduce the influence of extreme values, a common characteristic in e-commerce datasets where a small number of items dominate user attention. After transformation, these features were standardised using z-score normalisation to ensure consistent scaling across inputs which is an important consideration for many machine learning algorithms.

The propensity feature (`item_propensity`) plays a particularly significant role, as it integrates historical purchase information to enhance predictive power while aligning with the challenge’s evaluation metric. To prevent data leakage, the propensity was computed using only the training set. For items not seen during training, the global mean propensity was used as a fallback to ensure coverage for all prediction cases.

Exploratory analysis revealed that features such as `item_clicks`, `item_time`, and `item_categories` were positively correlated with the likelihood of purchase. However, the long-tail nature of the data introduces modelling challenges, as it means the model may not be able to learn sufficiently given the low proportion of features that may be indicative of purchase intent. This skewness is typical in e-commerce environments, where a small number of items receive a disproportionate share of user interactions.

For a more detailed examination of these features including visualisations, statistical summaries, and a tabular overview of features refer to **Appendix B**.

### 4.3 Session Classification Features

The first stage of the classification pipeline focuses on predicting whether a user session will result in a purchase. To support this task, features were engineered to capture session level characteristics that are indicative of purchase intent.

Session level features were derived by aggregating the clickstream data by `session_id`, enabling the calculation of behavioural metrics for each user session. The features include:

- `session_length`: Total duration of the session, computed as the time difference between the first and last click.
- `session_clicks`: Total number of clicks within the session, serving as a measure of user engagement.
- `unique_items`: Count of distinct items clicked, reflecting the variety of items explored.
- `unique_categories`: Number of distinct item categories clicked, indicating the diversity of user interest.
- `day_of_week`: Binary indicator denoting whether the session took place on a weekday or weekend.
- `hour_sin` / `hour_cos`: Sine and cosine transformations of the hour of the first click in the session, capturing cyclical patterns in user activity.
- `max_item_propensity`: Maximum historical propensity score among items clicked in the session, representing the highest individual purchase likelihood.
- `avg_item_propensity`: Average propensity score of all items clicked, providing a general measure of purchase probability across the session.

As with item level features, several session features such as `session_length` and `session_clicks` exhibited long-tail distributions. To mitigate the influence of extreme values, a log(1 + x) transformation was applied, followed by z-score normalisation to ensure scale consistency across features.

Temporal features were derived using the timestamp of the first click in the session to represent the session’s time of day. This approach was selected to capture behavioural patterns tied to when a user initiates a session, which may offer predictive value.

The inclusion of `max_item_propensity` and `avg_item_propensity` allows the model to incorporate historical purchase likelihood at the session level. Similar to the item level features, these propensity scores were calculated using only the training set to prevent data leakage.

Exploratory analysis showed that `session_length`, `session_clicks`, and `unique_items` were positively associated with purchase likelihood. However, as with item level features, these metrics displayed significant skew which may impact the model's ability to learn effectively.

A more detailed examination of these features, including visualisations and statistical summaries, is provided in **Appendix B**.

## 5. Model Training and Selection

This section outlines the model training and selection process for both stages of the classification pipeline. It includes a discussion of the models used, the evaluation metrics employed, and the approach taken to optimise model performance. For a detailed analysis of model training and evaluation, refer to **Appendix C**.

### 5.1 Modelling Overview

Given the binary nature of the task, predicting whether a session results in a purchase or not, and whether an item within that session is purchased, it is appropriate to frame the problem as a classification task rather than a regression problem. Moreover, the use of probabilistic classification models is particularly well suited, as they capture the inherent uncertainty in user behaviour and allow for flexible decision making through the adjustment of classification thresholds. This flexibility is especially valuable in the context of the challenge’s evaluation metric, which rewards correct predictions and penalises false positives, making threshold optimisation a key part of performance tuning.

For each classification stage, we evaluate the performance of three core model types:

- Logistic Regression
- Gradient Boosting Machine (GBM)
- Feedforward Neural Network (FNN)

Logistic Regression is used as a baseline due to its simplicity, interpretability, and efficiency in binary classification tasks. However, its linear nature limits its ability to capture complex feature interactions and non-linear relationships, which may be crucial in this setting.

To overcome these limitations, we implement a GBM model, which is well suited for structured data and is capable of modelling non-linear patterns and feature interactions effectively. Additionally, we develop a Feedforward Neural Network (FNN) with a basic architecture, comprising an input layer, a single hidden layer with ReLU activation, and a sigmoid activated output layer. The model is trained using binary cross-entropy loss, appropriate for binary classification problems. Note that we leave optimising the architecture of the FNN and GBM models out of scope for this task, focusing instead on feature engineering and model evaluation of basic architectures to demonstrate the feasibility of the approach.

The dataset is split into training, validation, and test sets to facilitate model evaluation. The training set is used to fit the models, the validation set is employed for hyperparameter tuning and threshold selection, and the test set is reserved for final performance evaluation. To prevent data leakage, splitting is performed using `session_id`, ensuring that the same session does not appear across multiple sets. This method was preferred over temporal splitting, as it preserves within session temporal information. However, future work may explore time based splits to better simulate generalisation to unseen time periods.

Threshold selection plays a key role in aligning model outputs with the challenge’s evaluation metric, which penalises false positives and rewards accurate purchase predictions. Therefore, we optimise the classification threshold on the validation set.

For session classification, the challenge metric imposes a penalty for false positives. This necessitates careful threshold tuning to **maximise recall** ensuring we correctly identify as many purchase sessions as possible, while maintaining sufficient precision to avoid over predicting purchases and incurring penalties.

In contrast, the item classification component, evaluated via a Jaccard based metric, does not penalise false positives explicitly. However, predicting non-purchased items reduces the Jaccard score. Consequently, the item model is **optimised for precision**, ensuring that predicted items are likely to be correct, while maintaining enough recall to avoid missing actual purchases.

### 5.2 Session Classification

For the session classification task, we implemented the three core models on the session level features described in *Section 4.3*.

To address the class imbalance, where purchase sessions constitute a small fraction of the dataset, we applied **upsampling** to the training set, increasing the representation of purchase sessions. This ensured that the models could learn from both classes effectively. Earlier training runs without upsampling resulted in poor model performance, as the models failed to generalise to the underrepresented purchase class.

The **GBM model** was implemented using the `LightGBM` library, chosen for its efficiency with large datasets and strong performance on structured data. It was trained using default hyperparameters, as hyperparameter optimisation was considered out of scope for this task.

The **FNN model** followed a simple architecture with one hidden layer of 8 neurons, a batch size of 2048, and training over 50 epochs. The model was trained using the **Adam** optimiser with a learning rate of 0.001 and **binary cross-entropy loss**. As with GBM, no hyperparameter tuning was conducted, as the focus of this task was on feature engineering and evaluation rather than exhaustive model tuning.

All models were evaluated on the validation set using the **challenge-specific evaluation metric**, with classification thresholds optimised to maximise performance. The results are summarised below:

| Model               | Evaluation Score | Optimised Threshold | Precision | Recall  |
|---------------------|------------------|---------------------|----------|--------|
| Logistic Regression | -7.8810          | 0.99                | 0.3826   | 0.0034 |
| GBM                 | 2.8107           | 0.95                | 0.5235   | 0.0084 |
| FNN                 | -0.7165          | 0.99                | 0.3333   | 0.0002 |

The **GBM model** significantly outperformed the other two approaches, achieving a positive evaluation score of **2.8107**. In contrast, the **Logistic Regression** model performed poorly, with a negative score of **-7.8810**, suggesting it was unable to capture the underlying patterns in the data. The **FNN model** also underperformed, yielding a score of **-0.7165**.

Based on these results, the **GBM model was selected as the final model** for the session classification task.

Generally, the thresholds for each model were set quite high, reflecting the challenge’s evaluation metric which penalises false positives. This approach prioritised recall, ensuring that as many purchase sessions as possible were identified, even at the cost of precision. From the optimum thresholds, it is evident that it was only appropriate to classify a session as a purchase if the model was highly confident, as indicated by the high thresholds (0.95 for GBM and 0.99 for Logistic Regression and FNN). This is likely due to the underlying model not being able to reliably predict purchase sessions, as indicated by the low recall scores.

### 5.3 Item Classification

For the item classification task, we used the features described in *Section 4.2*, which were engineered at the item level within each session. As with the session classification stage, we evaluated the same three core models.

To address the class imbalance, **upsampling** was applied to the training set similar to the session classification task. This ensured that the models could learn effectively from both classes, as purchase items are a small fraction of the total items clicked.

In line with the challenge’s evaluation metric (based on the Jaccard index), models were trained only on **sessions that resulted in a purchase**. This focused the learning process on relevant sessions and allowed the models to better identify purchase patterns. Although this setup may lead to **overly optimistic predictions**, as the model does not observe non-purchase sessions, this is acceptable within the evaluation framework, which does not explicitly penalise false positives in the item prediction stage. Therefore, the models were optimised to maximise **precision** while maintaining sufficient **recall**.

During threshold optimisation, we observed that the Jaccard component of the evaluation score increases steadily with higher thresholds up to a peak, after which it declines. This behaviour indicates the presence of an optimal classification threshold that balances precision and recall effectively.

The performance of each model on the validation set is summarised below:

| Model               | Evaluation Score | Optimised Threshold | Precision | Recall  |
|---------------------|------------------|---------------------|-----------|--------|
| Logistic Regression | 44886.5985       | 0.87                | 0.5604    | 0.9242 |
| GBM                 | 46448.7453       | 0.90                | 0.6004    | 0.9090 |
| FNN                 | 46336.5967       | 0.88                | 0.5908    | 0.9057 |

All models achieved high recall, which is expected given the optimistic nature of predictions when training only on purchase sessions. More importantly, precision scores were successfully tuned to higher values, indicating that the models are capable of identifying items with a higher likelihood of being purchased. In contrast, earlier experiments that included non-purchase sessions during training resulted in lower evaluation scores, suggesting reduced model effectiveness.

Notably, the **FNN model** demonstrated performance comparable to the GBM model despite its simpler architecture. This suggests that even a basic neural network was able to capture underlying behavioural patterns effectively. Further improvements may be possible through more sophisticated tuning, although this was beyond the scope of the current task.

Ultimately, the **GBM model was selected as the final model** for the item classification task, as it achieved the highest evaluation score and the best precision, making it most suitable for accurately predicting purchased items within sessions.

## 6. Evaluation and Results

This section presents the final model selection and performance evaluation on the test set. Note that the final results for the prediction task are saved in the `results/` folder in CSV format, with separate files for session and item predictions.

### 6.1 Final Model Selection

Based on the results from both stages of the classification task, the following models were selected as the final models:

- **Session Classification**: Gradient Boosting Machine (GBM)  
- **Item Classification**: Gradient Boosting Machine (GBM)

Although the **FNN model** performed comparably in the item classification task, the **GBM model** was selected due to its higher evaluation score and superior precision. The FNN model was implemented with a simple architecture, and while its results were promising, further optimisation could potentially yield even better performance.

It is also worth noting that this study evaluated only a limited set of models. Additional improvements could likely be achieved through more advanced model architectures, hyperparameter optimisation, and ensemble techniques. However, as the primary focus of this task was on **feature engineering and model evaluation**, comprehensive model tuning and architectural experimentation were beyond the intended scope and are left for future work.

### 6.2 Model Performance

The final prediction algorithm was evaluated on the test set, comprising of one third of the original dataset. During inference, the **session classification model** was first used to determine whether a session would lead to a purchase. For sessions predicted as purchases (with probabilities exceeding the optimised threshold), the **item classification model** was then applied to predict which items within those sessions were likely to be bought.

The following performance metrics were observed for the final models on the test set:

```plaintext
--- Phase 1: Gradient Boosting ---
Evaluation Score: 20.3226
Max Possible Score: 9352.1972
Percentage of Max Score: 0.22%
Precision: 0.5631
Recall: 0.0097

--- Phase 2: Gradient Boosting ---
Evaluation Score: 11989.2244
Max Possible Score: 169809.0000
Percentage of Max Score: 7.06%
Precision: 0.9148
Recall: 0.0499

--- Overall Statistics ---
Evaluation Score: 29516.2244
Max Possible Score: 275633.7059
Percentage of Max Score: 10.71%
```

The final model achieved an overall evaluation score of *29516.2244*, approximately *10.71%* of the maximum possible score. This indicates that while the model captures some predictive signal in the data, there remains significant room for improvement.

Notably, the fine-tuned item classification model yielded a high precision of *0.9148*, indicating that predicted purchases were generally accurate. However, its recall of *0.0499* reveals that a large number of actual purchases were missed. This limitation is compounded by the performance of the session classification model, which achieved only *0.0097* recall, highlighting that the majority of purchase sessions were not identified in the first stage.

The optimised threshold for session classification played a critical role in maintaining a positive contribution to the evaluation metric by balancing false positives and negatives. However, the model remains highly conservative, prioritising precision at the cost of recall, likely due to the inherent difficulty of detecting purchase intent from clickstream data alone.

To enhance performance, future work could prioritise improvements to the session classification stage, including:

- More advanced feature engineering
- Incorporation of additional data sources
- Use of more expressive model architectures
- Application of ensemble or hybrid modelling approaches

These enhancements could increase the model’s ability to identify purchase sessions effectively and unlock higher gains in the overall evaluation metric.

### 7. Conclusion

This report presented the approach taken to solve the two-stage classification task in the RecSys Challenge 2015, which involved predicting user sessions that would lead to purchases and identifying the specific items purchased within those sessions. The solution focused on engineered features derived from clickstream data and the application of probabilistic classification models.

Gradient Boosting Machines (GBMs) were selected as the final models for both stages of the pipeline. On the test set, the combined approach achieved an overall evaluation score of approximately 10.71% of the maximum possible score. While this demonstrates that the models were able to capture meaningful predictive signals, it also reflects the complexity of the task. For comparison, the top performing solution in the original challenge achieved a score close to 50% of the maximum, indicating that there is substantial room for improvement.

The item classification stage performed relatively well, especially in terms of precision, suggesting that once a purchase session is correctly identified, the model can accurately predict the items likely to be bought. However, the session classification stage proved to be a key bottleneck, with low recall severely limiting the number of sessions passed to the item model. This challenge illustrates the inherent difficulty of detecting purchase intent purely from clickstream data, where user behaviour is often ambiguous and highly variable.

These findings underscore the need for further work in this area. Future improvements could involve more advanced feature engineering, the use of additional data sources (such as user or product metadata), and the application of more sophisticated modelling techniques, including deep learning or ensemble based approaches. Enhancing threshold optimisation and integrating temporal context could also contribute to better predictive performance.

Overall, this project highlights the difficulty of accurately modelling user behaviour in real-world e-commerce environments, where intent is complex, subtle, and not always evident in raw interaction data. Nonetheless, the work lays a strong foundation for further exploration and demonstrates the potential of data driven approaches to improving recommender systems and digital marketing strategies.

## *References*

1. Ben-Shimon, D., Tsikinovsky, A., Friedmann, M., Shapira, B., Lior Rokach and Hoerle, J. (2015). RecSys Challenge 2015 and the YOOCHOOSE Dataset. doi:https://doi.org/10.1145/2792838.2798723.
2. Romov, P. and Sokolov, E. (2015). RecSys Challenge 2015. doi:https://doi.org/10.1145/2813448.2813510.
3. Guo, C. and Berkhahn, F., 2016. Entity embeddings of categorical variables. arXiv preprint arXiv:1604.06737. Available at: https://arxiv.org/abs/1604.06737 .

## *Appendix*

### Appendix A: Data EDA Notebook

This notebook contains a more detailed EDA of the RecSys Challenge 2015 dataset. The notebook is located in the `src` directory of the submission folder.
**Location**: `src/eda.ipynb`

### Appendix B: Feature Engineering and Analysis Notebook

This notebook contains a more detailed analysis of the features used in the prediction task. The notebook is located in the `src` directory of the submission folder.
**Location**: `src/analysis.ipynb`

### Appendix C: Model Training and Evaluation Notebook

This notebook contains a more detailed analysis of the model training and evaluation process. The notebook is located in the `src` directory of the submission folder.
**Location**: `src/training.ipynb`