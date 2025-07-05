"""
This module contains utility functions
used for the training and evaluation of
the models developed in this project.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score
)

def evaluation_score(
    solution_set: pd.DataFrame, 
    test_set: pd.DataFrame,
    incl_jaccard: bool = False,
    jaccard_only: bool = False
) -> float:
    """
    This function computes the evaluation score
    based on the evaluation measure defined in the
    challenge description.

    Parameters:

    - solution_set: DataFrame containing only the
    model's predictions.
    - test_set: DataFrame containing the ground truth 
    labels.
    - jaccard: Boolean indicating whether to use Jaccard
    similarity for evaluation of item purchases.

    Returns:
    - score: The evaluation score as a float.
    """
    # Assure types are correct
    solution_set = solution_set.copy()
    test_set = test_set.copy()

    solution_set['session_id'] = solution_set['session_id'].astype(str)
    test_set['session_id'] = test_set['session_id'].astype(str)

    if incl_jaccard or jaccard_only:
        solution_set['item_id'] = solution_set['item_id'].astype(str)
        test_set['item_id'] = test_set['item_id'].astype(str)

    # Define evaluation metrics
    Sl = set(solution_set['session_id']) # Sessions in solution
    S = set(test_set['session_id']) # Sessions in test set
    Sb = set(test_set[test_set['session_purchase'] == 1]['session_id']) # Sessions with purchase in test set
    delta = len(Sb) / len(S) if len(S) > 0 else 0.0 # Score delta factor for sessions

    # Optional for Jaccard similarity
    As_grouped = {}
    Bs_grouped = {}
    if incl_jaccard or jaccard_only:
        As_df = solution_set # Predicted items in session
        Bs_df = test_set[test_set['item_purchase'] == 1] # True items
        As_grouped = As_df.groupby('session_id')['item_id'].apply(set).to_dict() # Predicted items
        Bs_grouped = Bs_df.groupby('session_id')['item_id'].apply(set).to_dict() # True items
    
    score: float = 0.0

    for s in Sl:
        if s in Sb:
            # If session is correct, add to score
            if not jaccard_only:
                score += delta
            if incl_jaccard or jaccard_only:
                # If using Jaccard similarity, calculate the score
                As: set = As_grouped.get(s, set())
                Bs: set = Bs_grouped.get(s, set())
                if As and Bs:
                    intersection = len(As.intersection(Bs))
                    union = len(As.union(Bs))
                    score += intersection / union if union > 0 else 0.0
        else:
            if not jaccard_only:
                # If session is incorrect, subtract delta
                score -= delta

    return score

def visualise_optimisation(
    model_name: str,
    scores: np.ndarray,
    thresholds: np.ndarray,
    optimal_threshold: float,
    size: tuple = (10, 6)
) -> None:
    """
    This function visualises the optimisation
    process by plotting the evaluation scores
    against the thresholds.

    Parameters:
    - model_name: Name of the model being evaluated.
    - scores: Array of evaluation scores for each threshold.
    - thresholds: Array of thresholds used in the evaluation.
    - optimal_threshold: The optimal threshold selected.
    """
    plt.figure(figsize=size)
    plt.plot(thresholds, scores, marker='o', label='Evaluation Score')
    plt.axvline(optimal_threshold, color='red', linestyle='--', label='Optimal Threshold')
    plt.title(f'Optimisation of Threshold for {model_name}')
    plt.xlabel('Threshold')
    plt.ylabel('Evaluation Score')
    plt.legend()
    plt.grid()
    plt.show()

def optimise_threshold(
    model_name: str,
    solution_set: pd.DataFrame, 
    test_set: pd.DataFrame,
    jaccard: bool = False,
    incl_jaccard: bool = False
) -> float:
    """
    This function evaluates the model's
    predictions and selects the optimal
    threshold for classification based on
    the evaluation score.

    Parameters:
    - model_name: Name of the model being evaluated.
    - solution_set: DataFrame containing the model's
    predictions.
    - test_set: DataFrame containing the ground truth
    labels.
    - jaccard: Boolean indicating whether to use Jaccard
    similarity for evaluation of item purchases.

    Returns:
    - A dictionary containing the optimal threshold
    and the evaluation score.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    stage = 'item_prob' if jaccard else 'session_prob'
    probs = solution_set[stage].values

    scores = np.array([
        evaluation_score(
            solution_set[solution_set[stage] >= t],
            test_set,
            jaccard,
            incl_jaccard
        ) for t in thresholds
    ])
    optimal_threshold = thresholds[np.argmax(scores)]

    # Evaluation metrics 

    y_pred = (probs >= optimal_threshold).astype(int)
    y_true = test_set['item_purchase' if jaccard else 'session_purchase'].to_numpy()
    
    auc = roc_auc_score(y_true, solution_set[stage])
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Print metrics
    print(f"--- Model: {model_name} ---")
    print(f"Optimal Threshold: {optimal_threshold:.2f}")
    print(f"Evaluation Score: {scores.max():.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Visualise optimisation
    visualise_optimisation(
        model_name,
        scores,
        thresholds,
        optimal_threshold
    )
    
    return optimal_threshold

def evaluate_stage(
    model_name: str,
    solution_set: pd.DataFrame, 
    test_set: pd.DataFrame,
    stage: int,
    stage_threshold: float,
):
    """
    This function evaluates the model's predictions
    for a specific stage (session or item) and prints
    the evaluation score along with other metrics.

    Parameters:
    - model_name: Name of the model being evaluated.
    - solution_set: DataFrame containing the model's
    predictions.
    - test_set: DataFrame containing the ground truth
    labels.
    - stage: The stage of evaluation (1 for session, 2 for item).
    """
    # Perform evaluation based on the stage

    if stage == 1:
        # Stage 1: Session classification
        score = evaluation_score(
            solution_set[solution_set['session_prob'] >= stage_threshold],
            test_set,
        )

        max_score = evaluation_score(
            test_set[test_set['session_purchase'] == 1],
            test_set,
        )

        y_pred = np.array([
            1 if x >= stage_threshold else 0 
            for x in solution_set['session_prob']
        ])

        y_true = np.array(test_set['session_purchase'].values)
    else:
        # Stage 2: Item classification
        score = evaluation_score(
            solution_set[solution_set['item_prob'] >= stage_threshold],
            test_set,
            True,
            True
        )

        max_score = evaluation_score(
            test_set[test_set['item_purchase'] == 1],
            test_set,
            True,
            True
        )

        y_pred = np.array([
            1 if x >= stage_threshold else 0 
            for x in solution_set['item_prob'].values
        ])

        y_true = np.array(test_set['item_purchase'].values)

        
    percentage_of_max = (score / max_score) * 100 if max_score > 0 else 0.0
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(
        y_true, 
        solution_set['session_prob'] if stage == 1 
        else solution_set['item_prob']
    )
    
    # Print metrics
    print(f"--- Phase {stage}: {model_name} ---")
    print(f"Evaluation Score: {score:.4f}")
    print(f"Max Possible Score: {max_score:.4f}")
    print(f"Percentage of Max Score: {percentage_of_max:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

def evaluate_model(
    solution_set: pd.DataFrame, 
    test_set: pd.DataFrame,
) -> None:
    """
    This function evaluates the model's predictions
    using the evaluation score and prints the score
    along with other metrics such as ROC AUC, precision,
    and recall.

    Parameters:
    - model_name: Name of the model being evaluated.
    - solution_set: DataFrame containing the model's
    FINAL predictions on item data.
    - test_set: DataFrame containing the ground truth
    labels.
    - jaccard: Boolean indicating whether to use Jaccard
    similarity for evaluation of item purchases.
    """
    # Calculate evaluation score
    score = evaluation_score(
        solution_set,
        test_set,
        incl_jaccard=True,
        jaccard_only=False
    )

    max_score = evaluation_score(
        test_set[test_set['session_purchase'] == 1],
        test_set,
        incl_jaccard=True,
        jaccard_only=False
    )

    percentage_of_max = (score / max_score) * 100 if max_score > 0 else 0.0

    # Evaluation metrics

    # Print metrics
    print(f"--- Overall Statistics ---")
    print(f"Evaluation Score: {score:.4f}")
    print(f"Max Possible Score: {max_score:.4f}")
    print(f"Percentage of Max Score: {percentage_of_max:.2f}%")