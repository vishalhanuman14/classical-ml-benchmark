from utils import load_training_set, load_test_set


if __name__ == '__main__':
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2

    percentage_positive_instances_test = 0.2
    percentage_negative_instances_test = 0.2

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    # ══════════════════════════════════════════════════════════════════════
    #  HELPER FUNCTIONS
    # ══════════════════════════════════════════════════════════════════════

    import math
    import random
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def train(pos_train, neg_train, vocab, alpha=1.0):
        """
        Estimate Pr(y) and Pr(w | y) from training data.
        Laplace smoothing parameter alpha (0 = no smoothing).
        Returns a model dict.
        """
        N_pos = len(pos_train)
        N_neg = len(neg_train)
        N     = N_pos + N_neg

        log_prior_pos = math.log(N_pos / N)
        log_prior_neg = math.log(N_neg / N)

        freq_pos = {}
        freq_neg = {}

        for doc in pos_train:
            for w in doc:
                freq_pos[w] = freq_pos.get(w, 0) + 1

        for doc in neg_train:
            for w in doc:
                freq_neg[w] = freq_neg.get(w, 0) + 1

        total_pos = sum(freq_pos.values())
        total_neg = sum(freq_neg.values())
        V         = len(vocab)

        denom_pos = total_pos + alpha * V
        denom_neg = total_neg + alpha * V

        return {
            'log_prior_pos': log_prior_pos,
            'log_prior_neg': log_prior_neg,
            'freq_pos':      freq_pos,
            'freq_neg':      freq_neg,
            'denom_pos':     denom_pos,
            'denom_neg':     denom_neg,
            'alpha':         alpha,
            'vocab':         vocab,
        }

    def classify_log(doc, model):
        """
        Classify using log-probabilities (Eq. 7).
        Returns 'positive' or 'negative'.
        """
        alpha     = model['alpha']
        freq_pos  = model['freq_pos']
        freq_neg  = model['freq_neg']
        denom_pos = model['denom_pos']
        denom_neg = model['denom_neg']

        log_prob_pos = model['log_prior_pos']
        log_prob_neg = model['log_prior_neg']

        for w in doc:
            log_prob_pos += math.log((freq_pos.get(w, 0) + alpha) / denom_pos)
            log_prob_neg += math.log((freq_neg.get(w, 0) + alpha) / denom_neg)

        return 'positive' if log_prob_pos >= log_prob_neg else 'negative'

    def classify_raw(doc, model):
        """
        Classify using raw product of probabilities (Eq. 1) - Question 1.
        """
        freq_pos  = model['freq_pos']
        freq_neg  = model['freq_neg']
        denom_pos = model['denom_pos']
        denom_neg = model['denom_neg']
        alpha     = model['alpha']

        prob_pos = math.exp(model['log_prior_pos'])
        prob_neg = math.exp(model['log_prior_neg'])

        for w in doc:
            prob_pos *= (freq_pos.get(w, 0) + alpha) / denom_pos
            prob_neg *= (freq_neg.get(w, 0) + alpha) / denom_neg

        return 'positive' if prob_pos >= prob_neg else 'negative'

    def evaluate(pos_test, neg_test, model, use_log=True):
        """
        Classify all test documents and return accuracy, precision, recall,
        and confusion matrix counts (TP, FP, TN, FN).
        Positive class = positive reviews.
        """
        classify = classify_log if use_log else classify_raw

        TP = FP = TN = FN = 0

        for doc in pos_test:
            pred = classify(doc, model)
            if pred == 'positive':
                TP += 1
            else:
                FN += 1

        for doc in neg_test:
            pred = classify(doc, model)
            if pred == 'negative':
                TN += 1
            else:
                FP += 1

        total     = TP + FP + TN + FN
        accuracy  = (TP + TN) / total if total else 0
        precision = TP / (TP + FP)    if (TP + FP) else 0
        recall    = TP / (TP + FN)    if (TP + FN) else 0

        return {
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'accuracy':  accuracy,
            'precision': precision,
            'recall':    recall,
        }

    def print_results(label, results):
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")
        TP, FP = results['TP'], results['FP']
        FN, TN = results['FN'], results['TN']
        print(f"  Accuracy  : {results['accuracy']:.4f}")
        print(f"  Precision : {results['precision']:.4f}")
        print(f"  Recall    : {results['recall']:.4f}")
        print(f"\n  Confusion Matrix (rows=Actual, cols=Predicted):")
        print(f"                  Pred Pos   Pred Neg")
        print(f"  Actual Pos       {TP:6d}     {FN:6d}")
        print(f"  Actual Neg       {FP:6d}     {TN:6d}")

    # ══════════════════════════════════════════════════════════════════════
    #  Q1 - Standard Naive Bayes (raw product, alpha=1, 20% data)
    #       Uses pos_train / neg_train / vocab / pos_test / neg_test
    #       already loaded above by the default starter code.
    # ══════════════════════════════════════════════════════════════════════
    model_q1 = train(pos_train, neg_train, vocab, alpha=1.0)
    res_q1   = evaluate(pos_test, neg_test, model_q1, use_log=False)
    print_results("Q1 - Standard NB (raw product, alpha=1, 20% data)", res_q1)

    # ══════════════════════════════════════════════════════════════════════
    #  Q2 - Log-probability NB + Laplace smoothing; sweep alpha
    #       20% train, 20% test  (reuses same split as Q1)
    # ══════════════════════════════════════════════════════════════════════
    print("\n\nQ2 - Sweeping alpha (log-prob NB, 20% data) ...")
    alphas     = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    accuracies = []

    for a in alphas:
        m   = train(pos_train, neg_train, vocab, alpha=a)
        res = evaluate(pos_test, neg_test, m, use_log=True)
        accuracies.append(res['accuracy'])
        print(f"  alpha={a:<8}  accuracy={res['accuracy']:.4f}")

    # Full metrics for alpha=1
    model_q2 = train(pos_train, neg_train, vocab, alpha=1.0)
    res_q2   = evaluate(pos_test, neg_test, model_q2, use_log=True)
    print_results("Q2 - Log-prob NB, alpha=1, 20% data", res_q2)

    # Plot accuracy vs alpha
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, accuracies, marker='o', linewidth=2, color='steelblue')
    ax.set_xscale('log')
    ax.set_xlabel('alpha (log scale)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Q2: Accuracy vs Laplace Smoothing Parameter alpha', fontsize=13)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    for a, acc in zip(alphas, accuracies):
        ax.annotate(f'{acc:.3f}', (a, acc), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig('q2_alpha_sweep.png', dpi=150)
    print("\n  Plot saved -> q2_alpha_sweep.png")

    best_alpha = alphas[accuracies.index(max(accuracies))]
    print(f"\n  Best alpha = {best_alpha}  (accuracy={max(accuracies):.4f})")

    # ══════════════════════════════════════════════════════════════════════
    #  Q3 - 100% train, 100% test; best alpha; log-prob NB
    # ══════════════════════════════════════════════════════════════════════
    print("\n\nLoading data for Q3 (100% / 100%) - this may take a minute ...")
    random.seed(42)
    pos_train_full, neg_train_full, vocab_full = load_training_set(1.0, 1.0)
    pos_test_full,  neg_test_full              = load_test_set(1.0, 1.0)
    print(f"  Train: {len(pos_train_full)} pos, {len(neg_train_full)} neg | "
          f"Test: {len(pos_test_full)} pos, {len(neg_test_full)} neg | "
          f"Vocab: {len(vocab_full)}")

    model_q3 = train(pos_train_full, neg_train_full, vocab_full, alpha=best_alpha)
    res_q3   = evaluate(pos_test_full, neg_test_full, model_q3, use_log=True)
    print_results(f"Q3 - Log-prob NB, alpha={best_alpha}, 100% data", res_q3)

    # ══════════════════════════════════════════════════════════════════════
    #  Q4 - 30% train, 100% test; best alpha; log-prob NB
    # ══════════════════════════════════════════════════════════════════════
    print("\n\nLoading data for Q4 (30% train / 100% test) ...")
    random.seed(42)
    pos_train_30, neg_train_30, vocab_30 = load_training_set(0.3, 0.3)
    print(f"  Train: {len(pos_train_30)} pos, {len(neg_train_30)} neg | "
          f"Vocab: {len(vocab_30)}")

    model_q4 = train(pos_train_30, neg_train_30, vocab_30, alpha=best_alpha)
    res_q4   = evaluate(pos_test_full, neg_test_full, model_q4, use_log=True)
    print_results(f"Q4 - Log-prob NB, alpha={best_alpha}, 30% train / 100% test", res_q4)

    print("\n  Q3 vs Q4 comparison:")
    for metric in ('accuracy', 'precision', 'recall'):
        delta = res_q4[metric] - res_q3[metric]
        print(f"    {metric:<10}  Q3={res_q3[metric]:.4f}  Q4={res_q4[metric]:.4f}  "
              f"delta={delta:+.4f}")

    # ══════════════════════════════════════════════════════════════════════
    #  Q5 - Written question (no computation)
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "="*55)
    print("  Q5 - Discussion (answer in your written report)")
    print("="*55)
    print("  Consider: is accuracy, precision, or recall most")
    print("  important for classifying product reviews?")
    print("  Hint: think about what false positives/negatives")
    print("  mean for a business displaying review sentiment.")

    # ══════════════════════════════════════════════════════════════════════
    #  Q6 - Unbalanced dataset: 10% pos, 50% neg train; 100% test
    # ══════════════════════════════════════════════════════════════════════
    print("\n\nLoading data for Q6 (10% pos / 50% neg train, 100% test) ...")
    random.seed(42)
    pos_train_unbal, neg_train_unbal, vocab_unbal = load_training_set(0.1, 0.5)
    print(f"  Train: {len(pos_train_unbal)} pos, {len(neg_train_unbal)} neg | "
          f"Vocab: {len(vocab_unbal)}")

    model_q6 = train(pos_train_unbal, neg_train_unbal, vocab_unbal, alpha=best_alpha)
    res_q6   = evaluate(pos_test_full, neg_test_full, model_q6, use_log=True)
    print_results(f"Q6 - Unbalanced train (10% pos / 50% neg), alpha={best_alpha}", res_q6)

    print("\n  Q4 (balanced 30%) vs Q6 (unbalanced 10/50%) comparison:")
    for metric in ('accuracy', 'precision', 'recall'):
        delta = res_q6[metric] - res_q4[metric]
        print(f"    {metric:<10}  Q4={res_q4[metric]:.4f}  Q6={res_q6[metric]:.4f}  "
              f"delta={delta:+.4f}")

    print("\n\nAll experiments complete.")
