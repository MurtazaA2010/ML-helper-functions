best_score = 0
best_w = None

for w1 in [i/10 for i in range(1,10)]:
    for w2 in [i/10 for i in range(1,10)]:
        w3 = 1 - w1 - w2
        if w3 <= 0:
            continue
        preds = w1*xgb_preds + w2*lgb_preds + w3*cat_preds
        score = evaluate(preds)  
        if score > best_score:
            best_score = score
            best_w = (w1, w2, w3)
