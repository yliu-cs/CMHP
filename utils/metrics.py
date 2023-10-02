from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calc_metrics(
    labels: list
    , preds: list
):
    acc = accuracy_score(labels, preds) * 100
    mac, wtd = {}, {}
    _mac = precision_recall_fscore_support(labels, preds, average="macro")
    for i, x in enumerate(["pre", "rec", "f1"]):
        mac[x] = _mac[i] * 100
    _wtd = precision_recall_fscore_support(labels, preds, average="weighted")
    for i, x in enumerate(["pre", "rec", "f1"]):
        wtd[x] = _wtd[i] * 100

    metrics = {
        "acc": acc
        , "mac": mac
        , "wtd": wtd
    }
    return metrics