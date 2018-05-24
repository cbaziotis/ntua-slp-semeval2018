trial = open("us_trial.text", "r").readlines()
trial = set([line.rstrip().lower() for line in trial])

with open("us_train.ids", "r") as ids, \
        open("us_train_dedup.ids", "w") as ids_dedup, \
        open("us_train.text", "r") as text, \
        open("us_train_dedup.text", "w") as text_dedup, \
        open("us_train.labels", "r") as labels, \
        open("us_train_dedup.labels", "w") as labels_dedup:
    for i, (_id, _label, _text) in enumerate(zip(ids, labels, text)):
        if _text.rstrip().lower() not in trial:
            ids_dedup.write(_id)
            text_dedup.write(_text)
            labels_dedup.write(_label)
            print(i)
