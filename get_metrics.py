import json
import os

def get_metrics(path):
    val_metrics = []
    for epoch in range(0,100):
        try:
            filepath = os.path.join(path,"metrics_epoch_{0}.json".format(str(epoch)))
            data = json.load(open(filepath))
            val_metrics.append(data['validation_f1-measure-overall'])
        except Exception as e:
            return val_metrics
    
def display(a):
    for k in a:
        print(k)

ner_wb_5k_glove = get_metrics("saved_models/ner_wb_5k_glove")
display(ner_wb_5k_glove)
print("*"*100)
ner_xdom_wb_5k_wt_glove = get_metrics("saved_models/ner_xdom_wb_5k_wt_glove")
display(ner_xdom_wb_5k_wt_glove)



