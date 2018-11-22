archive = load_archive('saved_models/ner_xdom_wb_5k_wt_glove',weights_file='saved_models/ner_xdom_wb_5k_wt_glove/best.th')
logits = [archive.model.text_field_embedder.scalar_mix.scalar_parameters[i].item() for i in range(0,5)]
np.exp(logits)/np.exp(logits).sum()
