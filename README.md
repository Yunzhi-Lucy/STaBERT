# Classical Feature Embeddings Help in BERT-Based Human Mobility Prediction
## Abstract
Human mobility forecasting is crucial for disaster relief, city planning, and public health. However, existing models either only model location sequences or include time information merely as auxiliary input, thereby failing to leverage the rich semantic context provided by points of interest (POIs). To address this, we enrich a BERT-based mobility model with derived temporal descriptors and POI embeddings to better capture the semantics underlying human movement. We propose STaBERT (Semantic-Temporal aware BERT), which integrates both POI and temporal information at each location to construct a unified, semantically enriched representation of mobility. Experimental results show that STaBERT significantly improves prediction accuracy: for single-city prediction, the GEO-BLEU score improved from 0.34 to 0.75; for multi-city prediction, from 0.34 to 0.56.

### Usage example
    python3 train_task1_new_input.py --batch_size 8 --epochs 30 --embed_size 128 --layers_num 4 --heads_num 8 --dataset task1_dataset_new_input.csv --poi_dataset cell_POIcat.csv
    python3 val_task1_new_input.py --pth_file [file_path] --embed_size 128 --layers_num 4 --heads_num 8 --dataset task1_dataset_new_input.csv --poi_dataset cell_POIcat.csv
    
